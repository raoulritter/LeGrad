import math
import types
import torch
import sys
import gc
import inspect
import torch.nn as nn
import torch.nn.functional as F
from torchvision.transforms import Compose, Resize, InterpolationMode
import open_clip
import pdb
from open_clip.transformer import VisionTransformer
from open_clip.timm_model import TimmModel
from einops import rearrange

from .utils_cogvlm import hooked_resblock_forward, \
    hooked_attention_forward, \
    hooked_resblock_timm_forward, \
    hooked_attentional_pooler_timm_forward, \
    vit_dynamic_size_forward, \
    min_max, \
    hooked_torch_multi_head_attention_forward


def print_vram_usage():
    allocated = torch.cuda.memory_allocated() / 1024 ** 3  # Convert bytes to GB
    reserved = torch.cuda.memory_reserved() / 1024 ** 3    # Convert bytes to GB
    print(f"VRAM Usage - Allocated: {allocated:.2f} GB, Reserved: {reserved:.2f} GB")
    
    
class LeWrapper(nn.Module):
    """
    Wrapper around OpenCLIP to add LeGrad to OpenCLIP's model while keeping all the functionalities of the original model.
    """

    # COPY MODEL
    def __init__(self, model, layer_index=-2):
        super(LeWrapper, self).__init__()
        # ------------ copy of model's attributes and methods ------------
        for attr in dir(model):
            if not attr.startswith('__'):
                setattr(self, attr, getattr(model, attr))

        # ------------ activate hooks & gradient ------------
        self._activate_hooks(layer_index=layer_index)
        
            
    def _activate_hooks(self, layer_index):
        # ------------ identify model's type ------------
        print('Activating necessary hooks and gradients ....')
            
        if hasattr(self.model, 'vision'):
            
            # Store the original forward method for comparison
            original_forward = self.model.vision.forward
                
        total_layers = len(self.model.vision.transformer.layers)
        self.starting_depth = layer_index if layer_index >= 0 else total_layers + layer_index
        
        self._activate_self_attention_hooks()


    def _activate_self_attention_hooks(self):
        
        # Set gradients to TRUE for last X layers
        for name, param in self.model.vision.transformer.named_parameters():
            param.requires_grad = False
            if 'layers' in name:
                # get the depth from the parameter name
                depth = int(name.split('layers.')[1].split('.')[0])
                if depth >= self.starting_depth:
                    param.requires_grad = True
                    
        # Activate hooks
        for layer in range(self.starting_depth, len(self.model.vision.transformer.layers)):
            
            # TAKES ONLY THE TRANSFORMER BLOCKS (TRANSFORMER, MLP, LAYER NORM)
            current_layer = self.model.vision.transformer.layers[layer]
            
            # APPLY ATTENTION HOOK TO ATTENTION LAYER
            current_layer.attention.forward = types.MethodType(hooked_attention_forward, current_layer.attention)
                   
            current_layer.forward = types.MethodType(hooked_resblock_forward, current_layer)
                        
    def compute_legrad(self, text_embedding, image=None, apply_correction=True):
        self.compute_legrad_cogvlm(text_embedding, image)

    def compute_legrad_cogvlm(self, text_embedding, image=None):
        
        num_prompts = text_embedding.shape[0]
        
        if image is not None:
            
            # pdb.set_trace()
                
            # print("summary before forward")
            # print(torch.cuda.memory_summary())
                
            with torch.autograd.profiler.profile(use_cuda=True, record_shapes=True, profile_memory=True) as prof:
                _ = self.model.vision(image)
                
            # print(prof.key_averages().table(sort_by="cuda_memory_usage", row_limit=10))
                
            # print("summary after forward")
            # print(torch.cuda.memory_summary())     
              
        blocks_list = list(dict(self.model.vision.transformer.layers.named_children()).values())
        
        image_features_list = []
    
        # Collect images features (activations) for specified layers/blocks and postprocess them
        for layer in range(self.starting_depth, len(self.model.vision.transformer.layers)):
            
            intermediate_feat = self.model.vision.transformer.layers[layer].feat_post_mlp
            
            intermediate_feat = self.model.vision.linear_proj(intermediate_feat.mean(dim=0))

            #instead of sum try mean
            intermediate_feat = torch.mean(intermediate_feat, dim=0).unsqueeze(0)
                        
            image_features_list.append(intermediate_feat)
            
            
            
        # GET the number of tokens which is the size 
        num_tokens = blocks_list[-1].feat_post_mlp.shape[1] 
        w = h = int(math.sqrt(num_tokens))
        
        # ----- Get explainability map
        accum_expl_map = 0
        for layer, (blk, img_feat) in enumerate(zip(blocks_list[self.starting_depth:], image_features_list)):
            self.model.vision.zero_grad()

            # Compute similarity between text and image features
            sim = text_embedding @ img_feat.transpose(-1, -2)  # [1, 1]
            one_hot = F.one_hot(torch.arange(0, num_prompts)).float().requires_grad_(True).to(text_embedding.device)
            one_hot = torch.sum(one_hot * sim)
            
            
            attn_map = blocks_list[self.starting_depth + layer].attention.attention_map  # [b, num_heads, N, N]
            
            # Deleting paramater after
            del blocks_list[self.starting_depth + layer].attention.attention_map
            
            # -------- Get explainability map --------
            
            # Compute gradients
            # grad = torch.autograd.grad(one_hot, [attn_map], retain_graph=True, create_graph=True)[
            #     0]  # [batch_size * num_heads, N, N]
            grad = torch.autograd.grad(one_hot, [attn_map], retain_graph=False, create_graph=False)[
                0]  # [batch_size * num_heads, N, N]

            grad = torch.clamp(grad, min=0.)
            
            # Average attention and reshape
            image_relevance = grad.mean(dim=1).mean(dim=1)[:, 1:]  # average attn over [CLS] + patch tokens

            expl_map = rearrange(image_relevance, 'b (w h) -> 1 b w h', w=w, h=h)
            expl_map = F.interpolate(expl_map, scale_factor=14, mode='bilinear')  # [B, 1, H, W]

            accum_expl_map += expl_map

        # Min-Max Norm
        accum_expl_map = min_max(accum_expl_map)
        
        print("clearing gradients")
        for param in self.model.parameters():
            if param.grad is not None:
                param.grad = None
                
        for layer in range(self.starting_depth, len(self.model.vision.transformer.layers)):
            # Deleting paramater after
            del self.model.vision.transformer.layers[layer].feat_post_mlp
            
        # Synchronize and clear CUDA cache
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
            
        # Force garbage collection
        gc.collect()

        return accum_expl_map
        
class LePreprocess(nn.Module):
    """
    Modify OpenCLIP preprocessing to accept arbitrary image size.
    """

    def __init__(self, preprocess, image_size):
        super(LePreprocess, self).__init__()
        self.transform = Compose(
            [
                Resize((image_size, image_size), interpolation=InterpolationMode.BICUBIC),
                preprocess.transforms[-3],
                preprocess.transforms[-2],
                preprocess.transforms[-1],
            ]
        )

    def forward(self, image):
        return self.transform(image)