import requests
from PIL import Image
import torch
import pdb
import sys
import inspect
import os
import gc
from transformers import AutoModelForCausalLM, LlamaTokenizer
from torchvision.transforms import Resize, Compose, ToTensor, Normalize
from legrad import LeWrapper, visualize, visualize_save

def _get_text_embedding(model, tokenizer, query, device, image):
    if image is None:
        inputs = model.build_conversation_input_ids(tokenizer, query=query, history=[],
                                                    template_version='base')  # chat mode
    else:
        inputs = model.build_conversation_input_ids(tokenizer, query=query, history=[], images=[image])

    inputs = {
        'input_ids': inputs['input_ids'].unsqueeze(0).to(device),
        'token_type_ids': inputs['token_type_ids'].unsqueeze(0).to(device),
        'attention_mask': inputs['attention_mask'].unsqueeze(0).to(device),
        'images': [[inputs['images'][0].to(device).to(torch.bfloat16)]] if image is not None else None,
    }

    gen_kwargs = {"max_length": 2048, "do_sample": False}

    with torch.no_grad():
        token_ids = model.generate(**inputs, **gen_kwargs)
        token_ids = token_ids[:, inputs['input_ids'].shape[1]:]

        text_embeddings = []
        for i in range(token_ids.shape[1]):
            token_id = token_ids[0, i].unsqueeze(0).to(device)
            text_embedding = model.model.embed_tokens(token_id)
            text_embeddings.append((text_embedding, tokenizer.decode(token_id)))

    processed_image = inputs['images'][0][0]
    return text_embeddings, processed_image

def print_vram_usage():
    allocated = torch.cuda.memory_allocated() / 1024 ** 3  # Convert bytes to GB
    reserved = torch.cuda.memory_reserved() / 1024 ** 3    # Convert bytes to GB
    print(f"VRAM Usage - Allocated: {allocated:.2f} GB, Reserved: {reserved:.2f} GB")

def apply_transforms(image):
    image_size = 224  # This size should be confirmed from the model specifications
    transform = Compose([
        Resize((image_size, image_size)),
        ToTensor(),
        Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    processed_image = transform(image)
    return processed_image

# ------- model's parameters -------
MODEL_PATH = "THUDM/cogvlm-chat-hf"
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
torch_type = torch.bfloat16

# Obtain COGVLM model from HF
print("Loading Model")
model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    torch_dtype=torch_type,
    low_cpu_mem_usage=True,
    trust_remote_code=True
).to(DEVICE).eval()

tokenizer = LlamaTokenizer.from_pretrained("lmsys/vicuna-7b-v1.5")

# PROCESS IMAGE
url = 'http://images.cocodataset.org/val2014/COCO_val2014_000000000042.jpg'
image = Image.open(requests.get(url, stream=True).raw).convert('RGB')
text_embs, processed_image = _get_text_embedding(model, tokenizer, "What is in the image", DEVICE, image)

# Save text_embs and token_strs to disk
save_dir = 'embeddings'
os.makedirs(save_dir, exist_ok=True)
for i, (text_emb, token_str) in enumerate(text_embs):
    torch.save(text_emb, os.path.join(save_dir, f'text_emb_{i}.pt'))
    with open(os.path.join(save_dir, f'token_str_{i}.txt'), 'w') as f:
        f.write(token_str)

# Clear the model and text_embs from memory
del model
del text_embs
torch.cuda.empty_cache()
gc.collect()

# Process each text_emb and token_str one by one
for i in range(len(os.listdir(save_dir)) // 2):  # Assuming each text_emb and token_str pair
    # Reload the model
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch_type,
        low_cpu_mem_usage=True,
        trust_remote_code=True
    ).to(DEVICE).eval()
    model = LeWrapper(model)
    
    # Load text_emb and token_str
    text_emb = torch.load(os.path.join(save_dir, f'text_emb_{i}.pt'))
    with open(os.path.join(save_dir, f'token_str_{i}.txt'), 'r') as f:
        token_str = f.read()
    
    torch.cuda.synchronize()
    print_vram_usage()
    
    explainability_map = model.compute_legrad_cogvlm(image=processed_image, text_embedding=text_emb)
    explainability_map = explainability_map.to(torch.float32)
    print(f"Explainability map shape for token '{token_str}': ", explainability_map.shape)
    
    visualize(heatmaps=explainability_map, image=image, save_path=f'heatmap_{token_str}.png')
    
    # Free up memory
    del explainability_map
    del text_emb
    del model
    
    # Synchronize and clear CUDA cache
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    
    # Force garbage collection
    gc.collect()
    
    print_vram_usage()

# Clean up the saved files
import shutil
shutil.rmtree(save_dir)