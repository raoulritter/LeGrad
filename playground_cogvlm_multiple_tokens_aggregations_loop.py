import requests
from PIL import Image
import torch
import pdb
import string
import os
import regex as re 
import sys
import numpy as np
import spacy
import gc
import inspect
import pdb
from transformers import AutoModelForCausalLM, LlamaTokenizer
from torchvision.transforms import Resize, Compose, ToTensor, Normalize
from legrad import LeWrapper, LePreprocess, visualize, visualize_save


# ------- model's parameters -------
MODEL_PATH = "THUDM/cogvlm-chat-hf"
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

torch_type = torch.bfloat16


def print_vram_usage():
    allocated = torch.cuda.memory_allocated() / 1024 ** 3  # Convert bytes to GB
    reserved = torch.cuda.memory_reserved() / 1024 ** 3    # Convert bytes to GB
    print(f"VRAM Usage - Allocated: {allocated:.2f} GB, Reserved: {reserved:.2f} GB")
    
def remove_punctuation(text):
    cleaned_text = ''
    for word in text.split():
        # Remove punctuation only at the end of the word
        cleaned_word = word.rstrip(string.punctuation)
        cleaned_text += cleaned_word + ' '
    return cleaned_text.strip()


def _get_text_embedding(model, tokenizer, query, device, image):
    # Prepare inputs using the custom build_conversation_input_ids method

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
        
        token_ids = model.generate(**inputs, **gen_kwargs) # OBTAIN TOKEN_IDS 
        token_ids = token_ids[:, inputs['input_ids'].shape[1]:] # RESHAPE 
        caption = tokenizer.decode(token_ids[0]) # OBTAIN CAPTION
        
        
        caption = re.sub(r'[^\w\s]', '', caption)
        
        print("caption is: ", caption)
    
        # OBTAIN OBJECTS FROM ENTIRE SENTENCE
        nlp = spacy.load('en_core_web_sm')
        doc = nlp(caption)
        non_object_nouns = {"image", "picture", "photo"}
        objects = [chunk.root.text for chunk in doc.noun_chunks if chunk.root.pos_ == 'NOUN' and chunk.root.text.lower() not in non_object_nouns]
        
        #objects = [obj.strip('()') for obj in objects]
        objects = [obj.replace("'","") for obj in objects]
        print("objects after: ", objects)

        text_embeddings = []
        
        # strategies = ['first','sum','mean']
        
        for obj in objects:
            
            obj_token_ids = tokenizer.encode(obj, return_tensors='pt').to(device) # From object (STRING) back to token(s)
            obj_token_ids = obj_token_ids[:,1:]
            
            obj_embedding = model.model.embed_tokens(obj_token_ids) # From tokens to embeddings
            
            #print("obj: ", obj)
            #print("obj_embedding shape: ", obj_embedding.shape)
            
            # POSSIBLE AGGREGATION
            if obj_embedding.shape[1] > 1: 
                
                output_name = obj + "_" + 'mean' + "_" + str(obj_embedding.shape[1]) 
                obj_embedding_mean = torch.mean(obj_embedding, dim=1, keepdim=True)
                text_embeddings.append((obj_embedding_mean, output_name))
                
                # for strategy in strategies: 
                #     if strategy=='first': 
                #         #obj_embedding_first = obj_embedding[:, 0, :]  
                #         obj_embedding_first = obj_embedding[:, 0, :].unsqueeze(1)    
                #         #print("shape after first: ", obj_embedding_first.shape)    
                #         text_embeddings.append((obj_embedding_first, obj + "_" + strategy))
                #     elif strategy=='sum': 
                #         obj_embedding_sum = torch.sum(obj_embedding, dim=1, keepdim=True)
                #         #print("shape after sum: ", obj_embedding_sum.shape)  
                #         text_embeddings.append((obj_embedding_sum, obj + "_" + strategy))
                #     elif strategy=='mean': 
                #         obj_embedding_mean = torch.mean(obj_embedding, dim=1, keepdim=True)
                #         #print("shape after mean: ", obj_embedding_mean.shape)  
                #         text_embeddings.append((obj_embedding_mean, obj + "_" + strategy))
                          
            else: 
                text_embeddings.append((obj_embedding, obj))
                #print("shape regular: ", obj_embedding.shape)  
        
        # pdb.set_trace()
                
        #print("objects are: ", *[t[1] for t in text_embeddings])
        
    processed_image = inputs['images'][0][0]
    return text_embeddings, processed_image


# Define function to process a single image
# def process_image(image_path, output_dir):
#     print("Processing image:", image_path)
    
#     image = Image.open(image_path).convert('RGB')
    
#     text_embs, processed_image = _get_text_embedding(model, tokenizer, "What is in the image?", DEVICE, image)

#     print("vram before running any loop: ")
#     print_vram_usage()

#     processed_image = processed_image.unsqueeze(0)

#     del model 
#     torch.cuda.empty_cache()

#     for text_emb, token_str in text_embs:
        
#         model = AutoModelForCausalLM.from_pretrained(
#             MODEL_PATH,
#             torch_dtype=torch_type,
#             low_cpu_mem_usage=True,
#             trust_remote_code=True
#         ).to(DEVICE).eval()
        
#         model = LeWrapper(model)

#         torch.cuda.synchronize()
#         print("vram start of loop: ")
#         print_vram_usage()
        
#         explainability_map = model.compute_legrad_cogvlm(image=processed_image, text_embedding=text_emb)
#         explainability_map = explainability_map.to(torch.float32)
        
#         save_path = os.path.join(output_dir, f'heatmap_{token_str}.png')
#         visualize(heatmaps=explainability_map, image=image, save_path=save_path)
        
#         model.zero_grad()
#         print("obtained visualiation for:", token_str)
        
#         del explainability_map
#         del text_emb
        
#         torch.cuda.synchronize()
#         torch.cuda.empty_cache()
        
#         gc.collect()
        
#         print("vram end of loop: ")
        
#         del model

# Load model and tokenizer
print("Loading Model")



tokenizer = LlamaTokenizer.from_pretrained("lmsys/vicuna-7b-v1.5")

# Directory containing images
image_dir = '/home/jwiers/LeGrad/test_images'
output_base_dir = '/home/jwiers/LeGrad/outputs_all_new'

# Ensure output directory exists
os.makedirs(output_base_dir, exist_ok=True)

# Loop over each image in the directory
for image_file in os.listdir(image_dir):
    if image_file.lower().endswith(('.png', '.jpg', '.jpeg')):
        image_path = os.path.join(image_dir, image_file)
        
        # Create a directory for each image in the output directory
        image_output_dir = os.path.join(output_base_dir, os.path.splitext(image_file)[0])
        os.makedirs(image_output_dir, exist_ok=True)
        
        print("Processing image:", image_path)
    
        image = Image.open(image_path).convert('RGB')
        
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_PATH,
            torch_dtype=torch_type,
            low_cpu_mem_usage=True,
            trust_remote_code=True
        ).to(DEVICE).eval()
        
        text_embs, processed_image = _get_text_embedding(model, tokenizer, "What is in the image?", DEVICE, image)
        

        print("vram before running any loop: ")
        print_vram_usage()

        processed_image = processed_image.unsqueeze(0)

        del model 
        torch.cuda.empty_cache()

        for text_emb, token_str in text_embs:
            
            model = AutoModelForCausalLM.from_pretrained(
                MODEL_PATH,
                torch_dtype=torch_type,
                low_cpu_mem_usage=True,
                trust_remote_code=True
            ).to(DEVICE).eval()
            
            model = LeWrapper(model)

            torch.cuda.synchronize()
            print("vram start of loop: ")
            print_vram_usage()
            
            explainability_map = model.compute_legrad_cogvlm(image=processed_image, text_embedding=text_emb)
            explainability_map = explainability_map.to(torch.float32)
            
            save_path = os.path.join(image_output_dir, f'heatmap_{token_str}.pt')
            
            torch.save(explainability_map, save_path)
            
            # visualize(heatmaps=explainability_map, image=image, save_path=save_path)
            
            model.zero_grad()
            print("obtained visualiation for:", token_str)
            
            del explainability_map
            del text_emb
            
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
            
            gc.collect()
            
            print("vram end of loop: ")
            
            del model
            
            
            
            
