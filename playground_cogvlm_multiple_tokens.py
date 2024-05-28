import requests
from PIL import Image
import torch
import pdb
import sys
import spacy
import gc
import inspect
import pdb
import os
from transformers import AutoModelForCausalLM, LlamaTokenizer
from torchvision.transforms import Resize, Compose, ToTensor, Normalize
from legrad import LeWrapper, LePreprocess, visualize, visualize_save


def is_object(word):
    doc = nlp(word)
    non_object_nouns = {"image", "picture", "photo"}
    
    # Check if the word is a noun and not in the non-object list
    for token in doc:
        if token.pos_ == 'NOUN' and token.text.lower() not in non_object_nouns:
            return True
    return False

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
        
        
        token_ids = model.generate(**inputs, **gen_kwargs) # Obtain token_ids of caption 
        token_ids = token_ids[:, inputs['input_ids'].shape[1]:] 
        caption = tokenizer.decode(token_ids[0]) # From tokens to embeddings
        
        

        print("caption is: ", caption)
        
        
        text_embeddings = []
        
        # for i in range(token_ids.shape[1]):
        #     token_id = token_ids[0, i].unsqueeze(0).to(device)
        #     text_embedding = model.model.embed_tokens(token_id)
            
        #     token = tokenizer.decode(token_id)
            
        #     doc = nlp(token)

        #     if is_object(token): 
        #         print(f"{token} is an object")
        #         text_embeddings.append((text_embedding, token))
        
                
        # for i in range(token_ids.shape[1]):
        #     token_id = token_ids[0, i].unsqueeze(0).to(device)
        #     text_embedding = model.model.embed_tokens(token_id)
            
        #     token = tokenizer.decode(token_id)
            
        #     doc = nlp(token)

        #     if is_object(token): 
        #         print(f"{token} is an object")
        #         text_embeddings.append((text_embedding, token))        
        
        
        # OBTAIN OBJECTS FROM ENTIRE SENTENCE
        nlp = spacy.load('en_core_web_sm')
        doc = nlp(caption)
        non_object_nouns = {"image", "picture", "photo"}
        objects = [chunk.root.text for chunk in doc.noun_chunks if chunk.root.pos_ == 'NOUN' and chunk.root.text.lower() not in non_object_nouns]
        print(objects)  # Output: ['dog', 'woman']
        
        # for i in range(objects):
        #     token_id = token_ids[0, i].unsqueeze(0).to(device)
        #     text_embedding = model.model.embed_tokens(token_id)
        #     text_embeddings.append((text_embedding, objects[i]))
        
        
        for obj in objects:
            # pdb.set_trace()
            obj = obj.replace("'", "")
            # print(obj)
            # len(obj)
            obj_token_ids = tokenizer.encode(obj, return_tensors='pt').to(device)
            obj_embedding = model.model.embed_tokens(obj_token_ids)
            text_embeddings.append((obj_embedding, obj))
                    
        
        
        # pdb.set_trace()
        
        # TODO (1): From objects back to tokens  - Think this is done
        # TODO (2): From tokens back to embeddings - Think this is done.
        # TODO (3): Possibly aggegrate embeddings - What ae 
         
        
        
        
        
        
        # objects = ['dog', 'toilet']
        
                
        print("objects are: ", *[t[1] for t in text_embeddings])
        
        

    processed_image = inputs['images'][0][0]
    return text_embeddings, processed_image


def print_vram_usage():
    allocated = torch.cuda.memory_allocated() / 1024 ** 3  # Convert bytes to GB
    reserved = torch.cuda.memory_reserved() / 1024 ** 3    # Convert bytes to GB
    print(f"VRAM Usage - Allocated: {allocated:.2f} GB, Reserved: {reserved:.2f} GB")

# ------- model's paramters -------
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
# url = 'http://images.cocodataset.org/val2014/COCO_val2014_000000000042.jpg'
url = "http://images.cocodataset.org/val2017/000000039769.jpg"
file_name = os.path.splitext(os.path.basename(url))[0]


image = Image.open(
    requests.get(url, stream=True).raw).convert('RGB')

text_embs, processed_image = _get_text_embedding(model, tokenizer, "What is in the image?", DEVICE, image)

print("vram before running any loop: ")
print_vram_usage()

processed_image = processed_image.unsqueeze(0)


del model 
torch.cuda.empty_cache()

#model = LeWrapper(model)

for text_emb, token_str in text_embs:
    
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch_type,
        low_cpu_mem_usage=True,
        trust_remote_code=True
    ).to(DEVICE).eval()
    
    model = LeWrapper(model)

    # pdb.set_trace()
    torch.cuda.synchronize()
    print("vram start of loop: ")
    print_vram_usage()
    
    explainability_map = model.compute_legrad_cogvlm(image=processed_image, text_embedding=text_emb)
    explainability_map = explainability_map.to(torch.float32)
    torch.save(explainability_map, f'explainability_map_{file_name}_{token_str}.pt')

    
    
    # print(f"Explainability map shape for token '{token_str}': ", explainability_map.shape)
    
    visualize(heatmaps=explainability_map, image=image, save_path=f'heatmap_{file_name}_{token_str}.png')
    
    # pdb.set_trace()
    model.zero_grad()
    print("obtained visualiation for: ", token_str)
    
    # Free up memory
    del explainability_map
    del text_emb
    
    # Synchronize and clear CUDA cache
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    
    # Force garbage collection
    gc.collect()
    
    print("vram end of loop: ")
    print_vram_usage()
    # pdb.set_trace()
    del model
    
    
    
    