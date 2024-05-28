# import requests
# from PIL import Image
# import torch
# import pdb
# import spacy
# import gc
# from transformers import AutoModelForCausalLM, LlamaTokenizer
# from legrad import LeWrapper, visualize

# # Load spaCy model
# nlp = spacy.load('en_core_web_sm')

# def _get_text_embedding(model, tokenizer, query, device, image):
#     if image is None:
#         inputs = model.build_conversation_input_ids(tokenizer, query=query, history=[], template_version='base')
#     else:
#         inputs = model.build_conversation_input_ids(tokenizer, query=query, history=[], images=[image])

#     inputs = {
#         'input_ids': inputs['input_ids'].unsqueeze(0).to(device),
#         'token_type_ids': inputs['token_type_ids'].unsqueeze(0).to(device),
#         'attention_mask': inputs['attention_mask'].unsqueeze(0).to(device),
#         'images': [[inputs['images'][0].to(device).to(torch.bfloat16)]] if image is not None else None,
#     }

#     gen_kwargs = {"max_length": 2048, "do_sample": False}

#     with torch.no_grad():
#         token_ids = model.generate(**inputs, **gen_kwargs)
#         token_ids = token_ids[:, inputs['input_ids'].shape[1]:]
#         caption = tokenizer.decode(token_ids[0])

#         print("caption is: ", caption)

#         doc = nlp(caption)
#         non_object_nouns = {"image", "picture", "photo"}
#         objects = [chunk.root.text for chunk in doc.noun_chunks if chunk.root.pos_ == 'NOUN' and chunk.root.text.lower() not in non_object_nouns]
#         print(objects)

#         text_embeddings = []
#         for obj in objects:
#             obj_token_ids = tokenizer.encode(obj, return_tensors='pt').to(device)
#             obj_embedding = model.model.embed_tokens(obj_token_ids)
#             text_embeddings.append((obj_embedding, obj))
            

#         print("Objects are: ", [obj for _, obj in text_embeddings])

#     processed_image = inputs['images'][0][0]
#     return text_embeddings, processed_image

# def print_vram_usage():
#     allocated = torch.cuda.memory_allocated() / 1024 ** 3
#     reserved = torch.cuda.memory_reserved() / 1024 ** 3
#     print(f"VRAM Usage - Allocated: {allocated:.2f} GB, Reserved: {reserved:.2f} GB")

# # ------- model's parameters -------
# MODEL_PATH = "THUDM/cogvlm-chat-hf"
# DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
# torch_type = torch.bfloat16

# # Obtain COGVLM model from HF
# print("Loading Model")
# model = AutoModelForCausalLM.from_pretrained(
#     MODEL_PATH,
#     torch_dtype=torch_type,
#     low_cpu_mem_usage=True,
#     trust_remote_code=True
# ).to(DEVICE).eval()

# tokenizer = LlamaTokenizer.from_pretrained("lmsys/vicuna-7b-v1.5")

# # PROCESS IMAGE
# # url = 'http://images.cocodataset.org/val2014/COCO_val2014_000000000042.jpg'
# url = 'http://images.cocodataset.org/val2014/COCO_val2014_000000061471.jpg'


# image = Image.open(requests.get(url, stream=True).raw).convert('RGB')

# text_embs, processed_image = _get_text_embedding(model, tokenizer, "What is in the image?", DEVICE, image)

# print("vram before running any loop: ")
# print_vram_usage()

# processed_image = processed_image.unsqueeze(0)

# # Delete the model to clear memory before the loop
# del model
# torch.cuda.empty_cache()
# gc.collect()

# # Helper function to aggregate heatmaps
# def aggregate_heatmaps(heatmaps):
#     if len(heatmaps) == 0:
#         return None
#     aggregated_heatmap = torch.mean(torch.stack(heatmaps), dim=0)
#     return aggregated_heatmap

# for text_emb, token_str in text_embs:
#     pdb.set_trace()
#     token_heatmaps = []
#     for token_embedding in text_emb[1]:
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

#         explainability_map = model.compute_legrad_cogvlm(image=processed_image, text_embedding=token_embedding.unsqueeze(0))
#         explainability_map = explainability_map.to(torch.float32)
#         token_heatmaps.append(explainability_map)

#         model.zero_grad()
#         print("obtained visualiation for: ", token_str)

#         # Free up memory
#         del explainability_map
#         torch.cuda.synchronize()
#         torch.cuda.empty_cache()
#         gc.collect()

#         print("vram end of loop: ")
#         print_vram_usage()
        
#         del model
#         torch.cuda.empty_cache()
#         gc.collect()

#     aggregated_heatmap = aggregate_heatmaps(token_heatmaps)
#     visualize(heatmaps=aggregated_heatmap, image=image, save_path=f'heatmap_{token_str}.png')

# # Final cleanup
# torch.cuda.empty_cache()
# gc.collect()
# print("After cleanup:")
# print_vram_usage()



import requests
from PIL import Image
import torch
import spacy
import gc
import pdb
from transformers import AutoModelForCausalLM, LlamaTokenizer
from legrad import LeWrapper, visualize

# Load spaCy model
nlp = spacy.load('en_core_web_sm')

def _get_text_embedding(model, tokenizer, query, device, image):
    if image is None:
        inputs = model.build_conversation_input_ids(tokenizer, query=query, history=[], template_version='base')
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
        caption = tokenizer.decode(token_ids[0])

        print("caption is: ", caption)

        doc = nlp(caption)
        non_object_nouns = {"image", "picture", "photo"}
        objects = [chunk.root.text for chunk in doc.noun_chunks if chunk.root.pos_ == 'NOUN' and chunk.root.text.lower() not in non_object_nouns]
        print(objects)

        text_embeddings = []
        for obj in objects:
            obj = obj.replace("'", "")
            pdb.set_trace()
            obj_token_ids = tokenizer.encode(obj, return_tensors='pt').to(device)
            obj_embedding = model.model.embed_tokens(obj_token_ids)
            text_embeddings.append((obj_embedding, obj))

        print("Objects are: ", [obj for _, obj in text_embeddings])

    processed_image = inputs['images'][0][0]
    return text_embeddings, processed_image

def print_vram_usage():
    allocated = torch.cuda.memory_allocated() / 1024 ** 3
    reserved = torch.cuda.memory_reserved() / 1024 ** 3
    print(f"VRAM Usage - Allocated: {allocated:.2f} GB, Reserved: {reserved:.2f} GB")

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
url = 'http://images.cocodataset.org/val2014/COCO_val2014_000000061471.jpg'
image = Image.open(requests.get(url, stream=True).raw).convert('RGB')

text_embs, processed_image = _get_text_embedding(model, tokenizer, "What is in the image?", DEVICE, image)

print("vram before running any loop: ")
print_vram_usage()

processed_image = processed_image.unsqueeze(0)

# Delete the model to clear memory before the loop
del model
torch.cuda.empty_cache()
gc.collect()

# Helper function to aggregate heatmaps
def aggregate_heatmaps(heatmaps):
    if len(heatmaps) == 0:
        return None
    aggregated_heatmap = torch.mean(torch.stack(heatmaps), dim=0)
    return aggregated_heatmap

for text_emb, token_str in text_embs:
    token_heatmaps = []
    for token_embedding in text_emb[0]:  # Iterate over each token embedding
        # pdb.set_trace()
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

        explainability_map = model.compute_legrad_cogvlm(image=processed_image, text_embedding=token_embedding.unsqueeze(0))
        explainability_map = explainability_map.to(torch.float32)
        token_heatmaps.append(explainability_map)

        model.zero_grad()
        print("obtained visualization for token in: ", token_str)

        # Free up memory
        del explainability_map
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        gc.collect()

        print("vram end of loop: ")
        print_vram_usage()
        
        del model
        torch.cuda.empty_cache()
        gc.collect()

    aggregated_heatmap = aggregate_heatmaps(token_heatmaps)
    visualize(heatmaps=aggregated_heatmap, image=image, save_path=f'heatmap_aggregate_{token_str}.png')

# Final cleanup
torch.cuda.empty_cache()
gc.collect()
print("After cleanup:")
print_vram_usage()
