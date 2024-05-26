import requests
from PIL import Image
import torch
import pdb
import sys
import inspect
import pdb
from transformers import AutoModelForCausalLM, LlamaTokenizer
from torchvision.transforms import Resize, Compose, ToTensor, Normalize
from legrad import LeWrapper, LePreprocess, visualize, visualize_save


# def _get_text_embedding(model, tokenizer, query, device, image):

#     # # Prepare inputs using the custom build_conversation_input_ids method
#     # inputs = model.build_conversation_input_ids(tokenizer, query=query, history=[], images=[image])  # chat mode

#     history = []


#     print("query: ", query)
#     print("tokenizer: ", tokenizer)
#     print("history: ", history)
#     print("images: ", image)

#     input_by_model = model.build_conversation_input_ids(tokenizer, query=query, history=history, images=[image])

#     # if image is None:
#     #     inputs = model.build_conversation_input_ids(tokenizer, query=query, history=history, template_version='base')
#     # else:
#     #     inputs = model.build_conversation_input_ids(tokenizer, query=query, history=history, images=[image])

#     inputs = {
#             'input_ids': input_by_model['input_ids'].unsqueeze(0).to(DEVICE),
#             'token_type_ids': input_by_model['token_type_ids'].unsqueeze(0).to(DEVICE),
#             'attention_mask': input_by_model['attention_mask'].unsqueeze(0).to(DEVICE),
#             'images': [[input_by_model['images'][0].to(DEVICE).to(torch_type)]] if image is not None else None,
#         }


#     if 'cross_images' in inputs and inputs['cross_images']:
#             inputs['cross_images'] = [[inputs['cross_images'][0].to(DEVICE).to(torch.bfloat16)]]

#     gen_kwargs = {"max_length": 2048, "do_sample": False}

#     with torch.no_grad():
#         outputs = model.generate(**inputs, **gen_kwargs)
#         outputs = outputs[:, inputs['input_ids'].shape[1]:]

#     return outputs


# Define the preprocessing steps
# def create_cogvlm_preprocess(image_size=448):
#     from torchvision import transforms
#     preprocess_pipeline = transforms.Compose([
#         transforms.Resize((image_size, image_size)),
#         transforms.ToTensor(),
#         transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
#     ])
#     return preprocess_pipeline

# preprocess_pipeline = create_cogvlm_preprocess()


def _get_text_embedding(model, tokenizer, query, device, image):
    # Prepare inputs using the custom build_conversation_input_ids method

    if image is None:
        inputs = model.build_conversation_input_ids(tokenizer, query=query, history=[],
                                                    template_version='base')  # chat mode
    else:
        inputs = model.build_conversation_input_ids(tokenizer, query=query, history=[], images=[image])

        # try:
    #     source_code = inspect.getsource(model.build_conversation_input_ids)
    #     print("Source Code:\n", source_code)
    # except TypeError:

    #     print("Couldn't retrieve source code. Function may be built-in or compiled.")

    inputs = {
        'input_ids': inputs['input_ids'].unsqueeze(0).to(device),
        'token_type_ids': inputs['token_type_ids'].unsqueeze(0).to(device),
        'attention_mask': inputs['attention_mask'].unsqueeze(0).to(device),
        'images': [[inputs['images'][0].to(device).to(torch.bfloat16)]] if image is not None else None,
    }

    # if inputs['images'] is None or not inputs['images'][0]:
    #     raise ValueError("The image input is not properly initialized or is None")

    gen_kwargs = {"max_length": 2048, "do_sample": False}

    with torch.no_grad():
        
        
        token_ids = model.generate(**inputs, **gen_kwargs)
        
        

        token_ids = token_ids[:, inputs['input_ids'].shape[1]:]

        print(tokenizer.decode(token_ids[0][19]))

        #output = tokenizer.decode(token_ids[0])
        
        
        # TODO: Obtain token_ids for all the objects in the output and embed those 
        
        # all_tokens = []
        # for i in range(token_ids.shape[1]):
        #     all_tokens.append(token_ids[0][i])
            
            
        # pdb.set_trace()

        # all_tokens_tensor = torch.tensor(all_tokens).unsqueeze(0).to()
        
        text_embeddings = []
        for i in range(token_ids.shape[1]):
            # pdb.set_trace()
            token_id = token_ids[0, i].unsqueeze(0).to(device)
            text_embedding = model.model.embed_tokens(token_id)
            text_embeddings.append((text_embedding, tokenizer.decode(token_id)))


        
        # print(all_embedded_tokens)

        
        # print(all_embedd_token)

        # token_id_list =
        # Do it for the different tokens 
        # text_embedding = model.model.embed_tokens(all_embedded_tokens)
        # text_embedding = model.model.embed_tokens(token_ids[0,19])
        # text_embedding = model.model.embed_tokens(token_ids[0,0])
        
        # THIS TTEXT_EMBEDDING SHOULD THEN BE PASSED TO THE 
        
        # print("token_ids shape: ",token_ids)
        # print("text_embedding shape: ",text_embedding.shape)

        # output = tokenizer.decode(token_ids[0])
        
        # breakpoint()
        
        # print(inspect.getsource(model.model.embed_tokens))

        # print("text embeddings: ", output)

    # print("inputs images len:", len(inputs['images']))
    # print("inputs images shape: ", inputs['images'][0][0].shape)

    processed_image = inputs['images'][0][0]
    return text_embeddings, processed_image


def print_vram_usage():
    allocated = torch.cuda.memory_allocated() / 1024 ** 3  # Convert bytes to GB
    reserved = torch.cuda.memory_reserved() / 1024 ** 3    # Convert bytes to GB
    print(f"VRAM Usage - Allocated: {allocated:.2f} GB, Reserved: {reserved:.2f} GB")



def apply_transforms(image):
    # Define the image size expected by the model
    image_size = 224  # This size should be confirmed from the model specifications

    resize_transform = Resize((image_size, image_size))
    to_tensor_transform = ToTensor()
    normalize_transform = Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    transform = Compose([
        resize_transform,
        to_tensor_transform,
        normalize_transform
    ])

    processed_image = transform(image)

    return processed_image


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
url = 'http://images.cocodataset.org/val2014/COCO_val2014_000000000042.jpg'
# url = "http://images.cocodataset.org/val2017/000000039769.jpg"

image = Image.open(
    requests.get(url, stream=True).raw).convert('RGB')
# image = Image.open(image_url)

# image_tensor = preprocess_pipeline(image).unsqueeze(0).to(DEVICE)
# text_emb, processed_image = _get_text_embedding(model, tokenizer, "a photo of a cat", DEVICE, None)
# text_emb, processed_image = _get_text_embedding(model, tokenizer, "What is in the image", DEVICE, image)
text_embs, processed_image = _get_text_embedding(model, tokenizer, "What is in the image", DEVICE, image)

print_vram_usage()
# print(len(text_embs))



processed_image = processed_image.unsqueeze(0)

# print("obtained output embedding")

# print("text embedding shape: ", text_emb.shape)
# print("processed image shape: ", processed_image.shape)

model = LeWrapper(model)

# explainability_map = model.compute_legrad_cogvlm(image=processed_image, text_embedding=text_emb)

# explainability_map = explainability_map.to(torch.float32)
# print("explainability_map shape: ", explainability_map.shape)




# for text_emb, token_str in text_embs:
#     pdb.set_trace()
#     torch.cuda.synchronize()

    
#     print_vram_usage()
#     torch.cuda.empty_cache()
    
#     explainability_map = model.compute_legrad_cogvlm(image=processed_image, text_embedding=text_emb)
#     explainability_map = explainability_map.to(torch.float32)
#     print(f"Explainability map shape for token '{token_str}': ", explainability_map.shape)
#     visualize(heatmaps=explainability_map, image=image, save_path=f'heatmap_{token_str}.png')
#     # del explainability_map
#     #print all used variables
#     print("Used variables: ", locals().keys())
#     print()
#     print_vram_usage()


for text_emb, token_str in text_embs:
    pdb.set_trace()
    torch.cuda.synchronize()
    print_vram_usage()
    
    explainability_map = model.compute_legrad_cogvlm(image=processed_image, text_embedding=text_emb)
    explainability_map = explainability_map.to(torch.float32)
    print(f"Explainability map shape for token '{token_str}': ", explainability_map.shape)
    
    visualize(heatmaps=explainability_map, image=image, save_path=f'heatmap_{token_str}.png')
    
    # Free up memory
    del explainability_map
    del text_emb
    
    # Synchronize and clear CUDA cache
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    
    # Force garbage collection
    gc.collect()
    
    print_vram_usage()

    
   




# # data_config = timm.data.resolve_model_data_config(model)

# #explainability_map = model.compute_legrad_vmap_clip(image=image, text_embedding=text_embedding)

# # ___ (Optional): Visualize overlay of the image + heatmap ___
# visualize_save(heatmaps=explainability_map, image=image, save_path='output.png')
# visualize(heatmaps=explainability_map, image=image, save_path='output.png')