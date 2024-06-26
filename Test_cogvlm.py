import argparse
import requests
from PIL import Image
import torch
import numpy as np
import cv2
from transformers import AutoModelForCausalLM, LlamaTokenizer
from legrad import LeWrapper, LePreprocess, visualize
import matplotlib.pyplot as plt

# Parse arguments for CogVLM initialization
parser = argparse.ArgumentParser()
parser.add_argument("--quant", choices=[4], type=int, default=None, help='quantization bits')
parser.add_argument("--from_pretrained", type=str, default="THUDM/cogvlm-chat-hf", help='pretrained ckpt')
parser.add_argument("--local_tokenizer", type=str, default="lmsys/vicuna-7b-v1.5", help='tokenizer path')
parser.add_argument("--fp16", action="store_true")
parser.add_argument("--bf16", action="store_true")
parser.add_argument("--image", type=str, default="http://images.cocodataset.org/val2017/000000039769.jpg", help='image url')
parser.add_argument("--text", type=str, default="a photo of a cat", help='text query')
args = parser.parse_args()

# Setup device and data type
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
torch_type = torch.bfloat16 if args.bf16 else torch.float16

tokenizer = LlamaTokenizer.from_pretrained(args.local_tokenizer)

# Initialize model
model = AutoModelForCausalLM.from_pretrained(
    args.from_pretrained,
    torch_dtype=torch.bfloat16 if args.bf16 else torch.float16,
    low_cpu_mem_usage=False,
    trust_remote_code=True
).to(DEVICE).eval()

# Equip the model with LeGrad
model = LeWrapper(model)

# Define the preprocessing steps
def create_cogvlm_preprocess(image_size=448):
    from torchvision import transforms
    preprocess_pipeline = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    return preprocess_pipeline

preprocess_pipeline = create_cogvlm_preprocess()

# Function to load image from URL
def change_to_url(url):
    img_pil = Image.open(requests.get(url, stream=True).raw).convert('RGB')
    return img_pil


def _get_text_embedding(model, tokenizer, query, device, image):
    # Prepare inputs using the custom build_conversation_input_ids method
    inputs = model.build_conversation_input_ids(tokenizer, query=query, history=[], images=[image])  # chat mode
    inputs = {
        'input_ids': inputs['input_ids'].unsqueeze(0).to(device),
        'token_type_ids': inputs['token_type_ids'].unsqueeze(0).to(device),
        'attention_mask': inputs['attention_mask'].unsqueeze(0).to(device),
        'images': [[inputs['images'][0].to(device).to(torch.bfloat16 if args.bf16 else torch.float16)]],
    }

    gen_kwargs = {"max_length": 2048, "do_sample": False}

    with torch.no_grad():
        outputs = model.generate(**inputs, **gen_kwargs)
        outputs = outputs[:, inputs['input_ids'].shape[1]:]
        text_embeddings = tokenizer.decode(outputs[0])

    return text_embeddings

# Function to convert logits to heatmaps
def logits_to_heatmaps(logits, image_cv):
    logits = logits[0, 0].detach().cpu().numpy()
    logits = (logits * 255).astype('uint8')
    heat_map = cv2.applyColorMap(logits, cv2.COLORMAP_JET)
    viz = 0.4 * image_cv + 0.6 * heat_map
    viz = cv2.cvtColor(viz.astype('uint8'), cv2.COLOR_BGR2RGB)
    return viz

def main(image_url, text_query):
    image = change_to_url(image_url)
    image_tensor = preprocess_pipeline(image).unsqueeze(0).to(DEVICE)

    # Get text embeddings
    text_emb = _get_text_embedding(model, tokenizer, text_query, DEVICE, image)

    # Debugging statement for text embeddings
    print(f"Text Embeddings: {text_emb}")

    # Get visual features using the visual encoder (Eva2LargeEncoder)

    # Debugging statement to
    visual_encoder = model.model.vision
    with torch.no_grad():
        visual_features = visual_encoder(image_tensor)

    # Debugging statement for visual features
    print(f"Visual Features: {visual_features}")

    # Compute LeGrad logits
    logits_legrad = model.compute_legrad(text_embedding=text_emb, image=visual_features)

    # Debugging statement for logits
    print(f"Logits: {logits_legrad}")

    explainability_map = logits_to_heatmaps(logits_legrad, np.array(image))

    # Display the image with the heatmap
    plt.imshow(explainability_map)
    plt.axis('off')
    plt.show()


if __name__ == "__main__":
    main(args.image, args.text)