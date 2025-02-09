from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
import torch
import torch_directml
from diffusers import StableDiffusionPipeline
from fastapi.responses import Response
import io
import peft

app = FastAPI()

# Configure CORS (Allow all origins for testing)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize DirectML device
dml = torch_directml.device()

# Load Stable Diffusion model
# Load the model with proper authentication
model_id = "sd-legacy/stable-diffusion-v1-5"
pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
pipe = pipe.to(dml)

# Load LoRA Model
lora_model_path = "./Super Ani ver2_v2.0.safetensors"
try:
    pipe.load_lora_weights(lora_model_path, weights = 1.2)
    pipe.fuse_lora()
    pipe.unet.set_default_attn_processor()
    print("LoRA model loaded successfully.")
except Exception as e:
    print(f"Failed to load LoRA model: {e}")

@app.get("/")
def generate(
    prompt: str,
    cfg_scale: float = Query(5, description="How strictly the image follows the prompt"),
    steps: int = Query(50, description="Number of sampling steps"),
    width: int = Query(512, description="Image width"),
    height: int = Query(512, description="Image height"),
    sampler: str = Query("DPM++ 2M Karras", description="Sampling method"),
    seed: int = Query(None, description="Random seed")
):

    # Apply LoRA trigger word (if applicable)
    prompt = f"by Super Ani, {prompt}"
    # If seed is provided, set the generator to control randomness
    if seed is not None:
        generator = torch.manual_seed(seed)
    else:
        generator = None  # No seed, let the RNG generate a random one
    
    # Generate image
    image = pipe(prompt, 
                 guidance_scale=cfg_scale, 
                 num_inference_steps=steps,
                 height=height, 
                 width=width, 
                 sampler=sampler, 
                 generator=generator).images[0]

    # Convert image to bytes
    img_io = io.BytesIO()
    image.save(img_io, format="PNG")
    img_io.seek(0)

    return Response(content=img_io.getvalue(), media_type="image/png")

