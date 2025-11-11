from diffusers import KandinskyPriorPipeline, KandinskyPipeline
import torch
import time

# Start timer
t0 = time.time()

# ---- Load the prior pipeline (text → image embeddings) ----
prior = KandinskyPriorPipeline.from_pretrained(
    "kandinsky-community/kandinsky-2-1-prior",
    torch_dtype=torch.float32
)
prior.to("cpu")

# ---- Load the main diffusion pipeline ----
pipe = KandinskyPipeline.from_pretrained(
    "kandinsky-community/kandinsky-2-1",
    torch_dtype=torch.float32
)
pipe.to("cpu")

# ---- Prompt ----
prompt = "a lake by the mountains during sunset"

# ---- Step 1: Get text embeddings from the prior ----
image_embeds, negative_image_embeds = prior(prompt).to_tuple()

# ---- Step 2: Generate the actual 16:9 image ----
image = pipe(
    prompt=prompt,
    image_embeds=image_embeds,
    negative_image_embeds=negative_image_embeds,
    num_inference_steps=15,   # fewer steps → faster, less quality
    guidance_scale=6.0,       # lower guidance → smoother but less strict
    width=1280,               # smaller 16:9 (HD)
    height=720
).images[0]


# ---- Save the result ----
image.save("kandinsky_result_16x9.png")

print("✅ Done in", round(time.time() - t0, 1), "seconds")
