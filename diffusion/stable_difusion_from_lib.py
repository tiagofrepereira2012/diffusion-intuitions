# Using the stable difusion model from the course

# https://huggingface.co/stabilityai/stable-diffusion-2-1-base

from diffusers import StableDiffusionPipeline, EulerDiscreteScheduler
import torch


def main():
    model_id = "stabilityai/stable-diffusion-2-1-base"

    scheduler = EulerDiscreteScheduler.from_pretrained(model_id, subfolder="scheduler")
    pipe = StableDiffusionPipeline.from_pretrained(model_id, scheduler=scheduler, torch_dtype=torch.float16)
    pipe = pipe.to("cuda")

    prompt = "a photo of an astronaut riding a horse on mars"
    image = pipe(prompt).images[0]  
        
    #image.save("astronaut_rides_horse.png")
    image.save("horse_rinding_astronaut.png")

if __name__ == "__main__":
    main()