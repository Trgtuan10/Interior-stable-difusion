from diffusers import StableDiffusionPipeline
import torch

device = "cuda" if torch.cuda.is_available() else "cpu"

def load_model():
    pipe = StableDiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5", use_safetensors = True).to(device)

    pipe.load_lora_weights("TrgTuan10/Interior", weight_name="xsarchitectural-7.safetensors", adapter_name="architecture")
    trigger_words = " ,VERRIERES, DAYLIGHTINDIRECT, LIGHTINGAD, MAGAZINE8K, CINEMATIC LIGHTING, EPIC COMPOSITION"

    return pipe, trigger_words

def gen_interior(pipe, prompts, trigger_words="", neg="", seed=42, num_images=1, num_infer_steps=35, height=512, width=512):
    prompts = prompts + " high quality, lightning, luxury" + trigger_words
    
    generator = torch.Generator(device=device).manual_seed(seed)
    images = pipe(
        prompt=prompts,
        negative_prompt=neg,
        do_classifier_free_guidance=True,
        num_images_per_prompt=num_images,
        num_inference_steps=num_infer_steps,
        height=height,
        width=width,
        guidance_scale=7,
        generator=generator
    )
    return images

if __name__ == "__main__":
    pipe, trigger_words = load_model()
    prompts = "Raw photo, interior design minimalistic living room, studio quality, ambient lighting, 30mm,perfect window,voluemtric lightning,textiles pillows,"
    negative_prompt = "(multiple outlets:1.9),carpets,(multiple tv screens:1.9),2 tables,lamps,lightbuble,(plants:1.6)bad-hands-5, ng_deepnegative_v1_75t, EasyNegative, bad_prompt_version2, bad-artist-anime, bad-artist, bad-image-v2-39000, verybadimagenegative_v1.3, text, close up, cropped, out of frame, worst quality, low quality, jpeg artifacts, ugly, duplicate, morbid, mutilated, extra fingers, mutated hands, poorly drawn hands, poorly drawn face, mutation, deformed, blurry, dehydrated, bad anatomy, bad proportions, extra limbs, cloned face, disfigured, gross proportions, malformed limbs, missing arms, missing legs, extra arms, extra legs, fused fingers, too many fingers, long neck"
    images = gen_interior(pipe, prompts, neg=negative_prompt, trigger_words=trigger_words, num_images=1)
    image = images.images[0]
    image.save("interior.jpg")
    print("Image saved as interior.jpg")