import streamlit as st
from PIL import Image
from utils import icon
from streamlit_image_select import image_select
import random
from pathlib import Path

import sys
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))
from base_gen import load_model_base, gen_base
from image_condition_gen import load_controlnet_model, gen_controlnet

CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints"
MODEL_PATH = str(CHECKPOINT_DIR / "Interior.safetensors")
LORA_PATH = str(CHECKPOINT_DIR)


#pipeline
pipe = None

# UI configurations
st.set_page_config(page_title="Interor and Architect Generator",
                   page_icon=":bridge_at_night:",
                   layout="wide")
icon.show_icon(":foggy:")
st.markdown("# :rainbow[Interior and Architerct Generation]")


# Placeholders for images and gallery
generated_images_placeholder = st.empty()
gallery_placeholder = st.empty()




def main():
    
    st.sidebar.info("**Start here ↓**", icon="👋🏾")

    option_function = st.sidebar.selectbox(
        "Choose a function", ["Generate", "Fix Style"], index=0)

    #take image control
    if option_function == "Fix Style":
        image_controlnet = st.file_uploader("Upload image", type=["jpg", "jpeg", "png"])
        if image_controlnet is not None:
            image_controlnet = Image.open(image_controlnet)
            h, w = image_controlnet.size
            st.image(image_controlnet, caption='Uploaded Image', width="stretch")
            # image_controlnet.save("controlnet.jpg")
    
    # Form for user input
    with st.sidebar:
        with st.form("generation-form"):
            prompt = st.text_area(
                ":orange[**Enter prompt: ✍🏾**]",
                value="a modern living room with a sofa and a TV",
                height=150)
            negative_prompt = st.text_area(":orange[**Negative prompt 🙅🏽‍♂️**]",
                                            value="",
                                            help="This is a negative prompt, basically type what you don't want to see in the generated image",
                                            height=100)
            
            st.divider()
            num_outputs = st.slider(
                "Number of images to output", value=1, min_value=1, max_value=4)
            width = st.number_input("Width of output image", value=1024)
            height = st.number_input("Height of output image", value=768)

            # The Big Red "Submit" Button!
            submitted = st.form_submit_button(
                "Submit", type="primary", use_container_width=True)
    
    if submitted:
        with st.status('👩🏾‍🍳 Whipping up your words into art...', expanded=True) as status:
            st.write("⚙️ Model initiated")
            st.write("🙆‍♀️ Stand up and strecth in the meantime")
            
        # try: 
        if submitted:
            # Calling the replicate API to get the image
            with generated_images_placeholder.container():
                all_images = []  # List to store all generated images
                seed = random.randint(0, 100000)
                
                if option_function == "Generate":
                    pipe = load_model_base(MODEL_PATH)
                    pipe.load_lora_weights(LORA_PATH, weight_name="Interior_lora.safetensors")
                    pipe.fuse_lora(lora_scale=0.7)
                    output = gen_base(pipe, 
                                    prompt, 
                                    trigger_words="", 
                                    neg=negative_prompt, 
                                    num_images=num_outputs, 
                                    height=height, width=width,
                                    seed = seed)
                elif option_function == "Fix Style":
                    pipe = load_controlnet_model(MODEL_PATH)
                    pipe.load_lora_weights(LORA_PATH, weight_name="Interior_lora.safetensors")
                    pipe.fuse_lora(lora_scale=0.7)
                    output = gen_controlnet(pipe, 
                                        prompt, trigger_words="", 
                                        neg=negative_prompt, 
                                        num_images=num_outputs, 
                                        height=height, width=width, 
                                        image=image_controlnet,
                                        seed = seed)
                
                if output:
                    st.toast('Your image has been generated!', icon='😍')
                    # Save generated image to session state
                    st.session_state.generated_image = output

                    # Displaying the image
                    for image in st.session_state.generated_image:
                        with st.container():
                            # Display image with specified width to make it appear smaller
                            st.image(image, caption="Generated Image 🎈", width="stretch")  # Set width as desired

                            # Add image to the list
                            all_images.append(image)

                    # Save all generated images to session state
                    st.session_state.all_images = all_images
    

if __name__ == "__main__":
    
    main()

