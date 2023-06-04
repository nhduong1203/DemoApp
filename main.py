import streamlit as st
import torch
from torchvision import transforms, utils, models
from torchvision.models import resnet50
import numpy as np
import torch.nn as nn
from PIL import Image
from train import run_style_transfer
from utils import Normalization, image_loader, imshow, download_button
import io

@st.cache_resource
def load_model():
    """
    Loads a pre-trained VGG-19 model.

    Returns:
        torch.nn.Module: The loaded VGG-19 model.

    """
    cnn = models.vgg19(pretrained=True).features.to(device).eval()
    return cnn



if __name__ == '__main__':
    # define mean normalization, std normalization for VGG-19 model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cnn_normalization_mean = torch.tensor([0.485, 0.456, 0.406]).to(device)
    cnn_normalization_std = torch.tensor([0.229, 0.224, 0.225]).to(device)
    imsize = 128

    # download pretrained VGG-19 model
    with st.spinner("Downloading pretrained model may take a few minutes. Please wait ^^ ..."):
        cnn = load_model()

    # uploading images
    with st.sidebar:
        with st.form("Upload File"):
            origin_file = st.file_uploader("Chọn ảnh gốc")
            style_file = st.file_uploader("Chọn ảnh phong cách")
            submit = st.form_submit_button("Submit")

    
    if origin_file is not None and style_file is not None:
        # show uploaded images
        img1, img2 = st.columns(2)
        with img1:
            st.image(transforms.Resize((512, 512))(Image.open(origin_file)), caption = "Origin image")
        with img2:
            st.image(transforms.Resize((512, 512))(Image.open(style_file)), caption = "Style image")

        # load images to a tensor 
        origin = image_loader(origin_file)
        style = image_loader(style_file)
        input = origin.clone()                 # initialize the input (generated image)
        
        # run style transfer ... 
        with st.columns(5)[2]:
            with st.spinner("Transfer..."):
                output = run_style_transfer(cnn, cnn_normalization_mean, cnn_normalization_std,
                                    origin, style, input)
            
        # show the generated image
        output = imshow(output)
        with st.columns(4)[1]:
            st.image(output, width = 350)

        # download the generated image
        byte_stream = io.BytesIO()
        output.save(byte_stream, format='PNG')
        output_bytes = byte_stream.getvalue()
        with st.columns(5)[2]:
            download_button_str = download_button(output_bytes, "transfer_style_image.png","Download image")
            st.markdown(download_button_str, unsafe_allow_html=True)

           
        st.stop()
        

