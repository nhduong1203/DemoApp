from PIL import Image
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
import torch
import torch.nn as nn
import base64
import os
import json
import pickle
import uuid
import re

import streamlit as st
import pandas as pd

imsize = 512 if torch.cuda.is_available() else 128  
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

loader = transforms.Compose([
    transforms.Resize((imsize, imsize)),  
    transforms.ToTensor()])

unloader = transforms.ToPILImage()
plt.ion()

def image_loader(image_name):
    """
    Loads and prepares an image for processing.

    Args:
        image_name (str): The path or name of the image file.

    Returns:
        torch.Tensor: The loaded and preprocessed image tensor.

    """
    image = Image.open(image_name)     # load image
    image = loader(image).unsqueeze(0) # expand the image for batch_size = 1
    return image.to(device, torch.float)




def imshow(tensor):
    """
    Transfer tensor back to image.

    Args:
        tensor (torch.tensor): the input tensor.

    Returns:
        PILImage: PILImage obtained from tensor.
    """
    image = tensor.cpu().clone()  # we clone the tensor to not do changes on it
    image = image.squeeze(0)      # remove the fake batch dimension
    image = unloader(image)       # to PILImage
    return image 

def gram_matrix(input):
    """
    Calculate the Gram matrix of an input tensor.

    Args:
        input (torch.tensor): the input tensor.

    Returns:
        torch.Tensor: The Gram matrix of the input tensor.
    """
    a, b, c, d = input.size()                # Get the dimensions of the input tensor. For conv output, this is: a = batchsize, b = number_of_feature_maps, (c,d)=dimensions of a feature map
    features = input.view(a * b, c * d)      # Reshape the 2D tensor
    G = torch.mm(features, features.t())     # Calculate the Gram matrix
    return G.div(a * b * c * d)              # Normalize the Gram matrix by the number of elements
 



class Normalization(nn.Module):
    """
    A module for normalizing an image tensor.

    Args:
        mean (list, tuple, torch.Tensor): The mean values for normalization.
        std (list, tuple, torch.Tensor): The standard deviation values for normalization.
    """

    def __init__(self, mean, std):
        """
        Initializes a new instance of the Normalization module.
        """
        super(Normalization, self).__init__()
        self.mean = torch.tensor(mean).view(-1, 1, 1)
        self.std = torch.tensor(std).view(-1, 1, 1)

    def forward(self, img):
        # normalize ``img`` by subtracting the mean and dividing by the standard deviation
        return (img - self.mean) / self.std



def download_button(object_to_download, download_filename, button_text, pickle_it=False):
    """
    Generates a link to download the given object_to_download.

    Params:
    ------
    object_to_download:  The object to be downloaded.
    download_filename (str): filename and extension of file. e.g. mydata.csv,
    some_txt_output.txt download_link_text (str): Text to display for download
    link.
    button_text (str): Text to display on download button (e.g. 'click here to download file')
    pickle_it (bool): If True, pickle file.

    Returns:
    -------
    (str): the anchor tag to download object_to_download

    Examples:
    --------
    download_link(your_df, 'YOUR_DF.csv', 'Click to download data!')
    download_link(your_str, 'YOUR_STRING.txt', 'Click to download text!')

    """
    if pickle_it:
        try:
            object_to_download = pickle.dumps(object_to_download)
        except pickle.PicklingError as e:
            st.write(e)
            return None

    else:
        if isinstance(object_to_download, bytes):
            pass

        elif isinstance(object_to_download, pd.DataFrame):
            object_to_download = object_to_download.to_csv(index=False)

        # Try JSON encode for everything else
        else:
            object_to_download = json.dumps(object_to_download)

    try:
        # some strings <-> bytes conversions necessary here
        b64 = base64.b64encode(object_to_download.encode()).decode()

    except AttributeError as e:
        b64 = base64.b64encode(object_to_download).decode()

    button_uuid = str(uuid.uuid4()).replace('-', '')
    button_id = re.sub('\d+', '', button_uuid)

    custom_css = f""" 
        <style>
            #{button_id} {{
                background-color: rgb(255, 255, 255);
                color: rgb(38, 39, 48);
                padding: 0.25em 0.38em;
                position: relative;
                text-decoration: none;
                border-radius: 4px;
                border-width: 1px;
                border-style: solid;
                border-color: rgb(230, 234, 241);
                border-image: initial;

            }} 
            #{button_id}:hover {{
                border-color: rgb(246, 51, 102);
                color: rgb(246, 51, 102);
            }}
            #{button_id}:active {{
                box-shadow: none;
                background-color: rgb(246, 51, 102);
                color: white;
                }}
        </style> """

    dl_link = custom_css + f'<a download="{download_filename}" id="{button_id}" href="data:file/txt;base64,{b64}">{button_text}</a><br></br>'

    return dl_link
