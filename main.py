import streamlit as st
import torch-scatter
from torchvision import transforms, utils, models
from torchvision.models import resnet50
import numpy as np
import torch.nn as nn
from PIL import Image

def preprocess_image(image):
    # Load the image

    # Resize the image to 224x224
    image = image.resize((224, 224))

    # Convert the image to RGB
    image = image.convert("RGB")

    # Normalize the image
    image = np.array(image)
    image = image / 255.0  # Normalize pixel values to the range [0, 1]
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image = (image - mean) / std  # Normalize using mean and standard deviation

    # Expand dimensions
    image = np.expand_dims(image, axis=0)
    image = np.transpose(image, (0,3,1,2))
    image = torch.from_numpy(image).to(torch.double)

    return image

def change_layers(model):
    # model.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
    model.fc = nn.Linear(2048, 10, bias=True)
    return model


if __name__ == '__main__':
    
    model_path = "C:/Users/Hai Duong/.cache/torch/hub/checkpoints/resnet50-0676ba61.pth"
    model = resnet50()
    model.load_state_dict(torch.load(model_path))
    model = model.double()
    label = [i for i in range(1000)]
    
    uploaded_file = st.file_uploader("Chon file")
    if uploaded_file is not None:
        # st.image(uploaded_file)
        image = Image.open(uploaded_file)
        image = preprocess_image(image)
        print(model)
        
        model.eval()
        tensor_result = model(image)
        print(tensor_result)
        print(np.argmax(tensor_result.detach().numpy()))
        result = label[np.argmax(tensor_result.detach().numpy())]

        image = st.image(uploaded_file, caption=f'This is {result}', use_column_width=True)



# model = models.resnet50(pretrained=True)
