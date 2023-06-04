# Style Transfer App

## Introduction
A small application used to perform **style transfer**. Style transfer is a computer vision technique that allows us to recompose the content of an image in the style of another. If you’ve ever imagined what a photo might look like if it were painted by a famous artist, then style transfer is the computer vision technique that turns this into a reality.

### Built With
* [![Python][Python_img]][Python-url]
* [![Streamlit][Streamlit_img]][Streamlit-url]
* [![PyTorch][torch-img]][torch-url]

## Technical overview

## How to install
### Github

**Installation Dependencies:** required libraries are located in requirements.txt
* pandas 
* numpy
* streamlit
* opencv-python
* torch
* torchvision
* matplotlib

**Install Project:**

```
git clone https://github.com/nhduong1203/TransferApp.git
cd TransferApp
pip install requirements.txt
streamlit run main.py
```
### Docker
**Link Docker:** [Docker Repo][docker-url]

**Install Project:**
```
# Pull docker
docker pull sieucun/transfer_app

# Run image
docker run --name test -p 8501:8501 transfer_app

Access the application at: http://localhost:8501
```
## Experiments
### Deep Learning Model
The deep learning model used in this project is named VGG-19, its architecture is shown in the following image: 
<img src="./images/source_images/vgg19-architecture.png" width="900">


## Acknowledgments

[Python_img]: https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54
[Python-url]: python.org

[Streamlit_img]: https://user-images.githubusercontent.com/7164864/217935870-c0bc60a3-6fc0-4047-b011-7b4c59488c91.png
[Streamlit-url]: streamlit.io

[torch-img]: https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=for-the-badge&logo=PyTorch&logoColor=white
[torch-url]: https://pytorch.org/

[docker-url]: https://hub.docker.com/repository/docker/sieucun/transfer_app/general
