import streamlit as st
import os
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torchvision.transforms import functional as Fv
from PIL import Image
import matplotlib.pyplot as plt
from model import *
from time import ctime

# turn on streamlit wide-mode by default
st.set_page_config(layout="wide")

#device = "cuda" if torch.cuda.is_available() else "cpu"
device = "cpu"
uploaded_file = None
img_size = 256

# update on model change
model = "rguest_model.pth"
last_mod_time = os.path.getmtime(model)
updated_time = ctime()
current_mod_time = os.path.getmtime(model)
if current_mod_time != last_mod_time:
    last_mod_time = current_mod_time
    updated_time = ctime()
    st.experimental_rerun()

def soft_clip(tensor, q=0.1):
    lower, upper = tensor.quantile(q), tensor.quantile(1-q)
    return (1e-6 + tensor - lower) / (1e-6 + upper - lower)

def channel_to_image(tensor, q = 0.05, norm = False, cmap="cubehelix"):
    # input is (1, width, height) tensor
    tensor = tensor.detach().cpu()
    if norm:
        tensor = F.softmax(tensor, dim=0)
        tensor = F.softmax(tensor, dim=1)
        tensor = F.sigmoid(tensor)
        tensor = soft_clip(tensor, q)
    tensor = tensor.numpy()
    tensor = np.clip(tensor, 0, 1)
    tensor = np.moveaxis(tensor, 0, -1)
    colored_img = plt.get_cmap(cmap)(tensor)
    colored_img_rgb = (colored_img[:, :, :3] * 255).astype(np.uint8)
    #colored_img_rgb = ((1e-6 + colored_img_rgb - colored_img_rgb.min()) / 
    #                    (1e-6+ colored_img_rgb.max() - colored_img_rgb.min()))
    return np.rot90(np.rot90(np.rot90(colored_img_rgb)))

def run_model(image, layer=0):
    img = np.array(image)
    img = np.moveaxis(img, -1, 0)
    x = torch.tensor(img).to(device).unsqueeze(0).float() / 255
    x = model.encoder.conv1(x)
    x = model.encoder.bn1(x)
    x = model.encoder.relu(x)
    x = model.encoder.maxpool(x)
    if layer >= 1:
        x =  model.encoder.layer1(x)
    if layer >= 2:
        x =  model.encoder.layer2(x)
    if layer >= 3:
        x =  model.encoder.layer3(x)
    if layer >= 4:
        x =  model.encoder.layer4(x)
    return x.squeeze(0)

def display(xrays, q = 0.05, norm = False, cmap="bone"):
    channels, width, height = xrays.shape
    # 4 columns for 256x256 images, proportionally more for smaller
    # N images of width 256 are displayed in 4 columns and ceil(N/4) rows
    # N images of width 128 are displayed in 8 columns and ceil(N/4) rows
    # 4 * 256 / 128 = ...
    W = min(16,max(5,int(320 / width)))
    L = len(xrays)
    cols = st.columns(W)
    col_assignment = [i % W for i in range(L)]
    for img, j in zip(xrays, col_assignment):
        cols[j].image(channel_to_image(img,q,norm, cmap), use_column_width=True)


def center_crop(PIL_image) -> Image:
    # I want a proportions-preserving square crop from the center
    # of the image. use torchvision if it's more succint
    # e.g. if img_size is 256, and the image is 512x1024,
    # I want to crop the center 512x512, discarding 256 pixels
    # if image is small like 56 x 64, center crop to 56 x 56 
    # then resize to 256 x 256 bicubically
    # from the top and bottom.
    width, height = PIL_image.size
    if width > height:
        left = (width - height) // 2
        right = left + height
        top = 0
        bottom = height
    else:
        top = (height - width) // 2
        bottom = top + width
        left = 0
        right = width
    PIL_image = PIL_image.crop((left, top, right, bottom))
    PIL_image = PIL_image.resize((img_size, img_size), Image.BICUBIC)
    return PIL_image

@st.cache_data
def rgb_image_to_tensor(image: Image) -> torch.tensor:
    # Convert a PIL image to a PyTorch tensor
    # shuffling dimensions and squeezing so we haveh
    # (w, h 3) -> (1, 3, w, h)  
    img = image.convert("RGB")
    # now reshape from w x h x 3 to 3 x w x h
    img = np.array(img)
    # if values are from 0 to 255 (check first), normalize to 0 to 1 floats
    img = img if img.max() > 1 else img / 255
    img = Fv.to_tensor(img).unsqueeze(0) 
    return img

def tensor_to_rgb_image(tensor: torch.tensor) -> Image:
    # Convert a PyTorch tensor to a PIL image
    # reshaping from 1 x 3 x w x h to w x h x 3
    img = tensor.squeeze(0).detach().cpu().numpy()
    img = np.moveaxis(img, 0, -1)
    img = np.clip(img, 0, 1)
    img = (255 * img).astype(np.uint8)
    return Image.fromarray(img)

## app

model = torch.load("rmodel.pth",map_location=torch.device("cpu")).to(device)
model.load_state_dict(torch.load("rguest_model.pth", map_location=torch.device("cpu")))

uploaded_file = st.file_uploader("Choose image", label_visibility="hidden")
css = '''
    <style>
        [data-testid='stFileUploader'] {
            width: max-content;
        }
        [data-testid='stFileUploader'] section {
            padding: 0;
            float: left;
        }
        [data-testid='stFileUploader'] section > input + div {
            display: none;
        }
        [data-testid='stFileUploader'] section + div {
            float: right;
            padding-top: 0;
        }

    </style>
    '''

st.markdown(css, unsafe_allow_html=True)
sidebar, main = st.columns([1, 5])

if uploaded_file is None:
    main.warning("Please upload an image file")
    st.stop()
else:
    image = Image.open(uploaded_file)
    image = center_crop(image)    

    image = image.convert("RGB")
    ft_show = sidebar.radio(
        "Choose view",
        ["Layer 1", "Layer 2", "Layer 3","Layer 4","Input"],
        index=3)
    q = sidebar.slider('Contrast',0.0,0.2,value=0.0,step=0.01)
    norm = sidebar.checkbox("Normalize each", value=False)
    cmap = sidebar.radio("Colormap", ["bone","cubehelix","jet","coolwarm"])
with main:
    if ft_show == "Input":
        cx, c1, c2 = main.columns([1,2,2])
        view_source = c1.checkbox("src", value=False, key="source")
        view_output = c2.checkbox("out", value=True, key="output")
        if view_output:
            x = rgb_image_to_tensor(image).to(device)
            output = model.forward(x)

        if view_output and (view_source):
            d1, d2, d3 = st.columns(3)
            d1.image(image, use_column_width=True, caption="Source image")
            d2.image(tensor_to_rgb_image(soft_clip(output,q)), caption="Output Image", use_column_width=True)
            error = abs(output - x)
            d3.image(tensor_to_rgb_image(error), caption="Error Image", use_column_width=True)
        if view_output and not (view_source):
            d1, d2,d3 = st.columns([1,3,1])
            d2.image(tensor_to_rgb_image(soft_clip(output,q)).resize((600,600),Image.Resampling.NEAREST), caption="Output Image", use_column_width=False)
    elif ft_show.startswith("Layer"):
        num = int(ft_show.split()[-1])
        x_images = run_model(image, num)
        display(x_images, q, cmap = cmap)
