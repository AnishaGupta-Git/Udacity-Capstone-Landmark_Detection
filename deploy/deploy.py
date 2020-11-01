# # Dependencies

# This file assumes that we have the following dependencies available:
# * *.pt : PyTorch saved model file
# * "description.json" : Has information about all the landmark classes
# * "thresholds.json" : Has information about the minimum confidence level for prediction

# %%
#!pip install gradio


# %%
import os
import PIL
import json
import torch
from torchvision import transforms

import requests
import io
import gradio as gr

# %%
PATH_MODEL = "./resnet34_bs-128_lr-3e3_ep-50.pt"
CLASS_LABELS =  ['Auroville', 'Buddhist Sanchi', 'Charminar', 'Chhatrapati Shivaji Terminus', 'Dakshineshwar', 'Gateway Of India', 'Golden Temple', 'Hampi', 'Hawa Mahal', 'Howrah Bridge', 'Humayun Tomb', 'India Gate', 'Jagannath Puri', 'Jantar Mantar', 'Jog Falls', 'Kanchenjunga', 'Lotus Temple', 'Meenakshi Temple', 'Mysore Palace', 'Qutub Minar', 'Red Fort', 'Sun Temple', 'Taj Mahal', 'Victoria Memorial', 'Wagah Border'] 


# %%
description = json.load(open("./description.json","r"))
thresholds = json.load(open("./thresholds.json","r"))

# Make sure all class names are all right!
assert set(description) == set(thresholds)
assert set(CLASS_LABELS) == set(description)


# %%
# Reads an image from img_path into a PIL.Image
# img_path: Can be a URL or a local file path
def img_path_to_data(img_path):
    # Check if the path is an image URL or local file
    if "http" in img_path or "ftp" in img_path or "www" in img_path:
        response = requests.get(img_path)
        img_data = PIL.Image.open(io.BytesIO(response.content))
    else:
        img_data = PIL.Image.open(img_path)
    
    # Convert to RGB for images with 4 layers like PNGs
    return img_data.convert("RGB")


# %%
# img_data: PIL.Image format image
# model: torch.model
# Returns a tuple of the class_id [0,24] with maximum probability and a torch.tensor of probabilities of all class_id's
def predict_on_img_data(img_data, model):
    test_transform = transforms.Compose([
                transforms.Resize((224,224)),
                transforms.ToTensor(),
                transforms.Normalize(
                    [0.485, 0.456, 0.406],  # Recommended values for ImageNet trained models
                    [0.229, 0.224, 0.225])
                ])
        
    img_tensor = test_transform(img_data).unsqueeze(0)
    model.eval()
    with torch.no_grad():
        proba = torch.nn.functional.softmax(model(img_tensor)).flatten()
    class_id = torch.argmax(proba).item()
    return (class_id,proba)


# %%
# Wrapper function, called by front-end UI framework
def predict(img):
    class_id, pred = predict_on_img_data(img,model)
    class_label = CLASS_LABELS[class_id]
    pred_float = {CLASS_LABELS[i]:float(pred[i]) for i in range(len(pred))}

    desc = description[class_label]
    if pred_float[class_label] < thresholds[class_label]:
        desc = '<p style="color:rgb(255, 0, 0);"><b>The model is not very confident about this prediction. Maybe this is something unfamiliar or confusing to it!</b></p>'
    return pred_float, desc

# # Create front-end user interface

# %%
model = torch.load(PATH_MODEL).cpu()


# %%
ui_title = "Indian Landmark Detection"
ui_desc = "Capstone Project by Anisha Gupta for Udacity MLE-Nanodegree, October 2020"


# %%
front_end = gr.Interface(fn= predict,
            inputs=    gr.inputs.Image(type="pil", label="Select image to check"),
            outputs=   [gr.outputs.Label(type="confidences", num_top_classes=3, label= "Predicted Landmark"),
                        gr.outputs.HTML(label= "Description")],
            title=      ui_title,
            description= ui_desc,
            allow_flagging= False)
front_end.launch()