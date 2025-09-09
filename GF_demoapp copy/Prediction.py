
### Model Prediction 

import cv2
import random
from matplotlib import pyplot as plt
import numpy as np 
import pandas as pd
import json
import torch
from PIL import Image
import torchvision.transforms as transforms
from torch.optim.lr_scheduler import StepLR
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image
import streamlit as st


def set_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    # For MPS-specific reproducibility (if available)
    if torch.backends.mps.is_available():
        if hasattr(torch.backends.mps, 'deterministic'):
            torch.backends.mps.deterministic = True
        torch.mps.manual_seed(seed)

    # Additional deterministic settings
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

import os
from Config import Config
from TransferLearning_v3 import get_model

def predict(imgs, clahe_imgs):

    config = Config()
    set_seeds(config.seed)

    # ---- Recreate model and load weights -----


    model_spec = ['mobilenetv3_large_100', 'timm']
    attention_layers = "all"
    model_weights_path = '/Users/averie/Documents/Algos/Open Source Code Tests/FinalTraining/results/20250907_000418/best_model.pth'
    classmapping_file = '/Users/averie/Documents/Algos/Open Source Code Tests/FinalTraining/results/20250907_000418/class_mapping.json'

    model, optimizer, scheduler = get_model(config, model_spec, attention_layers)

    checkpoint = torch.load(model_weights_path, map_location=config.device)
    model.load_state_dict(checkpoint['model_state_dict'], strict=False)  # strict=False allows partial load

    # 3. Re-create optimizer after freezing/unfreezing layers
    # Only trainable params will be in the optimizer
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(trainable_params, lr=config.lr, weight_decay=config.weight_decay)
    # 4. Re-create scheduler if needed
    scheduler = StepLR(optimizer, step_size=config.scheduler_step, gamma=config.scheduler_gamma)


    # ---- Mapping ------

    with open(classmapping_file) as f:
        mapping = json.load(f)

    class_to_id = mapping["class_to_id"]
    id_to_class = mapping["id_to_class"]


    # ---- Eval mode -----

    model = model.to(config.device)
    model.eval()

    # Apply transformation to tensor

    def expand_to_3_channels(x):
        return x.expand(3, -1, -1)

    transform = transforms.Compose([
        #transforms.ToImage(),
        transforms.Resize((224, 244)),
        transforms.ToTensor(),
        transforms.Lambda(expand_to_3_channels),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                            std=[0.229, 0.224, 0.225]) 
    ])

    predictions = []
    k = 2

    forgrad = []
    gradcam_imgs_imgpixel = []

    for img, clahe_img in zip(imgs, clahe_imgs):
        #image = cv2.imread(img, cv2.IMREAD_GRAYSCALE) 
        image = Image.open(img)
        clahe_image = Image.open(clahe_img)

        input_tensor = transform(image).unsqueeze(0).to(config.device) # use for model input, not for visualisation

        with torch.no_grad():
            output = model(input_tensor)
            probs = torch.nn.functional.softmax(output, dim=1)
            forgrad_confidence, forgrad_predicted_class = torch.max(probs, dim=1)
            forgrad.append(forgrad_predicted_class)

            for img_name, prob_vec in zip([img.split('/')[-1]], probs):
                prob_dict = {
                    id_to_class[pred_idx]: prob.item()
                    for pred_idx, prob in enumerate(prob_vec)
                }
                predictions.append({"Image": img_name, **prob_dict})


        target_layer = model.blocks[-1]#model.conv_head#model.blocks[-1]

        #if pred[0]['prob'] > 0.7:# and predicted_class.item() == true_label: 7
        """
        Checks if the model is confident in its prediction. Good for filtering out uncertain predictions that produce noisy or diffuse Grad-CAMs
        ⚠️ But: confident ≠ correct — the model might be confidently wrong!

        predicted_class.item() == true_label;   Ensures the model prediction is actually correct.
        Use this if you're debugging your model and want to see what it looks at when it's right
        """
        # Run Grad-CAM here
        # Grad-CAM (fresh tensor)
        gradcam_input = transform(image).unsqueeze(0).to(config.device)
        gradcam_input.requires_grad_()   # re-enable gradient tracking
        cam = GradCAM(model=model, target_layers=[target_layer])
        grayscale_cam = cam(gradcam_input, targets=[ClassifierOutputTarget(forgrad_predicted_class.item())])[0]

        # Convert original img to RGB and scale for visualization, show_cam_on_image needs a numpy array (H, W, 3)
        img_gray = np.array(clahe_image.resize((224, 224))) / 255.0   # shape (224,224)
        img_rgb = np.stack([img_gray]*3, axis=-1).astype(np.float32)    # shape (224,224,3)

        # Resize Grad-CAM heatmap to match image size
        #img_rgb = np.stack([np.array(image.resize((224, 224))) / 255.0]*3, axis=-1).astype(np.float32)
        grayscale_cam_resized = cv2.resize(grayscale_cam, (img_rgb.shape[1], img_rgb.shape[0]))

        # Overlay heatmap
        vis = show_cam_on_image(img_rgb, grayscale_cam_resized, use_rgb=True)
        plt.imshow(vis); plt.show()
        #st.image(vis)
        gradcam_imgs_imgpixel.append(vis)

    # Build dataframe where each class gets its own column of probabilities
    predictions_df = pd.DataFrame(predictions)
    #display(predictions_df)

    # Drop "Image" column so only classes remain
    class_cols = [c for c in predictions_df.columns if c != "Image"]
    print(class_cols)
    topk_preds = []

    for _, row in predictions_df.iterrows():
        img_name = row["Image"]
        #1. Turn the dictionary into a list of (class, probability) pairs
        probs = row[class_cols].to_dict()  # dict: {class: prob}
        #2. Sort based on probability/confi
        sorted_items = sorted(probs.items(), key=lambda x: x[1], reverse=True)
        #print(sorted_items, 'sorted')
        # 3. Take only the first k elements
        #topk = dict(sorted_items[:k])
        topk = sorted_items[:k]
        #print(topk, 'topk')
        # 4. Rebuild dict with 'Image' + top 2 class predictions
        #topk_dict = {"Image": img_name, **topk}
        topk_library = [{"name": img_name, "class": cls, "prob": prob} for cls, prob in topk]
        topk_preds.append(topk_library)
        # 5. (Optional) Unpack nicely with enumerate
        for i, (cls, prob) in enumerate(topk):
            print(f"Rank {i+1}: {cls} ({prob:.2f})")

#        for rank, pred in enumerate(topk_preds, start=1):
#            if rank == 1:
#                st.write('rank', rank, pred[0]['prob'], 'pred')


    #topk_df = pd.DataFrame(topk_preds)

    return predictions_df, topk_preds, gradcam_imgs_imgpixel



def Grad(prob):

    target_layer = model.blocks[-1]

    if prob > 0.7:# and predicted_class.item() == true_label: 7
        """
        Checks if the model is confident in its prediction. Good for filtering out uncertain predictions that produce noisy or diffuse Grad-CAMs
        ⚠️ But: confident ≠ correct — the model might be confidently wrong!

        predicted_class.item() == true_label;   Ensures the model prediction is actually correct.
        Use this if you're debugging your model and want to see what it looks at when it's right
        """
        # Run Grad-CAM here
        cam = GradCAM(model=model, target_layers=[target_layer])
        grayscale_cam = cam(input_tensor=input_tensor, targets=[ClassifierOutputTarget(predicted_class.item())])[0]

        # Convert original img to RGB and scale for visualization
        img_rgb = np.stack([np.array(img.resize((224, 224))) / 255.0]*3, axis=-1).astype(np.float32)

        # Overlay heatmap
        vis = show_cam_on_image(img_rgb, grayscale_cam, use_rgb=True)
