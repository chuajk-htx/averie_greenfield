
# 5. Model Setup - in this case its transfer learning 
import torch
import torch.nn as nn
import torchvision.models as torchvision_models
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR, CosineAnnealingLR, OneCycleLR
import math

import timm  # PyTorch Image Models
from tqdm import tqdm
from InjectAttention_v7_2 import inject_attention


# Function to load model from either torchvision or timm
def load_pretrained_model(model_spec, unfreeze_last_n=0):
    """Load model, freeze layers, and replace classifier."""
    
    model_name, source = model_spec[0], model_spec[1]
    
    # Load from torchvision or timm
    if source == 'torchvision' and hasattr(torchvision_models, model_name):
        print(f"Loading {model_name} from torchvision")
        print(f"Torch version during training: {torch.__version__}")
        model = getattr(torchvision_models, model_name)(pretrained=True)
    
    elif source == 'timm':
        try:
            print(f"Loading {model_name} from timm")
            print(f"Timm version during training: {timm.__version__}")
            model = timm.create_model(model_name, pretrained=True)
            print(f"✅ Loaded {model_name} with pretrained weights")

        except Exception as e:
            if "No pretrained weights exist" in str(e):
                pretrained_cfg = timm.get_pretrained_cfg(model_name)

                if pretrained_cfg and pretrained_cfg.has_weights:
                    print(f"⚠️ Weights exist but failed to load, trying force download")
                    try:
                        model = timm.create_model(model_name, pretrained=True, force_download=True)

                    except Exception as e:
                        print(f"⚠️ Force download failed: {e}")
                else:
                    print(f"⚠️ No pretrained weights available")

                #print(f"❌ Model '{model_name}' exists but has NO pretrained weights.")
            #raise ValueError(f"Model {model_name} not found in torchvision or timm!")

    else:
        raise ValueError(f"Unknown source. Use 'torchvision' or 'timm'")

    # Freeze all layers
    for param in model.parameters():
        param.requires_grad = False

    # If need partial freezing (e.g., unfreeze last few layers)
    if unfreeze_last_n > 0: # need declare in fxn arg when using
        layers = list(model.children())[-unfreeze_last_n:]
        for layer in layers:
            for param in layer.parameters():
                param.requires_grad = True

    return model


def replace_classifier(model, num_classes, dropout, device):
    """Unified classifier replacement for all model types."""

    model = model.to(device)  # Ensure entire model is on correct device

    # Determine which attribute contains the classifier
    classifier_attr = None
    for attr in ['fc', 'classifier', 'head']:
        if hasattr(model, attr):
            classifier_attr = attr
            break
    if not classifier_attr:
        print("Model has no recognizable classifier (fc/classifier/head)")
        #raise ValueError("Model has no recognizable classifier (fc/classifier/head)")

    # Get the original classifier layer
    original_classifier = getattr(model, classifier_attr)
    print(original_classifier)
    
    # Extract input features

    # Case 1: Classifier is Sequential (common in torchvision)
    if isinstance(original_classifier, nn.Sequential):
        linear_layers = [layer for layer in original_classifier if isinstance(layer, nn.Linear)]
        if not linear_layers:
            print(f"Sequential {classifier_attr} contains no Linear layer")
            #raise ValueError(f"Sequential {classifier_attr} contains no Linear layer")
        num_features = min(layer.in_features for layer in linear_layers) # Find the smallest in_feature Linear layer in Sequential
    
    # Case 2: Direct Linear layer (common in some models)
    elif isinstance(original_classifier, nn.Linear):
        num_features = original_classifier.in_features

    # Case 3: Custom classifier structure (e.g., some TIMM models)
    else: # Try to get in_features if it exists (some custom heads may expose this)
        if hasattr(original_classifier, 'in_features'):
            num_features = original_classifier.in_features
        else:
            print("Unsupported {classifier_attr} type: {type(original_classifier)}.")
            #raise ValueError(
            #    f"Unsupported {classifier_attr} type: {type(original_classifier)}. "
            #    "Expected Sequential/Linear or object with in_features attribute."
            #)

    # Create new classifier
    new_classifier = nn.Sequential(
        nn.Linear(num_features, 512),
        nn.ReLU(),
        nn.Dropout(dropout),
        nn.Linear(512, num_classes)
    ).to(device)

    # Initializes weights with kaiming and biases to zeros if they exist
    for layer in new_classifier:
        if isinstance(layer, nn.Linear):
            nn.init.kaiming_normal_(layer.weight, mode='fan_in', nonlinearity='relu')
            if layer.bias is not None:
                nn.init.constant_(layer.bias, 0)

    # Replace the classifier
    setattr(model, classifier_attr, new_classifier)
    
    return model



def get_model(config, model_spec, attention_layers):

    # 1. Load the base model architecture first and 2. freezing all layers
    model = load_pretrained_model(model_spec).to(config.device)
#    model = models.densenet121(pretrained=True).to(config.device)  # Or whatever architecture D-NetPAD uses
    # If D-NetPAD uses a custom architecture, # 1. Initialize the model architecture: model = DNetPAD_Model().to(config.device)  # Instantiate custom model from the .py model file

    if config.use_attention:
        model = inject_attention(model, model_spec, attention_type=config.attention_type, layers=tuple(attention_layers), device=config.device)

    # 3. Modify classifier
    model = replace_classifier(model, config.num_classes, config.dropout, config.device)

    # Get ALL trainable parameters (dynamic) # Selects what to train
    trainable_params = [ 
        param for param in model.parameters() 
        if param.requires_grad
    ]
    print("Number of trainable params:", sum(p.numel() for p in trainable_params))

    # 5. Define optimizer (now handles unfrozen layers too)
    optimizer = optim.Adam(trainable_params, # Will include classifier + any unfrozen layers
                           lr=config.lr, 
                           weight_decay=config.weight_decay)#0.0001)

    scheduler = StepLR(optimizer, step_size=config.scheduler_step, gamma=config.scheduler_gamma)  # Reduce LR every 5 epochs, gamma = 0.1
#    scheduler = CosineAnnealingLR(optimizer, T_max=8*(math.ceil(6026/32)), eta_min=0.01)
#    scheduler = OneCycleLR(optimizer, max_lr = 0.005, epochs=8, steps_per_epoch=(math.ceil(6026/32)), pct_start=0.3, anneal_strategy="cos")

    # Check if on correct device
    #for name, param in model.named_parameters():
    #    if param.device.type != config.device:
    #        print(f"{name} is on {param.device}, expected {config.device}")

    model = model.to(config.device)

#    print(model)

    return model, optimizer, scheduler