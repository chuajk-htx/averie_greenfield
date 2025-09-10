
# ---------------------------
# Unified Attention Injection
# ---------------------------
import torch
import torch.nn as nn
from AttentionMechanism_v2 import ChannelAttention, HybridAttention
from torchvision.models.densenet import _DenseLayer
import timm
from timm.layers import SqueezeExcite

# Checks for SE inside a block. For models that have existing attention module, to append attention after blocks without modifying existing SE
def block_contains_se(block):
    for name, submodule in block.named_modules():
        if isinstance(submodule, SqueezeExcite):
            return True
    return False


def inject_attention(model, model_spec, attention_type="ChannelAttention", layers=(), device=None):
    """
    Inject attention module after each block of specified layers for supported models.

    Args:
        model: The base model (e.g., torchvision/timm model).
        model_name: Name string to infer model type (e.g., 'resnet50', 'mobilenetv3').
        layers_to_modify: Dict[str, List[int]] or List[str], indicating which layers/blocks to inject into.
        AttentionMechanism: Callable that takes block_channels and returns attention module.

    Returns:
        model: Modified model with attention modules.
    """
    model_name, model_source = model_spec[0], model_spec[1]

    modified = []

    def attach_attention(module, channels):
        if attention_type == "ChannelAttention":
            return nn.Sequential(module, ChannelAttention(channels))
        elif attention_type == "Hybrid":
            return nn.Sequential(module, HybridAttention(channels))
        else:
            return module

    if 'res' in model_name: # for resnet and resnet
        for layer_name in ['layer1', 'layer2', 'layer3', 'layer4']:
            if layer_name in layers or "all" in layers:
                layer = getattr(model, layer_name) # Gets the layer from the model dynamically e.g. layer = model.layer1
                
                for i, block in enumerate(layer):
                    if hasattr(block, 'conv3'): # to handle diff Resnet variants
                        block_channels = block.conv3.out_channels  # Bottleneck
                    else:
                        block_channels = block.conv2.out_channels  # BasicBlock (Resnet18/34) # Assumes each block ends with conv2
                    new_block = attach_attention(block, block_channels) # Gets its output channels to use in the attention module
                    setattr(layer, str(i), new_block) # Replaces the original BasicBlock with the new Sequential(block, attention) module; setattr trigger PyTorch to re-register the module properly (direct assignment to indexing doesn't trigger registration)
                    modified.append(f"{layer_name}.{i}")

    elif 'densenet' in model_name:
        found_block = False

        for db_name in ['denseblock1', 'denseblock2', 'denseblock3', 'denseblock4']:
            if db_name in layers or "all" in layers:
                try:
                    block = getattr(model.features, db_name) # get denseblocks in Sequential through 'features' children
                    found_block = True

                    for name, subblock in block.named_children(): # get subblock items through children - 'denselayerX' , 'subblock'
                        conv_layers = []
                        for subname, submodule in subblock.named_children(): # subname is conv1/2, submodules e.g. conv2d, batchnorm etc  # get 2nd convolutional layer (conv2) inside a DenseLayer - each DenseLayer typ has 2 convs - conv1 reduces dimensionality (bottleneck), conv2 produces actual output that is concatenated to densenet feature map
                            if isinstance(submodule, nn.Conv2d):
                                conv_layers.append(submodule)
                        if not conv_layers:
                            continue
                        channels = conv_layers[-1].out_channels

                        if 'timm' in model_source:
                            block[name] = attach_attention(subblock, channels)
                        elif 'torch' in model_source:
                            setattr(block, name, attach_attention(subblock, channels)) # or equivalently block._modules[name] = attach_attention(subblock, channels) 
                        
                        modified.append(f"{db_name}.{name}")

                except AttributeError:
                    continue 

    elif "mobilenetv3" in model_name:

        mobilenetv3_layer_map = {
            "layer1": [0],       # blocks[0]
            "layer2": [1],       # blocks[1]
            "layer3": [3],       # blocks[3]
#            "layer1_layer2": [0, 1],
#            "layer2_layer3": [1, 3],
#            "layer1_layer3": [0, 3],
        }

        block_container = getattr(model, 'blocks', getattr(model, 'features', None))

        selected_block_indices = []
        if "all" in layers:
            selected_block_indices = list(range(len(block_container)))
        else:
            for l in layers:
                selected_block_indices.extend(mobilenetv3_layer_map.get(l, []))        


        #for name, block in block_container.named_children(): # outcome of for name, block in block_con.named_children(): = for i, block in enumerate(block_con):                    
        for i, block in enumerate(block_container): # outcome of for name, block in block_con.named_children(): = for i, block in enumerate(block_con):

            if i not in selected_block_indices:
                continue  # skip blocks not requested
            
            for j, subblock in enumerate(block):  # block is InvertedResidual or DepthwiseSeparableConv

                    # CASE 1: Block already has an SE slot
                    if hasattr(subblock, 'se'):
                        # Skip if SE is already implemented (not Identity)
                        if not isinstance(subblock.se, nn.Identity):
                            continue  

                        if hasattr(subblock, 'conv_dw'):
                            out_channels = subblock.conv_dw.out_channels
                        else:
                            continue # fallback to something else or skip injection

                        print(out_channels)
                        subblock.se = ChannelAttention(out_channels)
                        modified.append(f"blocks.{i}.{j}.se")
                        print(f"Injected ChannelAttention with out_channels={out_channels} at block {i}.{j}")
                        continue
                    
                    else:
                        print(f"❌ Skipping {subblock}, {type(subblock)} — not an SE block")

                    # CASE 2: No SE slot → attach attention after block output
                    #conv_layers = [m for m in block.modules() if isinstance(m, nn.Conv2d)]
                    #if not conv_layers:
                    #    continue  # skip if no conv layers found
                    #out_channels = conv_layers[-1].out_channels

                    #block[j] = attach_attention(block, out_channels)
                    #modified.append(f"blocks.{i}.{j}")

    elif "tf_efficientnet" in model_name:
        efficientnet_layer_map = {
            "layer1": [0],       # blocks[0]
            "layer2": [1],       # blocks[1]
            "layer3": [2],       # blocks[2]
#            "layer1_layer2": [0, 1],
#            "layer2_layer3": [1, 2],
#            "layer1_layer3": [0, 2],
        }

        block_container = getattr(model, 'blocks', getattr(model, 'features', None))

        selected_block_indices = []
        if "all" in layers:
            selected_block_indices = list(range(len(block_container)))
        else:
            for l in layers:
                selected_block_indices.extend(efficientnet_layer_map.get(l, []))        


        #for name, block in block_container.named_children(): # outcome of for name, block in block_con.named_children(): = for i, block in enumerate(block_con):                    
        for i, block in enumerate(block_container): # outcome of for name, block in block_con.named_children(): = for i, block in enumerate(block_con):

            if i not in selected_block_indices:
                continue  # skip blocks not requested
            
            for j, subblock in enumerate(block):  # block is InvertedResidual or DepthwiseSeparableConv

                    # CASE 1: Block already has an SE slot
                    if hasattr(subblock, 'se'):
                        # Skip if SE is already implemented (not Identity)
                        if not isinstance(subblock.se, nn.Identity):
                            continue  

                        if hasattr(subblock, 'conv_exp'):
                            out_channels = subblock.conv_exp.out_channels
                        else:
                            continue # fallback to something else or skip injection

                        print(out_channels)
                        subblock.se = ChannelAttention(out_channels)
                        modified.append(f"blocks.{i}.{j}.se")
                        print(f"Injected ChannelAttention with out_channels={out_channels} at block {i}.{j}")
                        continue
                    
                    else:
                        print(f"❌ Skipping {subblock}, {type(subblock)} — not an SE block")

    # Verification and debugging
    if modified:
        print(f"✅ Successfully injected {attention_type} at: {modified}")
        # Verify injection by checking for attention modules
        attention_modules = [name for name, module in model.named_modules() 
                     if isinstance(module, ChannelAttention)]
        
        #attention_modules = [name for name, _ in model.named_modules() if 'ChannelAttention' in name]
        print(f"🔍 Found {len(attention_modules)} attention modules in model")

        #attention_submodules = []
        #for namee, blo in model.named_modules():
        #    module_class_name = blo.__class__.__name__.lower()
        #    if 'ChannelAttention' in module_class_name:
        #        attention_submodules.append(namee)
        #print(f"🔍 Found {len(attention_submodules)} attention modules in model")


    else:
        print("⚠️ No attention layers were injected. Model structure analysis:")
        print(f"Model class: {model.__class__.__name__}")
        print("Top-level modules:")
        for name, module in model.named_children():
            print(f"- {name}: {module.__class__.__name__}")
#            if name == 'features':
#                print("  Features submodules:")
#                for i, feat in enumerate(module.children()):
#                    print(f"  {i}: {feat.__class__.__name__}")
#                    if 'DenseBlock' in feat.__class__.__name__:
#                        print(f"    (Contains {len(feat)} layers)")



    return model