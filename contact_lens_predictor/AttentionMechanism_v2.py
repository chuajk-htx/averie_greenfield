
import torch
import torch.nn as nn

class ChannelAttention(nn.Module):
    def __init__(self, in_channels, reduction_ratio=4): 
        """
        in_channels = no. of channels in input 
        reduction ratio = controls how much we reduce the no. of features in the hidden layer (to make it smaller and faster)
        super() initializes the parent class (nn.Module) correctly
        """
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1) # to extract 1 value per channel (shape becomes [B, C, 1, 1])
#        self.max_pool = nn.AdaptiveMaxPool2d(1)

        hidden_channels = max(in_channels // reduction_ratio, 1)  # max(...,1) to prevent hidden channels from becoming 0 if in_channels < reduction_ratio
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, kernel_size=1, stride=1, padding=0),
            nn.Hardswish(inplace=True),
            nn.Conv2d(hidden_channels, in_channels, kernel_size=1, stride=1, padding=0),
            nn.Sigmoid()
        )


    def forward(self, x):
        #print(f"ChannelAttention forward: x.shape = {x.shape}")  # Expect [B, C, 1, 1] or [B, C, H, W]

        # [B, C, H, W] -> [B, C, 1, 1]
        y = self.avg_pool(x)  
        # Apply 1x1 Conv MLP
        y = self.conv(y)
        # Multiply attention weights
        return x * y.expand_as(x).to(x.device) # remove .view() flattening — Conv2d handles [B, C, 1, 1] natively.

        #print(f"🔁 Running forward in ChannelAttention: input shape {x.shape}")

        return x * channel_weights.expand_as(x).to(x.device) # element-wise multiplication, multiply original features by attention weights


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        padding = (kernel_size - 1) // 2  # keep size same
        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=padding)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # Pool along channel axis → [B, 1, H, W]
        avg_pool = torch.mean(x, dim=1, keepdim=True)
        max_pool, _ = torch.max(x, dim=1, keepdim=True)

        # Concatenate: [B, 2, H, W]
        spatial_info = torch.cat([avg_pool, max_pool], dim=1)

        # Learn spatial attention map
        attention = self.sigmoid(self.conv(spatial_info))  # [B, 1, H, W]
        return (x * attention).to(x.device)
    

class HybridAttention(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super(HybridAttention, self).__init__()
        self.channel_attn = ChannelAttention(in_channels, reduction)
        self.spatial_attn = SpatialAttention()

    def forward(self, x):
        x = self.channel_attn(x)
        x = self.spatial_attn(x)
        return x.to(x.device)