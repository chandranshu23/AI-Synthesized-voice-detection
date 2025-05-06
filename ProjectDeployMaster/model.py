import torch
import torch.nn as nn
from fusion import AsymBiChaFuse  # our fusion block

#for API
from collections import OrderedDict

# Bottleneck block with expansion factor 4
class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_channels, out_channels, stride=1):
        super(Bottleneck, self).__init__()
        # 1x1 conv to reduce channels
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        # 3x3 conv for spatial processing
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        # 1x1 conv to expand channels
        self.conv3 = nn.Conv2d(out_channels, out_channels * self.expansion, kernel_size=1,
                               bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels * self.expansion)
        self.relu = nn.ReLU(inplace=True)

        # Downsample for shortcut if needed
        self.downsample = None
        if stride != 1 or in_channels != out_channels * self.expansion:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels * self.expansion, kernel_size=1,
                          stride=stride, bias=False),
                nn.BatchNorm2d(out_channels * self.expansion)
            )

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)
        return out

# Modified ResNet101v2 with fusion between every stage
class ResNet101v2(nn.Module):
    def __init__(self, num_classes=2):
        super(ResNet101v2, self).__init__()
        # Input convolution for one-channel images
        self.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1   = nn.BatchNorm2d(64)
        self.relu  = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        # Stage 1: layer1 -> output channels: 64*4 = 256
        self.layer1 = self._make_layer(64, 64, blocks=3, stride=1)
        # Stage 2: layer2 -> output channels: 128*4 = 512
        self.layer2 = self._make_layer(256, 128, blocks=4, stride=2)
        # Stage 3: layer3 -> output channels: 256*4 = 1024
        self.layer3 = self._make_layer(512, 256, blocks=23, stride=2)
        # Stage 4: layer4 -> output channels: 512*4 = 2048
        self.layer4 = self._make_layer(1024, 512, blocks=3, stride=2)

        # Downsampling modules for fusion:
        # Fuse between layer1 (256 channels, resolution ~128x128) and layer2 (512, ~64x64)
        self.fuse_down1 = nn.Sequential(
            nn.AvgPool2d(kernel_size=2, stride=2),
            nn.Conv2d(256, 512, kernel_size=1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True)
        )
        # Fuse between layer2 (512, ~64x64) and layer3 (1024, ~32x32)
        self.fuse_down2 = nn.Sequential(
            nn.AvgPool2d(kernel_size=2, stride=2),
            nn.Conv2d(512, 1024, kernel_size=1, bias=False),
            nn.BatchNorm2d(1024),
            nn.ReLU(inplace=True)
        )
        # Fuse between layer3 (1024, ~32x32) and layer4 (2048, ~16x16)
        self.fuse_down3 = nn.Sequential(
            nn.AvgPool2d(kernel_size=2, stride=2),
            nn.Conv2d(1024, 2048, kernel_size=1, bias=False),
            nn.BatchNorm2d(2048),
            nn.ReLU(inplace=True)
        )

        # Fusion blocks (AsymBiChaFuse) for each fusion stage
        self.fusion_block1 = AsymBiChaFuse(channels=512, r=4)   # for fusing stage1 & stage2
        self.fusion_block2 = AsymBiChaFuse(channels=1024, r=4)  # for fusing stage2 & stage3
        self.fusion_block3 = AsymBiChaFuse(channels=2048, r=4)  # for fusing stage3 & stage4

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(2048, num_classes)

    def _make_layer(self, in_channels, out_channels, blocks, stride):
        layers = []
        layers.append(Bottleneck(in_channels, out_channels, stride))
        for _ in range(1, blocks):
            layers.append(Bottleneck(out_channels * Bottleneck.expansion, out_channels))
        return nn.Sequential(*layers)

    def forward(self, x):
        # Input: (B,1,512,512)
        x = self.conv1(x)    # -> (B,64,256,256)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)  # -> (B,64,128,128) approximately

        # Stage 1
        x1 = self.layer1(x)  # x1: (B,256,128,128)

        # Stage 2
        x2 = self.layer2(x1) # x2: (B,512,64,64)
        # Fuse stage1 and stage2:
        x1_down = self.fuse_down1(x1)  # x1_down: (B,512,64,64)
        fused12 = self.fusion_block1(x2, x1_down)  # fused12: (B,512,64,64)

        # Stage 3
        x3 = self.layer3(fused12)  # x3: (B,1024,32,32)
        # Fuse stage2 and stage3:
        # Note: use original x2 if desired or fused12. Here we use x2:
        x2_down = self.fuse_down2(x2)  # x2_down: (B,1024,32,32)
        fused23 = self.fusion_block2(x3, x2_down)  # fused23: (B,1024,32,32)

        # Stage 4
        x4 = self.layer4(fused23)  # x4: (B,2048,16,16)
        # Fuse stage3 and stage4:
        x3_down = self.fuse_down3(x3)  # x3_down: (B,2048,16,16)
        fused34 = self.fusion_block3(x4, x3_down)  # fused34: (B,2048,16,16)

        x = self.avgpool(fused34)  # -> (B,2048,1,1)
        x = x.view(x.size(0), -1)
        x = self.fc(x)  # -> (B, num_classes)
        return x


#for API
def load_model(path: str, device=None):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = ResNet101v2(num_classes=1)
    checkpoint = torch.load(path, map_location=device)

    # Strip 'model.' if it exists in keys
    state_dict = checkpoint.get("state_dict", checkpoint)
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        new_key = k.replace("model.", "") if k.startswith("model.") else k
        new_state_dict[new_key] = v

    model.load_state_dict(new_state_dict)
    model.to(device)
    model.eval()

    return model


# Example usage:
if __name__ == "__main__":
    model = ResNet101v2(num_classes=2)
    dummy_input = torch.randn(1, 1, 512, 512)  # one-channel input
    output = model(dummy_input)
    print("Output shape:", output.shape)  # Expected: (1, 2)
