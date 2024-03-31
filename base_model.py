import torch
import torch.nn as nn
import torchvision.models as models

class Backbone(nn.Module):
    """
    Encoder backbone, which is a modified ResNet18
    """
    
    def __init__(self, pretrained=True, backbone=models.resnet18):
        """ Module initializer """
        super().__init__()
        self.backbone = backbone(pretrained=pretrained)
        self.backbone.avgpool = nn.Identity()
        self.backbone.fc = nn.Identity()
    
    def forward(self, x):
        """ Forward pass through all blocks, keeping all intermediate representations """
        x = self.backbone.conv1(x)
        x = self.backbone.bn1(x)
        x0 = x = self.backbone.relu(x)
        x = self.backbone.maxpool(x)
        
        x1 = x = self.backbone.layer1(x)
        x2 = x = self.backbone.layer2(x)
        x3 = x = self.backbone.layer3(x)
        x4 = x = self.backbone.layer4(x)
        return x0, x1, x2, x3, x4
    
class DecoderBlock(nn.Module):
    """
    Simple decoder block, which upsamples via a transposed convolution
    """
    
    def __init__(self, in_channels, out_channels, upsampling=2):
        """ """
        super().__init__()
        self.decoder = nn.Sequential(
                nn.ConvTranspose2d(in_channels, out_channels, kernel_size=4, stride=upsampling, padding=1),
                nn.ReLU(),
                nn.BatchNorm2d(out_channels)
            )
        return
        
    def forward(self, x):
        """ """
        y = self.decoder(x)
        return y
    
class SegModel(nn.Module):
    """
    Simple implementation of an FCN-ish module using a ResNet-18 encoder
    """
    
    def __init__(self, num_classes=23, pretrained=True, backbone=models.resnet18):
        """ """
        super().__init__()
        self.encoder = Backbone(pretrained=True, backbone=backbone)
        
        self.out_conv_4 = DecoderBlock(in_channels=512, out_channels=256, upsampling=2)
        self.out_conv_3 = DecoderBlock(in_channels=256, out_channels=128, upsampling=2)
        self.out_conv_2 = DecoderBlock(in_channels=128, out_channels=64, upsampling=2)
        self.out_conv_1 = DecoderBlock(in_channels=64, out_channels=64, upsampling=2)
        self.out_conv_0 = DecoderBlock(in_channels=64, out_channels=64, upsampling=2)
        self.final_conv = nn.Conv2d(in_channels=64, out_channels=num_classes, kernel_size=5, padding=2)

        self.out_conv_4_depth = DecoderBlock(in_channels=512, out_channels=256, upsampling=2)
        self.out_conv_3_depth = DecoderBlock(in_channels=256, out_channels=128, upsampling=2)
        self.out_conv_2_depth = DecoderBlock(in_channels=128, out_channels=64, upsampling=2)
        self.out_conv_1_depth = DecoderBlock(in_channels=64, out_channels=64, upsampling=2)
        self.out_conv_0_depth = DecoderBlock(in_channels=64, out_channels=64, upsampling=2)
        self.final_conv_depth = nn.Conv2d(in_channels=64, out_channels=1, kernel_size=5, padding=2)
        return
    
        
    def forward(self, x):
        """ Forward pass """
        x0, x1, x2, x3, x4 = self.encoder(x)
        y4 = self.out_conv_4(x4)
        y3 = self.out_conv_3(y4 + x3)
        y2 = self.out_conv_2(y3 + x2)
        y1 = self.out_conv_1(y2 + x1)
        y = self.out_conv_0(y1 + x0)
        y = self.final_conv(y)

        y4_depth = self.out_conv_4_depth(x4)
        y3_depth = self.out_conv_3_depth(y4_depth + x3)
        y2_depth = self.out_conv_2_depth(y3_depth + x2)
        y1_depth = self.out_conv_1_depth(y2_depth + x1)
        y_depth = self.out_conv_0_depth(y1_depth + x0)
        y_depth = self.final_conv_depth(y_depth)
        return y, y_depth
    
    def freeze_encoder(self):
        """ """
        for param in self.encoder.parameters():
            param.requires_grad = False
        
    def unfreeze_encoder(self):
        """ """
        for param in self.encoder.parameters():
            param.requires_grad = True