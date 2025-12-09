import torch
import torch.nn as nn
import torch.nn.functional as F

class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.ReLU(inplace=True)
        )
    def forward(self, x):
        return self.conv(x)

class UNet(nn.Module):
    def __init__(self, in_channels, out_channels, base_features=64):
        super().__init__()
        self.enc1 = DoubleConv(in_channels, base_features)
        self.enc2 = DoubleConv(base_features, base_features*2)
        self.enc3 = DoubleConv(base_features*2, base_features*4)
        self.pool = nn.MaxPool2d(2)
        
        self.bottleneck = DoubleConv(base_features*4, base_features*8)
        
        self.up3 = nn.ConvTranspose2d(base_features*8, base_features*4, 2, stride=2)
        self.dec3 = DoubleConv(base_features*8, base_features*4)
        self.up2 = nn.ConvTranspose2d(base_features*4, base_features*2, 2, stride=2)
        self.dec2 = DoubleConv(base_features*4, base_features*2)
        self.up1 = nn.ConvTranspose2d(base_features*2, base_features, 2, stride=2)
        self.dec1 = DoubleConv(base_features*2, base_features)
        
        self.out_conv = nn.Conv2d(base_features, out_channels, 1)
        
    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        b = self.bottleneck(self.pool(e3))
        
        d3 = self.dec3(torch.cat([self.up3(b), e3], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))
        
        out = self.out_conv(d1)
        return out