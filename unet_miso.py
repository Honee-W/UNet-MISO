# zqwang@2023

import torch
from torch import nn
import torch.nn.functional as F

### implementations for "A Causal U-net based Neural Beamforming Network for Real-Time Multi-Channel Speech Enhancement"

class ConvBlock(nn.ModuleList):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernels,
                 strides,
                 norm_before=False
                ):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernels, strides)
        self.norm = nn.BatchNorm2d(out_channels)
        self.activation = nn.PReLU()
        self.norm_before = norm_before
    
    def forward(self, x):
        x = self.conv(x)
        if self.norm_before:
            x = self.norm(x)
            x = self.activation(x)
        else:
            x = self.activation(x)
            x = self.norm(x)
        return x        


class ConvTransposeBlock(nn.ModuleList):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernels,
                 strides,
                 norm_before=False
                ):
        super().__init__()
        self.conv = nn.ConvTranspose2d(in_channels, out_channels, kernels, strides)
        self.norm = nn.BatchNorm2d(out_channels)
        self.activation = nn.PReLU()
        self.norm_before = norm_before
    
    def forward(self, x):
        x = self.conv(x)
        if self.norm_before:
            x = self.norm(x)
            x = self.activation(x)
        else:
            x = self.activation(x)
            x = self.norm(x)
        return x        


class Encoder(nn.Module):
    def __init__(self,
                 layers: int = 8,
                 mics: int = 8,
                 channels:list = [32, 32, 64, 64, 96, 96, 128, 256],
                 kernels: list = [(6,2), (6,2), (7,2), (6,2), (6,2), (6,2), (2,2), (2,2)],
                 strides: list = [(2,1), (2,1), (2,1), (2,1), (2,1), (2,1), (2,1), (1,1)]) -> None:
        super().__init__()
        self.encoder = nn.ModuleList()
        for i in range(layers):
            if i == 0:
                self.encoder.append(ConvBlock(mics, channels[i], kernels[i], strides[i]))
            else:
                self.encoder.append(ConvBlock(channels[i-1], channels[i], kernels[i], strides[i]))
    
    def forward(self, x: torch.tensor) -> torch.tensor:
        '''args:
            x: multi-channels specs in shape of [batch, channel, frequenchy, frame]
        '''
        layers_out = []
        for layer in self.encoder:
            x = layer(x)
            layers_out.append(x)
        
        return x, layers_out


class Decoder(nn.Module):
    def __init__(self,
                 layers: int = 8,
                 mics: int = 8,
                 channels:list = [32, 32, 64, 64, 96, 96, 128, 256],
                 kernels: list = [(6,2), (6,2), (7,2), (6,2), (6,2), (6,2), (2,2), (2,2)],
                 strides: list = [(2,1), (2,1), (2,1), (2,1), (2,1), (2,1), (2,1), (1,1)]) -> None:
        super().__init__()
        self.decoder = nn.ModuleList()
        self.layers = layers
        for i in range(layers):
            if i == layers - 1:
                self.decoder.append(ConvTransposeBlock(channels[layers-i-1]*2, mics, kernels[layers-i-1], strides[layers-i-1]))
            elif i == 0:
                self.decoder.append(ConvTransposeBlock(channels[layers-i-1], channels[layers-i-2], kernels[layers-i-1], strides[layers-i-1]))
            else:
                self.decoder.append(ConvTransposeBlock(channels[layers-i-1]*2, channels[layers-i-2], kernels[layers-i-1], strides[layers-i-1]))            
    
    def forward(self, x: torch.tensor, y: list) -> torch.tensor:

        for i, layer in enumerate(self.decoder):
            if i == 0:
                x = layer(x)
            else:
                residual = y[self.layers-i-1]
                if x.shape != residual.shape:
                    F_1, F_2 = x.shape[-2], residual.shape[-2]
                    if F_1 > F_2:
                        p = F_1 - F_2
                        residual = F.pad(residual, (0, 0, 0, p))
                    else:
                        p = F_2 - F_1
                        x = F.pad(x, (0, 0, 0, p))                    
                x = torch.cat([x, residual], dim=1)
                x = layer(x)
        return x


# weighted and sum beamformer
class UNet(nn.Module):
    def __init__(self,
                 n_fft: int = 512,
                 hop_length: int = 256,
                 mics: int = 8) -> None:
        super().__init__()
        self.mics = mics
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.encoder = Encoder(mics=mics)
        self.decoder = Decoder(mics=mics)

    def forward(self, x: torch.tensor) -> torch.tensor:
        """Args:
            wav in shape [batch, channel, length]
           Return:
           wav in shape [batch, 1, length] 
        """
        wavs = torch.chunk(x, dim=1, chunks=self.mics)
        specs = []
        for wav in wavs:
            # [batch, frequenchy, frame, 2]
            spec = torch.stft(wav.squeeze(1), self.n_fft, self.hop_length, window=torch.hann_window(self.n_fft).to(x.device), return_complex=True)
            spec = torch.view_as_real(spec)
            r, i = torch.chunk(spec, dim=-1, chunks=2)
            # [batch, frequenchy, frame] -> [batch, frequenchy*2, frame]
            spec = torch.concat([r.squeeze(-1), i.squeeze(-1)], dim=1)
            specs.append(spec)
        x = torch.stack(specs, dim=1)
        y = x
        y, layers_out = self.encoder(y)
        mask = self.decoder(y, layers_out)
        x = x * mask
        # [batch, channel, frequenchy*2, frame] -> [batch, frequenchy*2, frame]
        x = torch.sum(x, dim=1)
        # [batch, frequenchy*2, frame] -> [batch, frequenchy, frame, 2]
        x = torch.stack(torch.chunk(x, dim=1, chunks=2), dim=-1)
        x = torch.view_as_complex(x)
        pred_wav = torch.istft(x, self.n_fft, self.hop_length, window=torch.hann_window(self.n_fft).to(x.device)).unsqueeze(1)
        
        est = {
            "wav": pred_wav
        }
        
        return est


def foo_unet():
    wavs = torch.rand(4, 8, 16000*4)
    net = UNet()
    total_params = sum(param.numel() for param in net.parameters())
    print("total params: {}M".format(total_params/10e6))
    out = net(wavs)
    print("out", out.shape)
    
if __name__ == "__main__":
    foo_unet()