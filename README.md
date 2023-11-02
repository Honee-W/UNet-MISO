# UNet-MISO
An end-to-end lightweight multichannel speech enhancement network

A minimum implementation of [A Causal U-net based Neural Beamforming Network for Real-Time Multi-Channel Speech Enhancement](https://www.isca-speech.org/archive/pdfs/interspeech_2021/ren21_interspeech.pdf)  -- Interspeech 2021
***

This model acts as a weighted and sum beamformer

### How to use:

    1 add unet_miso.py to your model directory
    2 import UNet as model and ready to go 👻

### Model params:

    - params: 1.1M
    - n_fft: number of sample points for STFT
    - hop_length: hop size for STFT
    - mics: number of channels

### Metrics
    
#### Speech and noise dataset from DNS 2022 challenge, RIR is simulated for eight-microphone circular array 
                    
                      sisnr(dB)    snr(dB)    stoi
    1. noisy.          7.588       11.499     0.772
    2. enhanced       12.839       14.039     0.837


### Samples
 [8-channel noisy wav](data/noisy.wav)
 
 [target single-channel wav](data/clean.wav)

 [enhanced single-channel wav](data/enhanced.wav)