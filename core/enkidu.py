import os
import random
from typing import Any, Tuple, List, Callable, Dict
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from speechbrain.inference import SpeakerRecognition
from torch.utils.data import Dataset, DataLoader
import torchaudio.compliance.kaldi as Kaldi


class WienerFilter(nn.Module):
    def __init__(self, noise_est_frames: int = 10, eps: float = 1e-8):
        super(WienerFilter, self).__init__()
        self.noise_est_frames = noise_est_frames
        self.eps = eps
    
    def forward(self, noisy_stft: torch.Tensor) -> torch.Tensor:
        """
        Apply Wiener filter to the noisy STFT
        """
        mag = torch.abs(noisy_stft)
        phase = torch.angle(noisy_stft)
        
        noise_mag_est = mag[:, :, :self.noise_est_frames].mean(dim=2, keepdim=True)
        noise_mag_est = noise_mag_est.expand_as(mag)
        
        gain = mag**2 / (mag**2 + noise_mag_est**2 + self.eps)
        
        enhanced_mag = gain * mag
        
        real = enhanced_mag * torch.cos(phase)
        imag = enhanced_mag * torch.sin(phase)
        
        return torch.complex(real, imag)


class Enkidu(nn.Module):

    def __init__(
        self,
        model: SpeakerRecognition,
        # privacy protecting options
        steps: int = 40,
        alpha: float = 0.1,
        decay: float = 0.2,
        mask_ratio: float = 0.3,
        augmentation: bool = True,
        noise_level = 0.1,
        noise_smooth: bool = True,
        frame_length: int = 30,
        # waveform info
        sample_rate: int = 16000,
        # for frequency options
        n_fft: int = 1024,
        hop_length = 512,
        win_length = 1024,
        device: str | torch.device = 'cuda:0',
    ):
        """
        Initialize the Enkidu universal noise protection framework:

        Args:
            model: SpeakerRecognition, the model to extract the embedding
            steps: int, the number of steps to optimize the universal noise
            alpha: float, the learning rate of the optimizer
            mask_ratio: float, the ratio of the mask on the source waveform
            augmentation: bool, whether to use data augmentation
            noise_level: float, the level of the noise
            noise_smooth: bool, whether to use the Wiener filter to smooth the noise on the frequency domain
            frame_length: int, the length of the frame for adding noise
            sample_rate: int, the sample rate of the audio
            n_fft: int, the number of FFT bins
            hop_length: int, the hop length
            win_length: int, the window length
            device: str | torch.device, the device to run the model
        """
        super().__init__()

        self.model = model
        self.steps = steps
        self.alpha = alpha
        self.decay = decay
        self.mask_ratio = mask_ratio
        self.augmentation = augmentation
        self.noise_level = noise_level
        self.noise_smooth = noise_smooth
        self.frame_length = frame_length
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        self.device = device

        self.model.to(self.device)
        self.window = torch.hann_window(self.n_fft).to(self.device)
        self.loss_func = nn.MSELoss()

    def augmentation(self, x: Tensor) -> Tensor:
        """
        Apply data augmentation to the input waveform tensor.
        x shape is [L], L: Length of waveform tensor.
        """
        x = x.reshape(1, -1)

        noise = torch.randn_like(x) * 0.01
        x = x + noise

        # Time shifting
        shift = int(x.size(1) * 0.1)  # shift by 10% of length
        shift = torch.randint(-shift, shift, (1,)).item()
        if shift > 0:
            x = torch.cat((x[:, shift:], torch.zeros(x.size(0), shift, device=x.device)), dim=1)
        elif shift < 0:
            x = torch.cat((torch.zeros(x.size(0), -shift, device=x.device), x[:, :shift]), dim=1)
        
        # Volume scaling
        scale = torch.FloatTensor(x.size(0)).uniform_(0.8, 1.2).to(x.device)
        x = x * scale[:, None]

        return x.flatten()

    def add_noise(
        self,
        source_waveform: Tensor,
        noise_real: Tensor,
        noise_imag: Tensor,
        # addition options
        mask_ratio: float = 0.0, # for augmentation
        random_offset: bool = False, # for augmentation
        noise_smooth: bool = True,
    ):
        """
        Adding frequential noise to the source waveform

        Argument:
            source_waveform: shape [1, N] tensor ready to be tiled noise
            noise_real: real number of the frequential noise
            noise_imag: imag number of the frequential noise

            mask_ratio: float number to control the mask place on source waveform, when only for adding noise, should using default value
            random_offset: bool to control random initialize the offset or not
        
        """
        source_device = source_waveform.device
        source_waveform = source_waveform.flatten().to(self.device)

        noise_real_device = noise_real.device
        noise_imag_device = noise_imag.device
        noise_real = noise_real.to(self.device)
        noise_imag = noise_imag.to(self.device)

        source_stft = torch.stft(
            input=source_waveform,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            return_complex=True,
            window=self.window
        ).unsqueeze(0).to(self.device)

        # get real and imag part of the stft
        stft_real = source_stft.real
        stft_imag = source_stft.imag

        if self.frame_length > source_stft.size(2):
            stft_real[:, :, :source_stft.size(2)] += noise_real[:, :, :source_stft.size(2)] * self.noise_level
            stft_imag[:, :, :source_stft.size(2)] += noise_imag[:, :, :source_stft.size(2)] * self.noise_level
        else:
            
            patch_num = source_stft.shape[-1] // self.frame_length
            offset = random.randint(0, source_stft.shape[-1] % self.frame_length) if random_offset else 0
            
            rand_mask = torch.rand((patch_num,)) >= mask_ratio
            
            if mask_ratio > 0 and rand_mask.sum() == 0:
        
                rand_idx = random.randint(0, patch_num-1)
                rand_mask[rand_idx] = True
            
            for idx in range(patch_num):
                lower = offset + idx * self.frame_length
                upper = offset + (idx + 1) * self.frame_length
                if rand_mask[idx]:
                    stft_real[:, :, lower:upper] += noise_real * self.noise_level
                    stft_imag[:, :, lower:upper] += noise_imag * self.noise_level

        # merge to complex spectrum
        stft_noisy = torch.complex(stft_real, stft_imag)

        if noise_smooth:
            stft_noisy = WienerFilter()(stft_noisy)

        refined_waveform = torch.istft(
            stft_noisy,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=self.window,
            length=source_waveform.size(-1)
        ).reshape(1, -1).to(self.device)

        refined_waveform = refined_waveform.clamp(-1.0, 1.0)
        
        noise_real = noise_real.to(noise_real_device)
        noise_imag = noise_imag.to(noise_imag_device)

        return refined_waveform.to(source_device)

    @staticmethod
    def extract_embedding(x: Tensor, model: SpeakerRecognition, device: str | torch.device = 'cuda:0') -> Tensor:
        """
        Extract the embedding from the input waveform tensor.
        """
        x = x.reshape(1, -1).to(device)
        embedding = model.encode_batch(x).reshape(1, -1).to(device)
        
        return embedding

    @classmethod
    def verify_waveforms(
        cls,
        waveform_1: Tensor,
        waveform_2: Tensor,
        model: SpeakerRecognition,
        device: str | torch.device = 'cuda:0',
    ) -> float:
        """
        Verify the similarity between the source and target waveforms.
        """
        waveform_1 = waveform_1.reshape(1, -1).to(device)
        waveform_2 = waveform_2.reshape(1, -1).to(device)
        with torch.no_grad():
            embedding_1 = cls.extract_embedding(waveform_1, model, device)
            embedding_2 = cls.extract_embedding(waveform_2, model, device)
        
        similarity = F.cosine_similarity(embedding_1, embedding_2).mean().item()

        return similarity
    
    @staticmethod
    def perceptual_loss(
        clean_waveform: Tensor,
        noisy_waveform: Tensor,
        eps: float = 1e-8
    ) -> Tensor:

        diff = noisy_waveform - clean_waveform
        num = (diff ** 2).mean(dim=1)
        den = (clean_waveform ** 2).mean(dim=1) + eps
        return (num / den).mean()


    def total_loss(
        self,
        clean_waveform: Tensor,
        noisy_waveform: Tensor,
        lambda_perceptual: float = 0.1,        
    ) -> Tensor:

        clean_embedding = self.extract_embedding(clean_waveform, self.model, self.device)
        noisy_embedding = self.extract_embedding(noisy_waveform, self.model, self.device)

        main_loss = 1 - self.loss_func(clean_embedding, noisy_embedding)
        perceptual_loss = self.perceptual_loss(clean_waveform, noisy_waveform)

        return main_loss + lambda_perceptual * perceptual_loss

    
    def forward(self, audio_list: List[Tensor]):
        """
        Optimize the universal noise of Enkidu

        Args:
            audio_list: List[Tensor], the list of audio tensors, shape is [1, N], and sample rate is 16000
        """
        universal_noise_real = torch.randn((1, self.n_fft // 2 + 1, self.frame_length), requires_grad=True, device=self.device)
        universal_noise_imag = torch.randn((1, self.n_fft // 2 + 1, self.frame_length), requires_grad=True, device=self.device)

        optimizer = torch.optim.Adam(params=[universal_noise_real, universal_noise_imag], lr=self.alpha, weight_decay=self.decay)

        for epoch in range(self.steps):
            optimizer.zero_grad()

            for idx, audio in enumerate(audio_list):
                # audio = audio.flatten().to(self.device)

                audio = audio.to(self.device)
                noisy_audio = self.add_noise(
                    audio, 
                    universal_noise_real, 
                    universal_noise_imag,
                    mask_ratio=self.mask_ratio,
                    random_offset=False,
                    noise_smooth=self.noise_smooth
                )

                embedding = self.extract_embedding(audio, self.model, self.device)
                noisy_embedding = self.extract_embedding(noisy_audio, self.model, self.device)
                similarity = F.cosine_similarity(embedding, noisy_embedding).mean().item()

                loss = self.total_loss(audio, noisy_audio)
                print(f'Epoch: {epoch}, instance: {idx}:')
                print(f'loss: {loss.item()}')
                print(f'similarity: {similarity}')

                loss.backward()

            optimizer.step()


        return universal_noise_real.detach().cpu(), universal_noise_imag.detach().cpu()