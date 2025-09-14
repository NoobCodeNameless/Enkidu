"""
Test script for Enkidu
"""
import torch
import torchaudio
from speechbrain.inference import SpeakerRecognition

from wavdataset import WaveformPrivacyDataset
from core import Enkidu

# basic config
device = 'cuda:0'

wave_dataset = WaveformPrivacyDataset(
    dataset_dir='/home/fz/codes/voice_minimization/dataset/LibriSpeech/test-clean',
    sample_rate=16000,
    mono=True,
    wav_format='flac',
    transforms=None,
)

# just test with 40 samples
sample_list = wave_dataset.get_speaker_samples(0)[:40]

waveform_encrypted_model = Enkidu(
    model=SpeakerRecognition.from_hparams('speechbrain/spkrec-ecapa-voxceleb', run_opts={"device": device}),
    steps=10,
    alpha=0.1,
    mask_ratio=0.3,
    frame_length=30,
    noise_level=0.4,
    device=device
)

noise_real, noise_imag = waveform_encrypted_model(sample_list)

# the voice you want to encrypt
# assume a 10s voice
benign_voice = torch.randn(1, 160000)
torchaudio.save('Enkidu/benign_voice.wav', benign_voice, 16000)

encrypted_voice = waveform_encrypted_model.add_noise(
    benign_voice,
    noise_real,
    noise_imag,
    mask_ratio=0.3,
    random_offset=False,
    noise_smooth=True
)

print(encrypted_voice)
print('done')