"""CLI version for Enkidu"""
import os
import argparse

import torch
import torchaudio
from torchaudio.transforms import Resample
from speechbrain.inference import SpeakerRecognition

from wavdataset import WaveformPrivacyDataset
from core import Enkidu

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Enkidu options
    parser.add_argument("--audios_dir", type=str, required=True)
    parser.add_argument("--sample_rate", type=int, default=16000, required=False)
    parser.add_argument("--mono", type=bool, required=False)
    parser.add_argument("--wav_format", type=str, required=True, choices=["wav", "flac", "m4a", "ogg", "mp3"])
    parser.add_argument("--steps", type=int, default=40, required=False)
    parser.add_argument("--alpha", type=float, default=0.1, required=False)
    parser.add_argument("--mask_ratio", type=float, default=0.3, required=False)
    parser.add_argument("--frame_length", type=int, default=30, required=False)
    parser.add_argument("--noise_level", type=float, default=0.1, required=False)
    parser.add_argument("--noise_smooth", type=bool, default=False, required=False)
    parser.add_argument("--device", type=str, default="cuda:0", required=False)

    # Encryption options
    parser.add_argument("--input_waveform", type=str, required=True)
    parser.add_argument("--output_waveform", type=str, required=True)
    args = parser.parse_args()

    # find the audios in the audios_dir
    audios = []
    for file in os.listdir(args.audios_dir):
        if file.endswith(args.wav_format):
            # read the audio, and resample and mono it if needed
            audio, original_sr = torchaudio.load(os.path.join(args.audios_dir, file))
            if original_sr != args.sample_rate:
                resampler = Resample(original_sr, args.sample_rate)
                audio = resampler(audio)
            if audio.shape[0] > 1:
                audio = audio.mean(dim=0)
            audios.append(audio)

    waveform_encrypted_model = Enkidu(
        model=SpeakerRecognition.from_hparams('speechbrain/spkrec-ecapa-voxceleb', run_opts={"device": args.device}),
        steps=args.steps,
        alpha=args.alpha,
        mask_ratio=args.mask_ratio,
        frame_length=args.frame_length,
        noise_level=args.noise_level,
        noise_smooth=args.noise_smooth,
    )

    noise_real, noise_imag = waveform_encrypted_model(audios)

    input_waveform, original_sr = torchaudio.load(args.input_waveform)
    if original_sr != args.sample_rate:
        resampler = Resample(original_sr, args.sample_rate)
        input_waveform = resampler(input_waveform)

    encrypted_waveform = waveform_encrypted_model.add_noise(
        input_waveform,
        noise_real,
        noise_imag,
    )

    torchaudio.save(args.output_waveform, encrypted_waveform, args.sample_rate)
    print(f"Encrypted waveform saved to {args.output_waveform}")