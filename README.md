# Enkidu
Enkidu: Universal Frequential Perturbation for Real-Time Audio Privacy Protection against Voice Deepfakes

**Enkidu** is an open-source implementation of the framework described in  
**"Enkidu: Universal Frequential Perturbation for Real-Time Audio Privacy Protection against Voice Deepfakes" (ACM MM 2025)**.  

The framework generates lightweight and imperceptible **Universal Frequential Perturbations (UFPs)** in the frequency domain, which can be attached to user audio to protect against voice cloning and deepfake attacks in real time.

## ✨ Features
- **Universal Perturbation (UFP)**: One-time optimization, reusable across arbitrary user audio.  
- **Real-Time Protection**: Efficient frequency-domain perturbation, deployable on CPU/GPU in real time.  
- **Flexible Training**: Combines embedding separation loss and perceptual loss.  
- **API + CLI**: Use as a Python module or as a command-line tool.

## ⚙️ Installation

Dependencies:
- Python >= 3.9 (**3.10 recommended**)
- torch == 2.4.0
- torchaudio == 2.4.0
- speechbrain == 1.0.0

## 📂 Project Structure

```csharp
Enkidu/                   
├── core/                         # Core implementation    
│   ├── __init__.py    
│   └── enkidu.py                 # Main Enkidu model     
│       
├── wavdataset/                   # Dataset utilities      
│   ├── __init__.py              
│   └── waveform_privacy_dataset.py                
│                    
├── test_enkidu.py                # Example usage (Python API)                
├── cli_enkidu.py                 # Command-line interface                     
└── README.md                    
```


## 🚀 Usage

1. **Python API**:

You can call Enkidu directly inside Python scripts:

```python
import torch
import torchaudio
from speechbrain.inference import SpeakerRecognition
from wavdataset import WaveformPrivacyDataset
from core import Enkidu

device = 'cuda:0'

# Load dataset (LibriSpeech example)
wave_dataset = WaveformPrivacyDataset(
    dataset_dir='/path/to/LibriSpeech/test-clean',
    sample_rate=16000,
    mono=True,
    wav_format='flac',
)

# Get 40 samples from a single speaker
sample_list = wave_dataset.get_speaker_samples(0)[:40]

# Initialize Enkidu model
enkidu = Enkidu(
    model=SpeakerRecognition.from_hparams('speechbrain/spkrec-ecapa-voxceleb', run_opts={"device": device}),
    steps=10,
    alpha=0.1,
    mask_ratio=0.3,
    frame_length=30,
    noise_level=0.4,
    device=device,
)

# Optimize universal noise
noise_real, noise_imag = enkidu(sample_list)

# Protect a new 10-second voice sample
benign_voice = torch.randn(1, 160000)  # Simulated 10s waveform
torchaudio.save('benign_voice.wav', benign_voice, 16000)

encrypted_voice = enkidu.add_noise(
    benign_voice,
    noise_real,
    noise_imag,
    mask_ratio=0.3,
    random_offset=False,
    noise_smooth=True,
)

torchaudio.save('encrypted_voice.wav', encrypted_voice, 16000)
print("Encrypted audio saved to encrypted_voice.wav")

```

2. **Command-line Interface (CLI)**:

You can also run Enkidu directly from the command line:

```bash
python cli_enkidu.py \
    --audios_dir /path/to/train_audios \
    --wav_format flac \
    --steps 100 \
    --alpha 0.1 \
    --mask_ratio 0.3 \
    --frame_length 30 \
    --noise_level 0.4 \
    --device cuda:0 \
    --input_waveform benign_voice.wav \
    --output_waveform encrypted_voice.wav
```

The CLI takes the following arguments:

Enkidu options (training UFP noise):
  - `--audios_dir` (str, required): Directory of training audios for optimizing universal noise.
  - `--sample_rate` (int, optional): Target sample rate (resampling will be applied if different).
  - `--mono` (bool, optional): Convert input to mono if set.
  - `--wav_format` (str, required): Audio format to search under audios_dir (wav, flac, m4a, ogg, mp3).
  - `--steps` (int, default=40): Number of optimization steps for noise learning.
  - `--alpha` (float, default=0.1): Learning rate for optimizer.
  - `--mask_ratio` (float, default=0.3): Proportion of frames masked during training (for augmentation).
  - `--frame_length` (int, default=30): Frame length in STFT domain for tiling perturbations.
  - `--noise_level` (float, default=0.1): Amplitude scaling factor of noise.
  - `--noise_smooth` (bool, default=False): Apply Wiener filtering to smooth noise in frequency domain.
  - `--device` (str, default="cuda:0"): Device to run the model (cpu, cuda:0, etc.).

Encryption options (applying learned noise):
  - `--input_waveform` (str, required): Path to the input (benign) audio file.
  - `--output_waveform` (str, required): Path to save the encrypted (protected) audio.



## 📖 Citation

If you find this work useful, please cite our paper:

```bibtex
@misc{feng2025enkidu,
      title={Enkidu: Universal Frequential Perturbation for Real-Time Audio Privacy Protection against Voice Deepfakes}, 
      author={Zhou Feng and Jiahao Chen and Chunyi Zhou and Yuwen Pu and Qingming Li and Tianyu Du and Shouling Ji},
      year={2025},
      eprint={2507.12932},
      archivePrefix={arXiv},
      primaryClass={cs.SD},
      url={https://arxiv.org/abs/2507.12932}, 
}
```

