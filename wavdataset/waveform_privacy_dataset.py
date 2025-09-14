import os
from typing import Literal, List, Callable, Tuple

import torchaudio
import torch
from torchaudio.transforms import Resample
from torch.utils.data import Dataset
from torch import Tensor


class WaveformPrivacyDataset(Dataset):
    """
    Waveform Privacy Dataset, the dataset is used for the universal noise protection framework
    the dataset could be used for the training and testing of the universal noise protection framework
    
    Function include:
    - load the dataset
    - resample the audio to the sample rate required for optimization
    - convert the audio to the mono audio
    - transform the audio for need
    - get the audio list of the appointed speaker

    Loaded dataset dir should follows the structure like:
    dataset_dir/
    ├── speaker_1/
    │   ├── 00000.wav(or other format)
    │   ├── 00001.wav
    │   └── ...
    ├── speaker_2/
    │   ├── 00000.wav
    │   ├── 00001.wav
    │   └── ...
    └── ...
    """
    def __init__(
        self,
        dataset_dir: str,
        sample_rate: int = 16000,
        mono: bool = True,
        wav_format: Literal['wav', 'flac', 'mp3', 'm4a', 'ogg', 'wma', 'aac', 'm4b', 'm4r', 'm4a', 'm4b', 'm4r'] = 'wav',
        transforms: List[Callable[[Tensor], Tensor]] | None = None,
        ):

        super().__init__()

        # basic config
        self.dataset_dir = dataset_dir
        self.sample_rate = sample_rate
        self.mono = mono
        self.wav_format = wav_format
        self.transforms = transforms

        # dataset info
        self.speakers = [speaker for speaker in os.listdir(dataset_dir)]
        
        # each sample's path and label(speaker)
        self.samples = []
        self.speaker_samples = {speaker: [] for speaker in self.speakers}
        for speaker in self.speakers:
            # find all the files in the speaker directory, even in the second or third level
            for dirpath, dirnames, filenames in os.walk(os.path.join(self.dataset_dir, speaker)):
                for file in filenames:
                    if file.endswith(self.wav_format):
                        self.samples.append((os.path.join(dirpath, file), speaker))
                        self.speaker_samples[speaker].append(os.path.join(dirpath, file))
            
            # for file in os.listdir(os.path.join(self.dataset_dir, speaker)):
            #     if file.endswith(self.wav_format):
            #         self.samples.append((os.path.join(self.dataset_dir, speaker, file), speaker))
            #         self.speaker_samples[speaker].append(os.path.join(self.dataset_dir, speaker, file))

    def _load_waveform(self, waveform_path: str) -> Tensor:

        waveform, sr = torchaudio.load(waveform_path)

        if self.mono:
            waveform = torch.mean(waveform, dim=0, keepdim=True)

        if sr != self.sample_rate:
            resampler = Resample(orig_freq=sr, new_freq=self.sample_rate)
            waveform = resampler(waveform)

        if self.transforms is not None:
            waveform = self.transforms(waveform)

        # ensure the waveform is [1, N]
        if waveform.dim() == 3:
            waveform = waveform.squeeze(0)
        
        assert waveform.dim() == 2 and waveform.size(0) == 1, f'Unexpected waveform shape: {waveform.shape}'

        return waveform

    def __getitem__(self, index: int) -> Tuple[Tensor, str]:
        waveform_path, speaker = self.samples[index]
        waveform = self._load_waveform(waveform_path)
        return waveform, speaker

    def __len__(self) -> int:
        return len(self.samples)

    def get_speaker_samples(self, index: str | int) -> List[Tensor]:
        """
        Return an unaligned waveform list of the appointed speaker for usage

        Args:
            index: str | int, the speaker name or index

        Returns:
            List[Tensor], the unaligned waveform list of the appointed speaker
        """

        if isinstance(index, int):
            speaker = self.speakers[index]
        else:
            speaker = index

        samples = self.speaker_samples[speaker]
        return [self._load_waveform(sample) for sample in samples]