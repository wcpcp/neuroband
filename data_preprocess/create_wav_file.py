import os
from moviepy.editor import VideoFileClip
from concurrent.futures import ThreadPoolExecutor

import argparse
import os
import os.path as P
from copy import deepcopy
from functools import partial
from glob import glob
from multiprocessing import Pool
from pathlib import Path

import librosa
import numpy as np
import torchvision
from tqdm import tqdm
import torch

class MelSpectrogram(object):
    def __init__(self, sr, nfft, fmin, fmax, nmels, hoplen, spec_power, inverse=False):
        self.sr = sr
        self.nfft = nfft
        self.fmin = fmin
        self.fmax = fmax
        self.nmels = nmels
        self.hoplen = hoplen
        self.spec_power = spec_power
        self.inverse = inverse

        self.mel_basis = librosa.filters.mel(sr=sr, n_fft=nfft, fmin=fmin, fmax=fmax, n_mels=nmels)

    def __call__(self, x):
        if self.inverse:
            spec = librosa.feature.inverse.mel_to_stft(
                x, sr=self.sr, n_fft=self.nfft, fmin=self.fmin, fmax=self.fmax, power=self.spec_power
            )
            wav = librosa.griffinlim(spec, hop_length=self.hoplen)
            return wav
        else:
            spec = np.abs(librosa.stft(x, n_fft=self.nfft, hop_length=self.hoplen)) ** self.spec_power
            mel_spec = np.dot(self.mel_basis, spec)
            return mel_spec

class LowerThresh(object):
    def __init__(self, min_val, inverse=False):
        self.min_val = min_val
        self.inverse = inverse

    def __call__(self, x):
        if self.inverse:
            return x
        else:
            return np.maximum(self.min_val, x)

class Add(object):
    def __init__(self, val, inverse=False):
        self.inverse = inverse
        self.val = val

    def __call__(self, x):
        if self.inverse:
            return x - self.val
        else:
            return x + self.val

class Subtract(Add):
    def __init__(self, val, inverse=False):
        self.inverse = inverse
        self.val = val

    def __call__(self, x):
        if self.inverse:
            return x + self.val
        else:
            return x - self.val

class Multiply(object):
    def __init__(self, val, inverse=False):
        self.val = val
        self.inverse = inverse

    def __call__(self, x):
        if self.inverse:
            return x / self.val
        else:
            return x * self.val

class Divide(Multiply):
    def __init__(self, val, inverse=False):
        self.inverse = inverse
        self.val = val

    def __call__(self, x):
        if self.inverse:
            return x * self.val
        else:
            return x / self.val


class Log10(object):
    def __init__(self, inverse=False):
        self.inverse = inverse

    def __call__(self, x):
        if self.inverse:
            return 10 ** x
        else:
            return np.log10(x)

class Clip(object):
    def __init__(self, min_val, max_val, inverse=False):
        self.min_val = min_val
        self.max_val = max_val
        self.inverse = inverse

    def __call__(self, x):
        if self.inverse:
            return x
        else:
            return np.clip(x, self.min_val, self.max_val)

class TrimSpec(object):
    def __init__(self, max_len, inverse=False):
        self.max_len = max_len
        self.inverse = inverse

    def __call__(self, x):
        if self.inverse:
            return x
        else:
            return x[:, :self.max_len]

class MaxNorm(object):
    def __init__(self, inverse=False):
        self.inverse = inverse
        self.eps = 1e-10

    def __call__(self, x):
        if self.inverse:
            return x
        else:
            return x / (x.max() + self.eps)

def normalize_wav(waveform):
    waveform = waveform - np.mean(waveform)
    waveform = waveform / (np.max(np.abs(waveform)) + 1e-8)
    return waveform * 0.5  # Manually limit the maximum amplitude into 0.5
sr = 16000
# trim_len = 620

TRANSFORMS = torchvision.transforms.Compose([
    # MelSpectrogram(sr=sr, nfft=1024, fmin=125, fmax=7600, nmels=128, hoplen=256, spec_power=1),
    MelSpectrogram(sr=sr, nfft=1024, fmin=0, fmax=8000, nmels=64, hoplen=160, spec_power=1),
    # LowerThresh(1e-5),
    # Log10(),
    # Multiply(20),
    # Subtract(20),
    # Add(100),
    # Divide(100),
    # Clip(0, 1.0),
    # TrimSpec(trim_len)
])

def dynamic_range_compression_torch(x, C=1, clip_val=1e-5):
    return np.log(np.clip(x, a_min=clip_val,a_max=None) * C)

def spectral_normalize_torch(magnitudes):
    output = dynamic_range_compression_torch(magnitudes)
    return output

def inv_transforms(x, folder_name='melspec_10s_22050hz'):
    '''relies on the GLOBAL contant TRANSFORMS which should be defined in this document'''
    if folder_name == 'melspec_10s_22050hz':
        i_transforms = deepcopy(TRANSFORMS.transforms[::-1])
    else:
        raise NotImplementedError
    for t in i_transforms:
        t.inverse = True
    i_transforms = torchvision.transforms.Compose(i_transforms)
    return i_transforms(x)


def get_spectrogram(audio_path, length, sr=16000):
    # wav, _ = librosa.load(audio_path, sr=None)
    wav, sr_new = librosa.load(audio_path, sr=sr)
    # wav = np.load(audio_path)
    
    # print(sr)
    # this cannot be a transform without creating a huge overhead with inserting audio_name in each
    y = np.zeros(length)
    if wav.shape[0] < length:
        y[:len(wav)] = wav
    else:
        y = wav[:length]
    
    # # wav:
    y = y[ : length - 1]        # ensure: 640 spec
    y = y.reshape(-1)
    y = normalize_wav(y)
    
    mel_spec = TRANSFORMS(y)
    mel_spec = spectral_normalize_torch(mel_spec)

    return y, mel_spec


def wav_to_spec(wav_file, spec_file):
    sr = 16000
    time = 10
    length = sr * time 
    _, mel_spec =get_spectrogram(wav_file, length, sr)
    # print(mel_spec)
    # print("Mel Spec Shape: {}".format(mel_spec.shape))
    # print("Finished!")
    np.save(spec_file, mel_spec)

# def extract_audio(video_path, audio_path):
#     with VideoFileClip(video_path) as video:
#         audio = video.audio
#         audio.write_audiofile(audio_path)
def extract_audio(video_path, audio_path, spec_path):
    if not os.path.exists(audio_path):
        with VideoFileClip(video_path) as video:
            audio = video.audio
            audio.write_audiofile(audio_path)
    wav_to_spec(audio_path, spec_path)

def extract_audio_from_videos(video_dir, wave_dir, spec_dir, max_workers=10):
    if not os.path.exists(wave_dir):
        os.makedirs(wave_dir)
    
    video_files = [f for f in os.listdir(video_dir) if f.endswith('.mp4')]
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = []
        for filename in video_files:
            video_path = os.path.join(video_dir, filename)
            audio_path = os.path.join(wave_dir, os.path.splitext(filename)[0] + '.wav')
            spec_path = os.path.join(spec_dir, os.path.splitext(filename)[0] + '_mel.npy')
            futures.append(executor.submit(extract_audio, video_path, audio_path, spec_path))
        
        for future in futures:
            future.result()  # 等待所有任务完成

# video_dir = '/workspace/data_dir/tmp_dyh/vgg_sound/video' # '/workspace/data_dir/tmp_dyh/video_example' #
# wave_dir = '/workspace/data_dir/tmp_dyh/vgg_sound/only_wav'
# spec_dir = '/workspace/data_dir/tmp_dyh/vgg_sound/mel_spec'

# extract_audio_from_videos(video_dir, wave_dir,spec_dir, max_workers=10)
# print(f"wav converted, all finished!!")
wav_to_spec('/workspace/wcp/Diff-Foley/CAVP_eval/sample_0_diff.wav',
            '/workspace/wcp/Diff-Foley/CAVP_eval/sample_0_diff.npy')