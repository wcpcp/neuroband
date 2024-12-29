import importlib
import argparse, os, sys, datetime, glob
import torch
import pytorch_lightning as pl
from pytorch_lightning import seed_everything
from pytorch_lightning.trainer import Trainer
from omegaconf import OmegaConf
from torch.utils.data import DataLoader, Dataset
from pytorch_lightning.callbacks import ModelCheckpoint, Callback, LearningRateMonitor
import time
from pytorch_lightning.utilities import rank_zero_info
import numpy as np
import soundfile as sf  
import yaml

def disabled_train(self, mode=True):
    """Overwrite model.train with this function to make sure train/eval mode
    does not change anymore."""
    return self

def get_obj_from_str(string, reload=False):
    module, cls = string.rsplit('.', 1)
    if reload:
        module_imp = importlib.import_module(module)
        importlib.reload(module_imp)
    return getattr(importlib.import_module(module, package=None), cls)


def instantiate_from_config(config):
    if not 'target' in config:
        raise KeyError('Expected key `target` to instantiate.')
    print(config['target'])
    return get_obj_from_str(config['target'])(**config.get('params', dict()))

def instantiate_from_config_input_dict(config):
    if not 'target' in config:
        raise KeyError('Expected key `target` to instantiate.')
    print(config['target'])
    return get_obj_from_str(config['target'])(config.get('params', dict()))


def nondefault_trainer_args(opt):
    parser = argparse.ArgumentParser()
    parser = Trainer.add_argparse_args(parser)
    args = parser.parse_args([])
    return sorted(k for k in vars(args) if getattr(opt, k) != getattr(args, k))

def save_waveform_from_batch(saveroot, waveform, sr, video_names,seed,i):
        todo_waveform = waveform.squeeze(1)
        for index, wf in enumerate(todo_waveform):
            wf = (
                wf / np.max(np.abs(wf))
            ) * 0.8  # Normalize the energy of the generation output
            savepath = os.path.join(saveroot, f"{video_names[index]}_index_{i}_seed_{seed}.wav")
            sf.write(savepath, wf, samplerate=sr)

def save_waveform(savepath, waveform, sr):
        todo_waveform = waveform[0,0]
        todo_waveform = (
            todo_waveform / np.max(np.abs(todo_waveform))
        ) * 0.8  # Normalize the energy of the generation output
        sf.write(savepath, todo_waveform, samplerate=sr)

def merge_video_audio(video_path, audio_path, output_path):
    command = [
    "ffmpeg",
    "-i", video_path,          # 输入视频文件
    "-i", audio_path,          # 输入音频文件
    "-c:v", "copy",            # 复制视频流，不重新编码
    "-c:a", "aac",             # 编码音频流为aac
    "-map", "0:v:0",             # 映射视频流
    "-map", "1:a:0",             # 映射音频流，使用第二个输入文件的音频
    "-strict", "experimental", # 使用实验性AAC编码
    "-shortest",               # 如果音频或视频长度不同，取较短的
    output_path
    ]
    subprocess.run(command)

class EvalDataset(torch.utils.data.Dataset):
    def __init__(self,feat_folder,duration=9.92,fps=4):
        super().__init__()

        self.duration = duration
        self.fps = fps
        self.frames = int(duration*fps)
        self.feats = os.listdir(feat_folder)
        self.feats = list(map(lambda x:os.path.join(feat_folder,x), self.feats))

    def __len__(self):
        return len(self.feats)

    def __getitem__(self, idx):
        feat_path = self.feats[idx]
        video_name = feat_path.split('/')[-1][:-len('.npz')]
        example_feat = np.load(feat_path)['arr_0'].astype(np.float32)
        example_feat = example_feat[:self.frames,:]
        return video_name, example_feat

def inverse_op(spec):
    sr = 16000
    n_fft = 1024
    fmin = 0
    fmax = 8000
    nmels = 64
    hoplen = 160
    spec = spec.detach().cpu().numpy()
    spec_out = librosa.feature.inverse.mel_to_stft(spec,sr = sr, n_fft = n_fft, fmin=fmin, fmax=fmax, power=1)
    wav = librosa.griffinlim(spec_out, hop_length=hoplen)
    # wav = wav[None, :]
    import pdb;pdb.set_trace()
    return wav

def draw_mel_spec(spec):
    spec = spec.detach().cpu().numpy()
    mel_spec_db = librosa.power_to_db(spec,ref=np.max)
    plt.figure().set_figwidth(12)
    librosa.display.specshow(mel_spec_db, sr=16000, hop_length=160, x_axis='time', y_axis='mel',fmax=8000,fmin=0)
    plt.colorbar()
    plt.title('Mel Spectrogram')
    plt.savefig('/workspace/data_dir/tmp_dyh/Diff-Foley/training/neuroband/eval_result/mel_spectrogram.png', format='png', dpi=300)
    
if __name__ == "__main__":
    import yaml
    import ffmpeg
    import subprocess
    from tqdm import tqdm
    import librosa
    import matplotlib.pyplot as plt
    
    device = torch.device('cuda:6' if torch.cuda.is_available() else 'cpu')
    config = '/workspace/data_dir/tmp_dyh/Diff-Foley/training/neuroband/configs/evaluations/audioldm_large.yaml'
    config = yaml.load(open(config, "r"), Loader=yaml.FullLoader)
    # seed
    seed = 42
    seed_everything(seed)
    # Model:
    model = instantiate_from_config(config['model'])
    model = model.to(device)
    model.eval()
    # benchmark eval
    feat_folder = '/workspace/data_dir/tmp_dyh/eval_dataset/yt8m_valid_music_part_feats'
    steps = 25
    cfg_scale = 1
    repeat_times = 1
    pred_folder = f'/workspace/data_dir/tmp_dyh/eval_dataset/our_cfm_pred_cfg_{cfg_scale}_step_{steps}_repeat_{repeat_times}'
    os.makedirs(pred_folder,exist_ok=True)
    batch_size = 240
    eval_dataset = EvalDataset(feat_folder=feat_folder)
    eval_dataloader = DataLoader(eval_dataset, batch_size=batch_size, num_workers=8,shuffle=True)
    for i in range(repeat_times):
        for video_names, example_feats in tqdm(eval_dataloader):
            example_feats = example_feats.to(device)
            cond = model.get_learned_conditioning(example_feats)
            un_cond = torch.zeros(cond.shape).to(device)
            
            bs = example_feats.shape[0]
            # audio_samples, _ = model.sample_log_diff_sampler(cond, batch_size=bs, sampler_name='DPM_Solver', ddim_steps=steps, unconditional_guidance_scale=cfg_scale,unconditional_conditioning=un_cond)
            audio_samples = model.sample_cfm(cond,batch_size=bs, cfm_steps=steps, size_len=248, unconditional_guidance_scale=cfg_scale,unconditional_conditioning=un_cond)
            audio_samples = model.decode_first_stage(audio_samples)
            audio_samples = model.inverse_op(audio_samples)
            save_waveform_from_batch(saveroot=pred_folder,waveform=audio_samples,sr=16000,video_names=video_names,seed=seed,i=i)
        # seed += 100
        # seed_everything(seed)


    # merge_video_audio('/workspace/data_dir/tmp_dyh/Diff-Foley/training/neuroband/visual_tool/comic_video_cut/comic1_clip.mp4','/workspace/data_dir/tmp_dyh/Diff-Foley/training/neuroband/visual_tool/comic_result/test.wav',f'/workspace/data_dir/tmp_dyh/Diff-Foley/training/neuroband/visual_tool/comic_result/output_{seed}_step_{steps}_cfg_{cfg_scale}_test.mp4')

    # save_waveform('/workspace/data_dir/tmp_dyh/Diff-Foley/training/neuroband/eval_result/test.wav',audio_samples,16000)
    # merge_video_audio('/workspace/data_dir/tmp_dyh/Diff-Foley/training/neuroband/eval_result/origin.mp4','/workspace/data_dir/tmp_dyh/Diff-Foley/training/neuroband/eval_result/test.wav',f'/workspace/data_dir/tmp_dyh/Diff-Foley/training/neuroband/eval_result/output_{seed}_step_{steps}_cfg_{cfg_scale}_test.mp4')



    







        



        






