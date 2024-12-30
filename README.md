# **NeuroBand: A Neural Dubber of Any Musical Instrument**
Offical implementation of the paper: NeuroBand: A Neural Dubber of Any Musical Instrument.

## **News**
(🔥New) 2024/12/30 NeuroBand training Pipeline is released! See the 'training'.

(🔥New) 2023/11/5 NeuroBand inference Pipeline is released! See the 'inference'.
the NeuroBand include: video feature of CAVP from Diff-Foley, the audioldm, and the NeuroBand.

## **Abstract**
Humans are born with the ability to associate synchronized music with musical instrument performance videos. However, existing video-to-audio (V2A) models struggle to generate high-quality and temporally synchronized multi-category music. In this work, we address this issue by introducing an audio generation task based on instrument performance videos. To this end, we present InstruVideo, a novel video-text paired benchmark consisting of over 812 hours of clips across 106 different musical instruments and providing a rich and diverse dataset of musical performances. We propose \textit{NeuroBand}, a diffusion model specifically designed to generate high-fidelity audio from instrument performance videos. NeuroBand leverages Variational Autoencoder (VAE) and vocoders pre-trained on large audio datasets to enhance the quality of the generated audio. Experimental results on three instrumental test sets, \emph{e.g.,} YouTube-Music, Douyin-Music, and MUSIC-solo, demonstrate that NeuroBand pre-trained on large audio datasets can highly enhance the quality of the generated audio. Furthermore, employing flow matching as an optimization technique improves the temporal synchronization between the generated audio and the corresponding video. The dataset, code and models will be made publicly available for further research.
![image](https://github.com/wcpcp/neuroband/blob/main/asset/1c9b380e7992afc57c82913bc7aca8f.png)

## **prepare the environmemt**
```Bash
git clone https://github.com/wcpcp/neuroband.git
conda create -n neuroband python=3.10
conda activate neuroband
pip install -r requirements.txt
```

## **Inference Usages:**
1.Open the inference.py in inference folder.

2.Download the audioencoder from [audioldm](https://github.com/haoheliu/AudioLDM-training-finetuning/tree/main/data)

3.download the prepared model foler from Hugging Face🤗 here and place it under inference folder.

4.Run the inference.py

## **training:**

### Dataset
Please download our InstruVideo dataset we used in our paper.
The data structure is like this:

dataset
├── data_dir
│  ├── CAVP_feat
│  │   ├──Test
│  │   ├──Train
│  ├── Test
│  │   ├──audio_npy_spec
│  ├── Train
│  │   ├──audio_npy_spec
├── video_dir
│  ├── Test
│  ├── Train



### Training
```Bash
bash launch_audioldm_fm.sh
```

## TODO
* ✅ Release inference code
* ✅ Release training code
*  Release Neuroband v1.0
*  Release project page
*  Release paper
*  Release dataset InstruVideo
*  Release dataset YouTube-Music, Douyin-Music and MUSIC-solo

## Result
## 🖼️ Results

<table border="0" style="width: 100%; text-align: left; margin-top: 20px;">
  <tr>
    <td>
      <h2>Drum</h2>
      <video src="https://github.com/wcpcp/neuroband/blob/main/asset/4c41c1a120f6732cd4356073a6c4821c.mp4?raw=true" width="100%" controls autoplay loop></video>
    </td>
  </tr>
  <tr>
    <td>
      <h2>Accordion</h2>
      <video src="https://github.com/wcpcp/neuroband/blob/main/asset/4d08b9d53d0b363aa1cd87d4a29d0908.mp4?raw=true" width="100%" controls autoplay loop></video>
    </td>
  </tr>
  <tr>
    <td>
      <h2>Guitar</h2>
      <video src="https://github.com/wcpcp/neuroband/blob/main/asset/5cb8c80b954562b49e266fd9bd7e4bad.mp4?raw=true" width="100%" controls autoplay loop></video>
    </td>
  </tr>
  <tr>
    <td>
      <h2>Piano</h2>
      <video src="https://github.com/wcpcp/neuroband/blob/main/asset/51f039b89e99b0d9facbf6d8f6e74928.mp4?raw=true" width="100%" controls autoplay loop></video>
    </td>
  </tr>
  <tr>
    <td>
      <h2>Guzheng</h2>
      <video src="https://github.com/wcpcp/neuroband/blob/main/asset/47606ad9f57be942cf5ab6a9a38e2b5f.mp4?raw=true" width="100%" controls autoplay loop></video>
    </td>
  </tr>
</table>

<!-- **Drum**
<video width="320" height="240" controls>
  <source src="https://github.com/wcpcp/neuroband/blob/main/asset/4c41c1a120f6732cd4356073a6c4821c.mp4?raw=true" type="video/mp4">
  Your browser does not support the video tag.
</video>

**Accordion**
<video width="320" height="240" controls>
  <source src="https://github.com/wcpcp/neuroband/blob/main/asset/4d08b9d53d0b363aa1cd87d4a29d0908.mp4?raw=true" type="video/mp4">
  Your browser does not support the video tag.
</video>

**Guitar**
<video width="320" height="240" controls>
  <source src="https://github.com/wcpcp/neuroband/blob/main/asset/5cb8c80b954562b49e266fd9bd7e4bad.mp4?raw=true" type="video/mp4">
  Your browser does not support the video tag.
</video>

**Piano**
<video width="320" height="240" controls>
  <source src="https://github.com/wcpcp/neuroband/blob/main/asset/51f039b89e99b0d9facbf6d8f6e74928.mp4?raw=true" type="video/mp4">
  Your browser does not support the video tag.
</video>

**Guzheng**
<video width="320" height="240" controls>
  <source src="https://github.com/wcpcp/neuroband/blob/main/asset/47606ad9f57be942cf5ab6a9a38e2b5f.mp4?raw=true" type="video/mp4">
  Your browser does not support the video tag.
</video> -->


## Acknowledgement
Our work is based on [Diff-Foley](https://github.com/luosiallen/Diff-Foley.git) and [AudioLdm](https://github.com/haoheliu/AudioLDM-training-finetuning.git), thanks to all the contributors!