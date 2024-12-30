# ![neuroband](https://github.com/wcpcp/neuroband/blob/main/asset/neuroband.jpg) NeuroBand: A Neural Dubber of Any Musical Instrument
Offical implementation of the paper: NeuroBand: A Neural Dubber of Any Musical Instrument.

## 🔥**News**
(🔥New) 2024/12/30 NeuroBand training Pipeline is released! See the 'training'.

(🔥New) 2023/11/5 NeuroBand inference Pipeline is released! See the 'inference'.
the NeuroBand include: video feature of CAVP from Diff-Foley, the audioldm, and the NeuroBand.

## 📖 **Abstract**
Humans are born with the ability to associate synchronized music with musical instrument performance videos. However, existing video-to-audio (V2A) models struggle to generate high-quality and temporally synchronized multi-category music. In this work, we address this issue by introducing an audio generation task based on instrument performance videos. To this end, we present InstruVideo, a novel video-text paired benchmark consisting of over 812 hours of clips across 106 different musical instruments and providing a rich and diverse dataset of musical performances. We propose \textit{NeuroBand}, a diffusion model specifically designed to generate high-fidelity audio from instrument performance videos. NeuroBand leverages Variational Autoencoder (VAE) and vocoders pre-trained on large audio datasets to enhance the quality of the generated audio. Experimental results on three instrumental test sets, \emph{e.g.,} YouTube-Music, Douyin-Music, and MUSIC-solo, demonstrate that NeuroBand pre-trained on large audio datasets can highly enhance the quality of the generated audio. Furthermore, employing flow matching as an optimization technique improves the temporal synchronization between the generated audio and the corresponding video. The dataset, code and models will be made publicly available for further research.
![image](https://github.com/wcpcp/neuroband/blob/main/asset/1c9b380e7992afc57c82913bc7aca8f.png)

## **prepare the environmemt**
```Bash
git clone https://github.com/wcpcp/neuroband.git
conda create -n neuroband python=3.10
conda activate neuroband
pip install -r requirements.txt
```

## 🚀 **Inference Usages:**
1.Open the inference.py in inference folder.

2.Download the audioencoder from [audioldm](https://github.com/haoheliu/AudioLDM-training-finetuning/tree/main/data)

3.download the prepared model foler from Hugging Face🤗 here and place it under inference folder.

4.Run the inference.py

## 💻 **training:**

### Dataset
Please download our InstruVideo dataset we used in our paper.
The data structure is like this:
```
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
```


### Training
```Bash
bash launch_audioldm_fm.sh
```

## 🗓️ TODO
* ✅ Release inference code
* ✅ Release training code
*  Release Neuroband v1.0
*  Release project page
*  Release paper
*  Release dataset InstruVideo
*  Release dataset YouTube-Music, Douyin-Music and MUSIC-solo


## 🖼️ Results

<table border="0" style="width: 100%; text-align: left; margin-top: 20px;">
  <tr>
    <td>
      <h2>Drum</h2>
      <video src="https://github.com/user-attachments/assets/62a887a0-d614-48a2-a8a0-3e6723fdbee7" width="100%" controls autoplay loop></video>
    </td>
  </tr>
  <tr>
    <td>
      <h2>Accordion</h2>
      <video src="https://github.com/user-attachments/assets/ebb949b4-0be4-4b79-91c6-654eac520bf1" width="100%" controls autoplay loop></video>
    </td>
  </tr>
  <tr>
    <td>
      <h2>Guitar</h2>
      <video src="https://github.com/user-attachments/assets/bea4f0af-1779-4918-ad0d-329c46559736" width="100%" controls autoplay loop></video>
    </td>
  </tr>
  <tr>
    <td>
      <h2>Piano</h2>
      <video src="https://github.com/user-attachments/assets/0bdad2c4-a6e5-4c03-aea1-61334eadacd5" width="100%" controls autoplay loop></video>
    </td>
  </tr>
  <tr>
    <td>
      <h2>Guzheng</h2>
      <video src="https://github.com/user-attachments/assets/c6d63e95-8e74-43f4-98d0-f9833ea2c40a" width="100%" controls autoplay loop></video>
    </td>
  </tr>
</table>


<!-- ## 📧 Contact

If you have any comments or questions, please open a new issue or feel free to contact [Yibin Wang](https://codegoat24.github.io).

## 🖊️ Citation
🌟 If you find our work helpful, please leave us a star and cite our paper.

```bibtex
@article{LiFT,
  title={LiFT: Leveraging Human Feedback for Text-to-Video Model Alignment.},
  author={Wang, Yibin and Tan, Zhiyu, and Wang, Junyan and Yang, Xiaomeng and Jin, Cheng and Li, Hao},
  journal={arXiv preprint arXiv:2412.04814},
  year={2024}
}
``` -->

## Acknowledgement
Our work is based on [Diff-Foley](https://github.com/luosiallen/Diff-Foley.git) and [AudioLdm](https://github.com/haoheliu/AudioLDM-training-finetuning.git), thanks to all the contributors!