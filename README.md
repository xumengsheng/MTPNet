MTPNet

🔗 Permanent Resources

​​Paper​​: Frequency-Domain Prompt Guided Hybrid Mamba-Transformer for Enhanced Nighttime Flare Removal

Jiajun Wu, Bo Wang

​​Code Repository​​: https://github.com/xumengsheng/MTPNet
🔍Method

📄 Abstract

During nighttime imaging, lens flare induced by strong light sources severely degrades image quality. Existing methods often struggle to effectively distinguish complex flare patterns from background content, leading to restored images with distortions, optical artifacts, or structural damage to light sources. To address these issues, we propose a Mamba-Transformer hybrid network with frequency-domain prompt guidance (MTPNet) for nighttime flare removal. The network deeply integrates the strengths of Mamba and Transformer architectures and introduces an adaptive frequency-domain prompt block, which transforms frequency-domain processing into a learnable prompting mechanism to dynamically separate and suppress frequency components associated with flare to enhance the model’s perception of degradation patterns in the frequency domain. A feature Correlation module (FCM) is proposed to adaptively fuse encoder features and frequency-domain prompt features through dual spatial and channel attention mechanisms, effectively bridging the gap between semantic information and physical priors. By collaboratively employing Mamba and Transformer modules at different levels, the network achieves effective long-range dependency modeling with a more favorable efficiency profile, maintaining linear computational complexity. Extensive experiments on real-world and synthetic datasets demonstrate that MTPNet superior performance compared to state-of-the-art methods in both objective metrics and subjective visual quality. It not only significantly improves flare-specific evaluation metrics such as G-PSNR and S-PSNR but also delivers visually clearer and more natural results. The source code are available at: https://github.com/xumengsheng/MTPNet.

🐍Environments

The project is built with Python 3.10, Pytorch 1.7.0, CUDA 11.7

📁Data

Flare7K++ consists of Flare7K and Flare-R. Flare7K offers 5,000 scattering flare images and 2,000 reflective flare images, consisting of 25 types of scattering flares and 10 types of reflective flares. Flare-R offers 962 real-captured flare patterns. https://pan.baidu.com/share/init?surl=iNomlQuapPdJqtg3_uX_Fg&pwd=nips

The background images are sampled from [Single Image Reflection Removal with Perceptual Losses, Zhang et al., CVPR 2018]. We filter our most of the flare-corrupted images and overexposed images. https://pan.baidu.com/share/init?surl=BYPRCNSsVmn4VvuU4y4C-Q&pwd=zoyv

If you want to use Flare7K++ for training, please use:

python basicsr/train.py -opt options/uformer_flare7kpp_baseline_option.yml

💻 Code Installation
git clone https://github.com/xumengsheng/MTPNet.git
cd MTPNet
git checkout master

🏋️ Training
Pre-trained Models: Pre-trained model files are stored in the experiments/ folder. Please load the corresponding models according to the training configuration.

📊Evaluation Code

You can run the evaluate.py

📖 Citation

This work is currently under review. Please check back for the final citation details.

@article{Wu2025MTPNet,

title={Frequency-Domain Prompt Guided Hybrid Mamba-Transformer for Enhanced Nighttime Flare Removal},

author={Jiajun Wu and Bo Wang},

journal={Under Review at The Visual Computer},

year={2025},

note={Manuscript submitted for publication}

}

Contact If you have any question, please feel free to reach me out at 12023130662@stu.nxu.edu.cn.
