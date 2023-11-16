# Awesome papers for Diffusion Models for Multi-Modal Generation
[![Awesome](https://awesome.re/badge.svg)](https://awesome.re) 
![PRs Welcome](https://img.shields.io/badge/PRs-Welcome-green)
![Stars](https://img.shields.io/github/stars/JiaojiaoYe1994/Awesome-DIffusionModels4Multi-Modal)
[![Visits Badge](https://badges.pufler.dev/visits/yyysjz1997/Awesome-TimeSeries-SpatioTemporal-Diffusion-Model)](https://badges.pufler.dev/visits/yyysjz1997/Awesome-TimeSeries-SpatioTemporal-Diffusion-Model)


A curated list of Diffusion Models for Mutli-Modal Generation with awesome resources (paper, code, application, review, survey, etc.), which aims to comprehensively and systematically summarize the recent advances to the best of our knowledge.

## Table of contents
* 1. [Text-to-Image Generation](#text-to-image-generation)
* 2. [Scene Graph-to-Image Generation](#scene-graph-to-image-generation)
* 3. [Text-to-3D Generation](#text-to-3d-image-generation)
* 4. [Text-to-Motion Generation](#text-to-motion-generation)
* 5. [Text-to-Video Generation](#text-to-video-generation)
* 6. [Text-to-Audio Generation](#text-to-audio-generation)

## 1. Text-to-Image Generation
* A survey of vision-language pre-trained models. (2022). Yifan Du, Zikang Liu, Junyi Li, and Wayne Xin Zhao. 2022. [paper](https://arxiv.org/abs/2202.10936)
* Imagic: Text-Based Real Image Editing with Diffusion Models. (2022) Bahjat Kawar, Shiran Zada, Oran Lang, Omer Tov, Huiwen Chang, Tali Dekel, Inbar Mosseri, and Michal Irani. 2022. [paper](https://arxiv.org/abs/2210.09276)
* Least squares estimation without priors or supervision. Neural computation 23, 2 (2011), 374–420 Martin Raphan and Eero P Simoncelli. 2011.  [paper](https://dl.acm.org/doi/10.1162/NECO_a_00076)
* UniTune: Text-Driven Image Editing by Fine Tuning an Image Generation Model on a Single Image. SIGGRAPH 2023 (2022) Dani Valevski, Matan Kalman, Yossi Matias, and Yaniv Leviathan. 2022. [paper](https://arxiv.org/abs/2210.09477) (2022). 
* Hierarchical text-conditional image generation with clip latents. NiPs (2022). Aditya Ramesh, Prafulla Dhariwal, Alex Nichol, Casey Chu, and Mark Chen. 2022. [paper](https://proceedings.neurips.cc/paper_files/paper/2022/file/260a14acce2a89dad36adc8eefe7c59e-Paper-Conference.pdf)
* GLIDE: Towards Photorealistic Image Generation and Editing with Text-Guided Diffusion Models. ICML. 16784–16804. Alexander Quinn Nichol, Prafulla Dhariwal, Aditya Ramesh, Pranav Shyam, Pamela Mishkin, Bob Mcgrew, Ilya Sutskever, and Mark Chen. 2022.  [paper](https://arxiv.org/abs/2112.10741)
* Vector quantized diffusion model for text-to-image synthesis. In IEEE Conference on Computer Vision and Pattern Recognition. 10696–10706. Shuyang Gu, Dong Chen, Jianmin Bao, Fang Wen, Bo Zhang, Dongdong Chen, Lu Yuan, and Baining Guo. 2022. [paper](https://openaccess.thecvf.com/content/CVPR2022/papers/Gu_Vector_Quantized_Diffusion_Model_for_Text-to-Image_Synthesis_CVPR_2022_paper.pdf)

## 2. Scene Graph-to-Image Generation
* Roei Herzig, Amir Bar, Huijuan Xu, Gal Chechik, Trevor Darrell, and Amir Globerson. 2020. Learning canonical representations for scene graph to image generation. ECCV. 210–227. [paper](https://arxiv.org/abs/1912.07414)
* Justin Johnson, Agrim Gupta, and Li Fei-Fei. 2018. Image generation from scene graphs. In Proceedings of the IEEE conference on computer vision and pattern recognition. 1219–1228 [paper](https://arxiv.org/abs/1804.01622)[code](https://github.com/google/sg2im)
* Yikang Li, Tao Ma, Yeqi Bai, Nan Duan, Sining Wei, and Xiaogang Wang. 2019. Pastegan: A semi-parametric method to generate image from scene graph. Advances in NeurIPS 32 (2019). [paper](https://arxiv.org/abs/1905.01608) [code](https://github.com/yikang-li/PasteGAN)
* Robin Rombach, Andreas Blattmann, Dominik Lorenz, Patrick Esser, and Björn Ommer. 2022. High-resolution image synthesis with latent diffusion models. In CVPR. 10684–10695 [paper](https://openaccess.thecvf.com/content/CVPR2022/papers/Rombach_High-Resolution_Image_Synthesis_With_Latent_Diffusion_Models_CVPR_2022_paper.pdf) [code](https://github.com/CompVis/latent-diffusion)
* Robin Rombach, Andreas Blattmann, Dominik Lorenz, Patrick Esser, and Björn Ommer. 2022. High-resolution image synthesis with latent diffusion　models. In IEEE Conference on Computer Vision and Pattern Recognition. 10684–10695 [paper](https://openaccess.thecvf.com/content/CVPR2022/papers/Rombach_High-Resolution_Image_Synthesis_With_Latent_Diffusion_Models_CVPR_2022_paper.pdf) [code](https://github.com/CompVis/latent-diffusion)

## 3. Text-to-3D Generation
* Ben Poole, Ajay Jain, Jonathan T Barron, and Ben Mildenhall. 2022. Dreamfusion: Text-to-3d using 2d diffusion. arXiv preprint arXiv:2209.14988(2022) [paper](https://arxiv.org/abs/2209.14988) [review](https://openreview.net/forum?id=FjNys5c7VyY) [demo](https://dreamfusion3d.github.io/)
* Chen-Hsuan Lin, Jun Gao, Luming Tang, Towaki Takikawa, Xiaohui Zeng, Xun Huang, Karsten Kreis, Sanja Fidler, Ming-Yu Liu, and Tsung-Yi Lin. 2022. Magic3D: High-Resolution Text-to-3D Content Creation. in CVPR (2023). [paper](https://openaccess.thecvf.com/content/CVPR2023/papers/Lin_Magic3D_High-Resolution_Text-to-3D_Content_Creation_CVPR_2023_paper.pdf) [blog](https://research.nvidia.com/labs/dir/magic3d/#:~:text=Magic3D%20is%20a%20new%20text,avenues%20to%20various%20creative%20applications.)
* Jiale Xu, Xintao Wang, Weihao Cheng, Yan-Pei Cao, Ying Shan, Xiaohu Qie, and Shenghua Gao. 2022. Dream3D: Zero-Shot Text-to-3D Synthesis Using 3D Shape Prior and Text-to-Image Diffusion Models. CVPR (2023). [paper](https://openaccess.thecvf.com/content/CVPR2023/papers/Xu_Dream3D_Zero-Shot_Text-to-3D_Synthesis_Using_3D_Shape_Prior_and_Text-to-Image_CVPR_2023_paper.pdf) [code](https://bluestyle97.github.io/dream3d/) 

## 4. Text-to-Motion Generation
* Jihoon Kim, Jiseob Kim, and Sungjoon Choi. 2022. Flame: Free-form language-based motion synthesis & editing. in AAAI (2023). [paper](https://dl.acm.org/doi/10.1609/aaai.v37i7.25996) [code](https://github.com/kakaobrain/flame) [demo](https://kakaobrain.github.io/flame/)
* Guy Tevet, Sigal Raab, Brian Gordon, Yonatan Shafir, Daniel Cohen-Or, and Amit H Bermano. 2022. Human motion diffusion model. ICLR (2023). [paper](https://openreview.net/forum?id=SJ1kSyO2jwu) [code](https://github.com/GuyTevet/motion-diffusion-model) [demo](https://guytevet.github.io/mdm-page/)
* Mingyuan Zhang, Zhongang Cai, Liang Pan, Fangzhou Hong, Xinying Guo, Lei Yang, and Ziwei Liu. 2022. Motiondiffuse: Text-driven human motion generation with diffusion model. in arXiv (2022). [paper](https://arxiv.org/pdf/2208.15001.pdf) [code](https://github.com/mingyuan-zhang/MotionDiffuse) [demo](https://mingyuan-zhang.github.io/projects/MotionDiffuse.html)

## 5. Text-to-Video Generation
* Jonathan Ho, William Chan, Chitwan Saharia, Jay Whang, Ruiqi Gao, Alexey Gritsenko, Diederik P Kingma, Ben Poole, Mohammad Norouzi, David J Fleet, et al. 2022. Imagen video: High definition video generation with diffusion models. arXiv preprint arXiv:2210.02303 (2022). [paper](https://imagen.research.google/video/paper.pdf)
* Uriel Singer, Adam Polyak, Thomas Hayes, Xi Yin, Jie An, Songyang Zhang, Qiyuan Hu, Harry Yang, Oron Ashual, Oran Gafni, et al. 2022. Make-a-video: Text-to-video generation without text-video data. ICLR Poster (2022) [paper](https://openreview.net/forum?id=nJfylDvgzlq) [demo](https://makeavideo.studio/)
* Chenyang Qi, Xiaodong Cun, Yong Zhang, Chenyang Lei, Xintao Wang, Ying Shan, and Qifeng Chen. 2023. FateZero: Fusing Attentions for Zero-shot Text-based Video Editing. in ICCV Oral (2023). [paper](https://openaccess.thecvf.com/content/ICCV2023/supplemental/QI_FateZero_Fusing_Attentions_ICCV_2023_supplemental.pdf) [code](https://github.com/ChenyangQiQi/FateZero)
* Jay Zhangjie Wu, Yixiao Ge, Xintao Wang, Weixian Lei, Yuchao Gu, Wynne Hsu, Ying Shan, Xiaohu Qie, and Mike Zheng Shou. 2022. Tune-A-Video: One-Shot Tuning of Image Diffusion Models for Text-to-Video Generation. in ICCV (2023). [paper](https://openaccess.thecvf.com/content/ICCV2023/papers/Wu_Tune-A-Video_One-Shot_Tuning_of_Image_Diffusion_Models_for_Text-to-Video_Generation_ICCV_2023_paper.pdf) [code](https://github.com/showlab/Tune-A-Video)
* Video diffusion models. Ho, Jonathan and Salimans, Tim and Gritsenko, Alexey and Chan, William and Norouzi, Mohammad and Fleet, David J. in arXiv (2022). [demo](https://video-diffusion.github.io/)
 
## 6. Text-to-Audio Generation.
* Vadim Popov, Ivan Vovk, Vladimir Gogoryan, Tasnima Sadekova, and Mikhail Kudinov. 2021. Grad-tts: A diffusion probabilistic model for text-to-speech. In ICML. 8599–8608. [paper](https://arxiv.org/abs/2105.06337) [code](https://github.com/huawei-noah/Speech-Backbones/tree/main/Grad-TTS) [paper](https://grad-tts.github.io/)
* Lawrence R Rabiner. 1989. A tutorial on hidden Markov models and selected applications in speech recognition. Proc. IEEE 77, 2 (1989), 257–286. [paper](https://ieeexplore.ieee.org/document/18626)
* Sungwon Kim, Heeseung Kim, and Sungroh Yoon. 2022. Guided-TTS 2: A Diffusion Model for High-quality Adaptive Text-to-Speech with Untranscribed Data. PMLR (2023). [paper](https://proceedings.mlr.press/v162/kim22d.html) [demo](https://ksw0306.github.io/guided-tts2-demo/) [hf](https://huggingface.co/snu-ai/guided-tts2)
* Dongchao Yang, Jianwei Yu, Helin Wang, Wen Wang, Chao Weng, Yuexian Zou, and Dong Yu. 2022. Diffsound: Discrete Diffusion Model for Text-to-sound Generation. arXiv preprint arXiv:2207.09983 (2022). [paper](https://arxiv.org/abs/2207.09983) [code](https://github.com/yangdongchao/Text-to-sound-Synthesis)
* Jaesung Tae, Hyeongju Kim, and Taesu Kim. 2021. EdiTTS: Score-based Editing for Controllable Text-to-Speech. arXiv preprint arXiv:2110.02584 (2021) [paper](https://arxiv.org/abs/2110.02584) [code](https://github.com/neosapience/editts) [demo](https://editts.github.io/)
* Rongjie Huang, Zhou Zhao, Huadai Liu, Jinglin Liu, Chenye Cui, and Yi Ren. 2022. ProDiff: Progressive Fast Diffusion Model For High-Quality Text-to-Speech. in MM (2022). [paper](https://dl.acm.org/doi/abs/10.1145/3503161.3547855) [code](https://github.com/Rongjiehuang/ProDiff)
