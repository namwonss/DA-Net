# [DA-Net: Dual Attention Network for Haze Removal in Remote Sensing Image](https://ieeexplore.ieee.org/abstract/document/10679105)
**Namwon Kim, Il-Seok Choi, Seong-Soo Han, and Chang-Sung Jeong**
<br/>
IEEE Access, 2024
<br/>
[Manuscript](https://ieeexplore.ieee.org/abstract/document/10679105)

<br/> This is the **official implementation** of DA-Net: Dual Attention Network for Haze Removal in Remote Sensing Image (accepted by IEEE Access). <br/>


<br/> This code serves as a demo of DA-Net in our manuscript, offering a clear and organized workflow for haze removal in remote sensing images (RSI). <br/>

<table>
  <tr>
    <td><img src="img/clean.png" alt="Clean RSI"">
    <p>Clean RSI</p>
    </td>
    <td><img src="img/haze.png" alt="Haze">
    <p>Haze</p>
    </td>
    <td><img src="img/da-net.png" alt="Haze removal result (DA-Net)">
    <p>Dehazing result (DA-Net)</p>
    </td>
  </tr>
</table>
<br/>

# Abstract
Haze removal in remote sensing images is essential for practical applications in various fields such as weather forecasting, monitoring, mineral exploration and disaster management. The previous deep learning models make use of large convolutional kernel and attention mechanisms for efficient dehazing. However, it has drawbacks such as the loss of image details and low performance. In this paper, we shall present a new dual attention network, called DA-Net, for dehazing remote sensing images which achieves better dehazing performance while reducing model complexity sharply by exploiting a novel dual attention block where two modules, channel-spatial attention and parallel attention are serially connected. We propose a new architecture for parallel attention which achieves better dehazing performance by concatenating three different attention mechanisms in parallel: global channel attention, local channel attention and spatial attention. Moreover, we shall show that the concatenation of channel-spatial attention to parallel attention module enables detecting haze component information more accurately while reducing the model complexity proportional to the number of parameters by combining the channel and spatial information generated respectively from two different channel and spatial branches. Our experimental results show that DA-Net achieves much better performance for both synthetic and real image data sets compared to the other dehazing models in terms of quantitative and qualitative evaluations.
<br/>

# Requirements
[timm](https://anaconda.org/conda-forge/timm) <br/>
[torchinfo](https://anaconda.org/conda-forge/torchinfo) <br/>
[numpy](https://anaconda.org/anaconda/numpy) <br/>
[pillow](https://anaconda.org/anaconda/pillow) <br/>
[pytorch](https://pytorch.org/get-started/locally/) <br/>
[scikit-image](https://anaconda.org/anaconda/scikit-image) <br/>
[tqdm](https://anaconda.org/conda-forge/tqdm) <br/>
[ptflops](https://pypi.org/project/ptflops/) <br/>
[pytorch-msssim](https://pypi.org/project/pytorch-msssim/) <br/><br/>


# Dataset

## Synthetic Image Data

* **Remote Sensing Image Dehazing Dataset (RSID)**  
[https://github.com/chi-kaichen/Trinity-Net](https://github.com/chi-kaichen/Trinity-Net)  

* **Remote Sensing Image Cloud Removing Dataset (RICE)**  
[https://github.com/BUPTLdy/RICE_DATASET](https://github.com/BUPTLdy/RICE_DATASET)  

* **SateHaze1k**  
[https://www.kaggle.com/datasets/mohit3430/haze1k](https://www.kaggle.com/datasets/mohit3430/haze1k)  

## Real Image Data

* **Unmanned Aerial Vehicle Images (UAV)**  
[https://github.com/Lyndo125/Real-outdoor-UAV-remote-sensing-hazy-dataset](https://github.com/Lyndo125/Real-outdoor-UAV-remote-sensing-hazy-dataset)  

<br/><br/>


# Dataset path
* The dataset path should be organized as follows.
<br/><br/>
```
dataset/
├── RSID
│  ├── test
│  |   ├── GT
│  |   │   ├── 1.png
│  |   │   └── 2.png
│  |   │   └── ...
│  |   └── hazy
│  |       ├── 1.png
│  |       └── 2.png
│  |       └── ...
│  └── train
│       ├── GT
│       │   ├── 1.png
│       │   └── 2.png
│       │   └── ...
│       └── hazy
│           ├── 1.png
│           └── 2.png
│           └── ...
└── ...
```
<br/><br/>


# Running
* Install Python and necessary libraries (e.g., PyTorch, NumPy).<br/>

## Training
* After checking `option.py`, you can use the following command to train DA-Net:

```bash
python train.py
```
<br/>

## Demo (Inference)
* Make sure you have the pre-trained model weights (`.pk` file) ready.
* After completing the settings, you can run the demo using the command.
* The pre-trained model weights for testing synthetic image data can be found in the `trained_models` directory.

```bash
python demo.py
```

### Code Ocean
* Please click on Reproducible Run in Code Ocean. <br/>
[https://codeocean.com/capsule/3008254/tree](https://codeocean.com/capsule/3008254/tree)

<br/>

* Code Ocean provides a Python runtime environment to run the DA-Net demo.

<br/><br/>

# Citation

* Please cite our paper in your manuscript if it helps your research.

Bibtex:

```latex
@article{kim2024net,
  title={{DA-Net}: {D}ual-Attention Network for Haze Removal in Remote Sensing Image},
  author={Kim, Namwon and Choi, Il-Seok and Han, Seong-Soo and Jeong, Chang-Sung},
  journal={IEEE Access},
  volume={12},
  pages={136{297}--136{312}},
  year={2024},
  publisher={IEEE}
}
```

