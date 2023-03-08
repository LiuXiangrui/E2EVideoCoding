# E2EVideoCoding

## Training Specification

### Dependencies
```
torch~=1.13.1+cu117
compressai~=1.2.4.dev0
torchvision~=0.14.1+cu117
tqdm~=4.64.1
Pillow~=9.4.0
prettytable~=3.6.0
tensorboard~=2.12.0
thop~=0.1.1
```

### General Specifications
- Vimeo-90K is used as the training dataset, and the random crop is adopted to get patches with the size of 256x256.
- The batch is set to 4.
- Cheng2020Anchor is used as the codec for I frame.
- RD loss is adopted for optimization, i.e.,
$$ L =D+R, $$
where D is the reconstruction error of input frames and R is the bitrate of both residues and motion information.
- Lambda in RD loss is chosen from {256, 512, 1024, 2048}, and the quality index of intra codec is set as {3, 4, 5, 6} correspondingly.
- Pretrained weights are adopted for motion estimation based on SpyNet, if applicable (i.e., for DVC and DCVC).

### DVC
|      Stage      |                      Loss                       | Learning Rate | Epochs  |
|:---------------:|:-----------------------------------------------:|:-------------:|:-------:|
| WITH_INTER_LOSS | RD loss with $D=0.1(D_{Warp}+D_{Pred})+D_{Rec}$ |     1e-4      |   123   |
|  ONLY_RD_LOSS   |                     RD loss                     |  1e-4, 1e-5   | 77, 123 |
|     ROLLING     |                     RD loss                     |     1e-5      |   50    |
 
### DCVC
|       Stage       |              Loss              | Learning Rate | Epochs |
|:-----------------:|:------------------------------:|:-------------:|:------:|
|        ME         |     $D_{Warp}+R_{Motion}$      |     1e-4      |   50   |
|  RECONSTRUCTION   |           $D_{Rec}$            |     1e-4      |   50   |
| CONTEXTUAL CODING |      $D_{Rec}+R_{Frame}$       |     1e-4      |   50   |
|        ALL        | $D_{Rec}+R_{Frame}+R_{Motion}$ |     1e-5      |   50   |

### FVC (Anchor)
|  Stage   |  Loss   | Learning Rate | Epochs |
|:--------:|:-------:|:-------------:|:------:|
| TRAINING | RD loss |  1e-4, 1e-5   | 148, 7 |

- Note that the multi-frame fusion module is removed in the FVC (Anchor).
- 
- Examples of training configuration for each network are provided in {root}/Train.


## Test Conditions

### General Conditions
- Test sequences include HEVC CTC (Class B, C, D, E), UVG (1080p, 7 sequences), and MCL-JCV (1080p, 30 sequences). 
- IntraPeriod is set to 32 and all frames in the sequence are encoded.
- PSNR is used to assess the reconstruction quality.
- it-per-pixel (BPP) is used to measure the bit cost, where BPP = bitrate_kbps * 1000 / frame_rate / resolution for traditional methods.
- BD-BR and BD-PSNR are used to evaluate the RD performance of different methods, and the anchor is set as HEVC.

### Configurations of Traditional Methods

| Method |      Configuration      |      Version      | Intra Period |                                                                        Example                                                                         |
|:------:|:-----------------------:|:-----------------:|:------------:|:------------------------------------------------------------------------------------------------------------------------------------------------------:|
|  HEVC  | encoder_lowdelay_P_main |     HM-16.21      |      32      |                           TAppEncoder -c encoder_lowdelay_P_main.cfg --QP={QP} --DecodingRefreshType=2 --IntraPeriod=32 ...                            |                          |
|  x264  |        veryslow         | ffmepg-aeceefa622 |      32      | ffmpeg -pix_fmt yuv420p -c:v libx264 -preset veryslow -tune zerolatency -tune psnr -bf 2 -b_strategy 0 -sc_threshold 0 -flags +psnr -qp {QP} -g 32 ... |
 |  x265  |        veryslow         | ffmepg-aeceefa622 |      32      |        ffmpeg -pix_fmt yuv420p -c:v libx265 -preset veryslow -tune zerolatency -tune psnr -x265-params "qp={QP}:keyint=32" -flags +psnr -y ...         |
|  VVC   | encoder_lowdelay_P_vtm  |     VTM-19.2      |      32      |                            EncoderApp -c encoder_lowdelay_P_vtm.cfg --QP={QP} --DecodingRefreshType=2 --IntraPeriod=32 ...                             |


- Levels in the configurations of non-CTC sequences are set to 4.1 (the same as in the configurations of 1080p CTC sequences)


### Test Conditions for End-to-end Methods
- The test process is depicted as follows:
![Process of the test of end-to-end methods.png](assets%2Ffigures%2FProcess%20of%20the%20test%20of%20end-to-end%20methods.png)

- The conversion of color space will cause neglectable distortion.
- The resolution of sequences is padded into the integral times of 64.
- The distortion is measured in the YCbCr color space as follows:
$$PSNR=(6*PSNRY+PSNRU+PSNRV)/8.$$
- The rate is measured as follows:
$$Bpp=BitstreamSize / Resolution / NumFrames.$$
- The header bits are stipulated as follows:

| GOP Size | Quality Index | Num Frame | Width | Height |                                            
|:--------:|:-------------:|:---------:|:-----:|:------:|
|    u8    |      u8       |    u16    |  u16  |  u16   |

- The lambda and network channels of the P-frame codec as well as the quality of the I-frame codec are derived by the quality index

| Quality Index |  1  |  2  |  3   |  4   |                                            
|:-------------:|:---:|:---:|:----:|:----:|
|    Lambda     | 256 | 512 | 1024 | 2048 |
|    Quality    |  3  |  4  |  5   |  6   |

- Examples of testing configuration are provided in {root}/Test/Config/Network.

| Configuration Item |                              Help                               |
|:------------------:|:---------------------------------------------------------------:|
|    ckpt_folder     |  model weights folder, where models are named as 1024.pth etc.  |
|      use_gpu       |              indicates if use GPU for compression               |
|   seq_cfg_folder   |             folders of sequence configuration files             |
|    quality_idx     |                      quality index (1 ~ 4)                      |
|  compress_folder   |                    folder to save bitstream                     |
|   results_folder   | folder to save coding results, e.g., PSNR and bpp for per frame |

- Sequence configuration are provided in {root}/Test/Config/Sequences, all you need to modify is the `base_path` item.

## Performance 
TODO






















