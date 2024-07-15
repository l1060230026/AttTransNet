# Pytorch Implementation of AttTransNet

## Install
The latest codes are tested on Ubuntu 20.04, CUDA11.6, PyTorch 1.12 and Python 3.8:

## Semantic Segmentation (S3DIS)
### Data Preparation
Download 3D indoor parsing dataset (**S3DIS**) [here](http://buildingparser.stanford.edu/dataset.html)  and save in `data/s3dis/Stanford3dDataset_v1.2_Aligned_Version/`.
```
cd data_utils
python collect_indoor3d_data.py
```
Processed data will save in `data/stanford_indoor3d/`.
### Run
```
## Check model in ./models 
## e.g., pointnet2_ssg
python train_semseg_att.py --model pointnet2_sem_seg_att --test_area 5 --log_dir pointnet2_att
python test_semseg_att.py --log_dir pointnet2_att --test_area 5 --visual
```
Visualization results will save in `log/sem_seg/pointnet2_att/visual/` and you can visualize these .obj file by [MeshLab](http://www.meshlab.net/).

### Performance
|Model  | Precision |Class avg IoU
|--|--|--|
| PointNet++ (Pytorch) | 0.600 | 0.525 |
| RandLA (Pytorch) | 0.550 | 0.447 |
| AttTransNet (Pytorch) | **0.736** | **0.651**|

## Visualization
### Using show3d_balls.py
```
## build C++ code for visualization
cd visualizer
bash build.sh 
## run one example 
python show3d_balls.py
```
![](/visualizer/pic.png)
### Using MeshLab
![](/visualizer/pic2.png)


## Reference By
[yanx27/Pointnet_Pointnet2_pytorch](https://github.com/yanx27/Pointnet_Pointnet2_pytorch)<br>
[QingyongHu/RandLA-Net](https://github.com/QingyongHu/RandLA-Net)<br>
[POSTECH-CVLab/point-transformer](https://github.com/POSTECH-CVLab/point-transformer) <br>