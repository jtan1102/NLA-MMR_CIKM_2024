# This is the codebase for CIKM2024 submission 53


#### Environment setting

```
Python                    3.8.17
torch                     2.0.1
torch-geometric           2.3.1
torch-scatter             2.0.9
torchsummary              1.5.1
tokenizers                0.13.3
numpy                     1.23.5
nvidia-cublas-cu11        11.10.3.66
nvidia-cuda-runtime-cu11  11.7.99
nvidia-cudnn-cu11         8.5.0.96
transformers              4.26.1
tokenizers                0.13.3
scikit-learn              1.3.0
scipy                     1.10.1
pillow                    10.0.0
pandas                    2.0.3
six                       1.16.0
wheel                     0.38.4
tqdm                      4.66.1
```

#### Steps to reproduce


1.Download and prepare data
```
// for each dataset in mimic-iii/mimic-iv, perform the following commands
open the google drive and download the compressed mimic3/mimic4 data:
https://drive.google.com/drive/folders/14QF1oS2ziLqsycKDDWeDOmmfrJL2jzWL?usp=sharing

tar -xzvf <directory_to_download>/bio_pt.tar.gz
```

2.Clone this repo
```
cd <directory_to_code>
git clone https://github.com/x6p2n9q8a4/NLA-MMR_CIKM_2024
```

3.Puts the data to specific location

```
cd <directory_to_code>
//similar command for mimic-iv
mv <directory_to_download>/bio_pt <directory_to_code>/data/MIMIC-III_data/
```

4.Training and Inference

```
cd scripts

// For mimic-iii
python -u main.py
       --device <your_gpu_id>
       --dropout_ratio 0.2
       --bce_weight 0.95
       --output_model_dir <path_to_your_checkpoints>
       --max_visit_num 2
       --decay 1e-5
       --dataset MIMIC-III
       --batch_size 8
       --lr 5e-4
       --pt_mode bio_pt
       --Train

// For mimic-iv
python -u main.py
       --device <your_gpu_id>
       --dropout_ratio 0.2
       --bce_weight 0.95
       --output_model_dir <path_to_your_checkpoints>
       --max_visit_num 3
       --decay 3e-5        
       --dataset MIMIC-IV
       --batch_size 8 
       --lr 5e-4
       --pt_mode bio_pt
       --Train
```

#### Reference log output

In normal training process, you will see log as follows:

```
---------Epoch 39-----------
Start training!

training step: 0 / 1166, REC loss: 0.6013,   loss_bce: 0.2840, loss_multi: 6.6307

training step: 1000 / 1166, REC loss: 0.5458,   loss_bce: 0.2596, loss_multi: 5.9838
REC Loss: 0.55829	BCE Loss:0.26742	Multi Loss:6.08485	Time: 44.68471
```

We perform testing on the valid dataset after each training epoch and you will see log as follows:

```

Valid perfomance
Start testing!

DDI Rate: 0.0918, Jaccard: 0.5293, PRAUC: 0.7788, AVG_PRC: 0.6626, AVG_RECALL: 0.7324, AVG_F1: 0.6846, AVG_MED: 22.4029
REC Loss: 0.60064	BCE loss:0.28283	 Multi loss:6.63895	Time: 11.60027
best_ja 0.5321197118283028
best_epoch 38
```
