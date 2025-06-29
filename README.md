# KIKT:Weakly Supervised Dynamic Scene Graph Generation with Temporal-enhanced In-domain Knowledge Transferring

## Installation
Following PLA/environment.yaml to construct the virtual environment.

## Dataset
### Data preperation
1. For object detection results, we use pre-trained object detector [VinVL](https://github.com/pzzhang/VinVL), you can follow steps in our baseline [PLA](https://github.com/zjucsq/PLA/tree/master) to generate them on your own, or directly use our pre-processed detection results in step 3 below.

2. For dataset, download from [Action Genome](https://github.com/JingweiJ/ActionGenome).

3. Download necessary weakly-supervised annotation files and pre-trained weight (stored in [LINK](https://pan.baidu.com/s/1_b9CM8omaNwXUQsNTIfA5Q) with password 1234)), the final data structure should be like

```
| -- data
     | -- action-genome
           | -- frames    
           | -- videos    
           | -- annotations 
           | -- AG_detection_results_refine 
| -- refine
      | -- output # pre-trained relation aware transformer weight
| -- PLA
      | -- model # pre-trained scene graphe generation weight
| -- RAFT
      
```

## Evaluation

### Object Detection Performance on Relation-Aware Transformer(TIKT) Model
```
cd ~/refine
python scripts/evaluate.py # evaluate the performance of object detection
```
| Model  | AP@1 |AP@10|AR@1 | AR@10|Weight|
| --- | ----------- |----- |----- |----- |----- |
|PLA(baseline)    | 11.4 |11.6 |33.3 |37.6| -|
| Ours  | 24.3|26.6|30.2|45.1|[weight](https://pan.baidu.com/s/11y79PFA7RoULOfT_OxR3-A) password 1234|

### Scene Graph Generation Performance on DSGG(PLA) Model
```
cd ~/PLA
python test.py --cfg configs/final.yml # for final scene graph generation performance evaluation
```
| Model  | W/R@10|W/R@20|W/R@50|N/R@10|N/R@20|N/R@50|weight|
| --- | ----------- |----- |----- |----- |----- |----- |----- |
|PLA(baseline)    | 14.32|20.42|25.43|14.78|21.72|30.87|-|
| Ours  | 17.56| 22.33|27.45| 18.76|24.49 |33.92|[weight](https://pan.baidu.com/s/1ES3J0s2L0EKb45iPuPs6-A) password 1234|
## Training

### Step1. Optical Flow Extraction
We use [RAFT](https://github.com/princeton-vl/RAFT) to generate the optical flow in our data, you can either use our pre-processed optical flow (stored in []()) or generate them on you own by following steps:

```
cd ~/RAFT   ## download the RAFT ckpt accordingly
python process_optical_flow.py
python post_process.py
```
Then place the generated optical flow file for train and test set under folder ~/data/action-genome/.
### Step2. Relation-aware Refine Model Training
```
cd ~/refine
python scripts/train.py 
```
### Step3. Scene Graph Generation Model Training
```
cd ~/PLA
python train.py --cfg configs/oneframe.yml # after this line training, select the best oneframe ckpt as the model_path parameter in oneframe.yml for next line training
python train.py --cfg configs/final.yml # for video SGG model
```
## Acknowledgement
We build our project upon [PLA](https://github.com/zjucsq/PLA/tree/master), [RAFT](https://github.com/princeton-vl/RAFT), thanks for their works.
## Citation
```
```


