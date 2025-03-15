# Brep2X

Exploration based on [Brep2Seq](https://github.com/zhangshuming0668/Brep2Seq.git)

## Preparation

### Environment setup

```
https://github.com/suneric-adsk/Brep2X.git
cd Brep2X
conda env create -f environment.yml
conda activate brep2x
```

### Dataset

Original paper published [synthetic CAD models dataset](https://www.scidb.cn/en/detail?dataSetId=87b0695c592849618d3d22d0ab480849&version=V1) 

## Training

To reconstruct B-rep models, the network can be trained using:
```
python reconstruction.py train --ds /path/to/dataset --bs 64
```

The logs and checkpoints will be stored in a folder called `results` based on the experiment name and timestamp, and can be monitored with Tensorboard:

```
tensorboard --logdir results/<experiment_name>
```

## Test

The best checkpoints based on the smallest validation loss are saved in the results folder. The checkpoints can be used to test the model as follows:

```
python reconstruction.py test --ds /path/to/dataset --ckpt ./results/BrepToSeq/best.ckpt --bs 32
```

The predicted reconstruction sequences files are saved in the results folder `results/predicted_seq`.

## Visualization

For visualizing the generated sequence file

```
python visualization.py --json path/to/file.json
```
