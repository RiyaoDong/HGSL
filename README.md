# HGSL
Hard global softmin loss for fine-grained object retrieval.

Implementation of Hard global softmin loss for fine-grained object retrieval.

## Requirements
```
Python 3, PyTorch >= 1.1.0
```

## Datasets preparation

We conduct our experiments on four popular fine-grained datasets, e.g. CUB-200-2011, Stanford Cars, FGVC aircraft, Oxford Flowers. You can prepare the datasets yourself, or just download the datasets that we packed. Link https://drive.google.com/file/d/1CYTeZ8VooiC8AUKDyvJvivc_swbN42dI/view?usp=sharing

We unzip the datasets at "/content/datasets/data", you can change the directory and modify the train.py correspondingly. 
```
mkdir -p /content/datasets/data
unzip -d /content/datasets/data/ data_all.zip 
```

## Training ResNet50 on CUB with HGSL

```
./train.sh
```

