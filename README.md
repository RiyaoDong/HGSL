# HGSL

Implementation of Hard global softmin loss for fine-grained object retrieval, which is proposed in [Improved Fine-Grained Object Retrieval with Hard Global Softmin Loss Objective](under review).

## Requirements
```
Python 3, PyTorch >= 1.1.0
```

## Datasets preparation

We conduct our experiments on four popular fine-grained datasets, e.g. CUB-200-2011, Stanford Cars, FGVC aircraft, Oxford Flowers. You can prepare the datasets yourself, or just download the datasets that we packed. Link https://drive.google.com/file/d/1CYTeZ8VooiC8AUKDyvJvivc_swbN42dI/view?usp=sharing

We unzip the datasets to "/content/datasets/data", you can change the folder and change the train.py accordingly.
```
mkdir -p /content/datasets/data
unzip -d /content/datasets/data/ data_all.zip 
```

## Training ResNet50 on CUB with HGSL

```
bash train.sh
```

## Contact

Feel free to discuss papers/code with us through issues/emails!

wdecen@foxmail.com


