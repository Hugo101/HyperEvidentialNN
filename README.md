# HyperEvidentialNN
The initial implementation of the HENN: [Hyper Evidential Deep Learning to Quantify Composite Classification Uncertainty](https://openreview.net/forum?id=A7t7z6g6tM) which was accepted in ICLR 2024.
************************************************************************************************
 
 
## **Configuration**
 config.yml contains all experimental settings for our HENN model.
 

## **Dataset**
The folder "data" contains all data preparation files for different datasets, including tinyImageNet, Living17, Nonliving26, CIFAR100, and NAbirds.

Each data file contains different processing methods for different methods: HENN, DNN, and ENN, etc.

## **HENN**

- **GDD_main.py**
This is the main file to run our HENN code.
 
- **GDD_train.py**
This file contains the model training detail.


- **GDD_evaluate.py**
This file contains the evaluation of the HENN model in validation set.

- **GDD_test.py**
This file contains the evaluation metric calculation during test phase of the HENN model.

## **Baseline**
- baseline_DetNN.py is the main file for baseline DNN.
- baseline_ENN.py is the main file for baseline ENN.
 
 
## **Helper functions**
- helper_functions.py contains the necessary data preparation functions and necessary functions for HENN training.

## **Results Representation**
- All results are saved in cloud using wandb.
- roc_draw_plot.ipynb For figures we generated, such as AUROC curves.


## Citing
If you find HENN useful in your work, please consider citing the following BibTeX entry:
```python
@inproceedings{
li2024hyper,
title={Hyper Evidential Deep Learning to Quantify Composite Classification Uncertainty},
author={Changbin Li and Kangshuo Li and Yuzhe Ou and Lance M. Kaplan and Audun J{\o}sang and Jin-Hee Cho and DONG HYUN JEONG and Feng Chen},
booktitle={The Twelfth International Conference on Learning Representations},
year={2024},
url={https://openreview.net/forum?id=A7t7z6g6tM}
}
```
