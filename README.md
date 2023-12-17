# HyperEvidentialNN
The implementation of the HENN paper which was submitted and is under review.
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