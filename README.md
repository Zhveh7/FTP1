# Unifying Channel Independence and Mixing: Multi-Scale Patch Recursion for Global–Local Representation Synergy in Multivariate Time Series Forecasting

This repository contains the implementation of the paper **"Unifying Channel Independence and Mixing: Multi-Scale Patch Recursion for Global–Local Representation Synergy in Multivariate Time Series Forecasting"** using PyTorch. By leveraging a more sophisticated linear layer structure, We achieves significantly better prediction performance on several benchmark datasets compared to state-of-the-art methods, demonstrating its superior generalization ability and effectiveness.

## Main Module Components

### FTP
The overall architecture of FusionTimePatch (FTP). FTP consists of three core components: Dual-GLF introduces CI and CM perspectives in parallel, leveraging multi-scale patch recursion to capture both local and global temporal patterns; the CE module enhances salient channel features and diffuses them across channels, improving sensitivity to anomalies and underlying drivers; the Fusion layer linearly merges the outputs of Dual-GLF and CE, concatenates the result with the original encoder input, propagates the combined representation through the encoder stack, and finally applies an MLP head to produce the forecast. This design enables the synergistic modeling of unified CI/CM perspectives and temporal structures. Below are two types of FTP model diagrams.

![](imgs/StyleA.png)

![](imgs/StyleB.png)

## Environment Requirements

python==3.10.14

torch==2.3.0+cu121

numpy==1.26.4

scikit-learn==1.5.1

pandas==2.2.2

## Dataset

You can download the required datasets for this model directly from [Google Drive](https://drive.google.com/drive/folders/1mR8pk5J6dZAmByptTZ1LBt4l2B-0I8oA?usp=sharing). Once the datasets are downloaded, please place the `dataset` folder in the FTP directory.

## Running the Code

To run the model, execute the scripts located in `./scripts/long_term_forecast/` from the FTP directory.

For example, to predict the Weather dataset using FTP, run the following command:

`sh ./scripts/long_term_forecast/Weather_script/Weather_FTP.sh`

## Acknowledgements

We would like to thank the following GitHub repositories for their valuable codebases or datasets, which contributed significantly to time series prediction tasks:

https://github.com/zhouhaoyi/Informer2020

https://github.com/thuml/Time-Series-Library

https://github.com/thuml/iTransformer

https://github.com/Secilia-Cxy/SOFTS



















