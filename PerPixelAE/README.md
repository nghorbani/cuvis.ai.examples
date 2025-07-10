# Simple anomaly detection example with per-pixel AE

## Introduction

In this example, we provide a framework to train a deep, per-pixel anomaly detector.

## The dataset

In our example dataset we used a custom-built camera assembly that creates six channel cubes. Three channels are from
a 24-megapixel RGB-camera and the other three are SWIR channels with 1050nm, 1200nm and 1450nm wavelength respectively.

The dataset is precisely built for unsupervised multispectral anomaly detection. It features 255 images,
which are divided into normal and anomalous images. We used sawdust in a wooden tray to create unique images on which
the model could lean. As anomalies, we used PLA, alcohol, leaves, a fake leaf, PET, POMC, transparent plastic
foil as well as water to demonstrate the capabilities of our SWIR setup.

The dataset can be
downloaded [here](https://drive.google.com/drive/folders/1bTNNSiFBQdPLgFlt3DHt06KmShmeTftj?usp=drive_link).

Notes on what the validation images show can be found in the ``dataset_notes.md``

## Model

The model implements a classical anomaly detector in an encoder-decoder paradigm. We provide two configured network sizes, **small** and **large** which define networks with varying numbers of neurons in the fully connected layers. Additionally, the size of the encoding dimension can be configured with the included train config yaml file.

## Prerequisite

This example is written using Python version 3.12 and cuvis SDK version 3.3.1, which can be
downloaded [here](https://cloud.cubert-gmbh.de/s/qpxkyWkycrmBK9m?path=%2FCuvis%203.3.1).

To get this example running, please install PyTorch with CUDA support from
their [website](https://pytorch.org/get-started/locally/). This example is tested for PyTorch version 2.6.0+cu124.

## How to train

After downloading the sample dataset and extracting it into the data folder, we can now run the train.py script.

```
train.py -c example_train_config.yaml
```

The `example_train_config.yaml` has every parameter and path in it for the model and dataloader to work.
If you chose to alter the folder structure, you may need to change some paths in there before the training is able to
run.

## How to create a report for the model and dataset

You can use `report.py` in order to create a report of the model performance and generate a visual representation of
the outcome.
The script will create a folder at a specified location, infer the given datasets, and create a visually pleasing output.

```
reporting.py -c example_report_config.yaml
```