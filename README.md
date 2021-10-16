# FCNsignal

Base-resolution prediction of transcription factor binding signals by a deep learning framework. The performance of FCNsignal was evaluated on the [ChIP-seq and ATAC-seq datasets](https://www.encodeproject.org/).

<p align="center"> 
<img src=https://github.com/turningpoint1988/FCNsignal/flowchart.jpg>
</p>

## Prerequisites and Dependencies

- Pytorch 1.1 [[Install]](https://pytorch.org/)
- CUDA 9.0
- Python 3.6
- Python packages: biopython, scikit-learn, pyBigWig, scipy, pandas, matplotlib, seaborn
- Download [hg38.fa](https://hgdownload.soe.ucsc.edu/downloads.html#human) then unzip them and put them into `Genome/`

## Competing Methods

- [MEME Suite](https://meme-suite.org/meme/doc/download.html). It integrates several methods used by this paper, including MEME, STREME, TOMTOM and FIMO.
- [DeepCNN](https://github.com/turningpoint1988/DLBSS)
- [DanQ](https://github.com/uci-cbcl/DanQ)
- [FCNA\*](https://github.com/turningpoint1988/FCNA)
- [FCNA](https://github.com/turningpoint1988/FCNsignal)
- [BPNet](https://github.com/kundajelab/bpnet/)
- [LSGKM](https://github.com/Dongwon-Lee/lsgkm)
- [DeepEmbed](https://github.com/minxueric/ismb2017_lstm)
- [Deopen](https://github.com/kimmo1019/Deopen)
- [DeltaSVM](https://www.beerlab.org/deltasvm/)

## Data Preparation

```
python bed2signal.py -d <> -n <> -s <>
```

| Arguments   | Description                                                    |
| ----------- | -------------------------------------------------------------- |
| -d          | The path of datasets, e.g. /your_path/FCNsignal/HeLa-S3/CTCF   |
| -n          | The name of the specified dataset, e.g. CTCF                   |
| -s          | Random seed (default is 666)                                   |


## Model Training

Train FCNsignal models on specified datasets:

```
python run_signal.py -d <> -n <> -g <> -s <> -b <> -e <> -c <>
```

| Arguments  | Description                                                                      |
| ---------- | -------------------------------------------------------------------------------- |
| -d         | The path of a specified dataset, e.g. /your_path/FCNsignal/HeLa-S3/CTCF/data     |
| -n         | The name of the specified dataset, e.g. CTCF                                     |
| -g         | The GPU device id (default is 0)                                                 |
| -s         | Random seed                                                                      |
| -b         | The number of sequences in a batch size (default is 500)                         |
| -e         | The epoch of training steps (default is 50)                                      |
| -c         | The path for storing models, e.g. /your_path/FCNsignal/models/HeLa-S3/CTCF       |

### Output

Trained models for FCNsignal on the specified datasets. For example, A trained model can be found at ` /your_path/FCNsignal/models/HeLa-S3/CTCF/model.pth`.

## Model Classification

Test FCNsignal models on the specified test data:

```
python test_signal.py -d <> -n <> -g <> -c <>
```

| Arguments  | Description                                                                                 |
| ---------- | ------------------------------------------------------------------------------------------- |
| -d         | The path of a specified dataset, e.g. /your_path/FCNsignal/HeLa-S3/CTCF/data                |
| -n         | The name of the specified dataset, e.g. CTCF                                                |
| -g         | The GPU device id (default is 0)                                                            |
| -c         | The trained model path of a specified dataset, e.g. /your_path/FCNsignal/models/HeLa-S3/CTCF|

### Output

Generate `record.txt` indicating the mean squared error (MSE), the pearson correlation coefficient (Pearsonr), the area under the receiver operating characteristic curve (AUC) and the area under the precision-recall curve (PRAUC) of the trained model in predicting binding signals on the test data.

## Motif Prediction

Motif prediction on the specified test data:

```
python motif_prediction.py -d <> -n <> -g <> -t <> -c <> -o <>
```

| Arguments  | Description                                                                                 |
| ---------- | ------------------------------------------------------------------------------------------- |
| -d         | The path of a specified dataset, e.g. /your_path/FCNsignal/HeLa-S3/CTCF/data                |
| -n         | The name of the specified dataset, e.g. CTCF                                                |
| -g         | The GPU device id (default is 0)                                                            |
| -t         | The threshold value (default is 0.3)                                                        |
| -c         | The trained model path of a specified dataset, e.g. /your_path/FCNsignal/models/HeLa-S3/CTCF|
| -o         | The path of storing motif files, e.g. /your_path/FCNsignal/motifs/HeLa-S3/CTCF              |

### Output

Generate motif files in MEME format, which are subsequently used by TOMTOM.


## Locating TFBSs

Locating potential binding regions on the inputs of arbitrary length:

```
python TFBS_locating.py -i <> -n <> -g <> -t <> -w <> -c <>
```
| Arguments  | Description                                                                                 |
| ---------- | ------------------------------------------------------------------------------------------- |
| -i         | The input file in bed format, e.g. /your_path/FCNsignal/input.bed                           |
| -n         | The name of the specified dataset, e.g. CTCF                                                |
| -g         | The GPU device id (default is 0)                                                            |
| -t         | The threshold value to determine the binding regions (default is 1.5)                       |
| -w         | The length of the binding regions (default is 60)                                           |
| -c         | The trained model path of a specified dataset, e.g. /your_path/FCNsignal/models/HeLa-S3/CTCF|

### Output

The outputs include the base-resolution prediction of inputs and the positions of potential binding regions in the genome (in bed format). <br/>
We also provide the line plots of the above base-resolutiion prediction.

