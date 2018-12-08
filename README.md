#
Learning to Detect Violent Videos using Convolutional Long Short-Term Memory


The project uses Convolutional Neural Network architecture and Convolutional Long Short-Term Memory to classify video in four different classes based on type of violence in them. We started with [this](https://arxiv.org/abs/1709.06531) research paper as reference and then improved its performance by changing the architecture and then increasing the number of classes for classification of videos. It is an ongoing project so you may encounter some bugs.


#### Running
```
												###Using MakeFile###
Just download the makefile and run (downloads project and dataset and install dependencies)
$ make get-project

For Downloading dependencies 
$ make install-dependencies

For CrossValidation
$ make crossvalidate numEpochs=50 input='./dataset' folds=5

For Training
$ make train numEpochs=50 Model='vgg16' input='./dataset'
```
```
												###Using python in command line###
#### Prerequisites
* Python 3.5+
* Pytorch 0.3.0+
* matplotlib
* numpy
* torchvision
* tensorboardX
* sklearn
* opencv
* CUDA (optional but recommended)


python3 main-run-vr.py --numEpochs 100 \
--lr 1e-4 \
--stepSize 25 \
--decayRate 0.5 \
--seqLen 20 \
--trainBatchSize 16 \
--evalInterval 5 \
--evalMode horFlip \
--numWorkers 4 \
--outDir violence \
--modelUsed alexnet \
--pretrained True \
--trainRatio 0.6 \
--datasetDir ./dataset \
--crossValidation True \
--folds 5


python3 main-run-vr.py --numEpochs 100 \
--lr 1e-4 \
--stepSize 25 \
--decayRate 0.5 \
--seqLen 20 \
--trainBatchSize 16 \
--evalInterval 5 \
--evalMode horFlip \
--numWorkers 4 \
--outDir violence \
--modelUsed alexnet \
--pretrained True \
--trainRatio 0.6 \
--datasetDir ./dataset \
--crossValidation False \
```

