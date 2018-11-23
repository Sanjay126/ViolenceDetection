#
Learning to Detect Violent Videos using Convolutional Long Short-Term Memory


The source code associated with the paper [Learning to Detect Violent Videos using Convolutional Long Short-Term Memory](https://arxiv.org/abs/1709.06531), published in AVSS-2017. (*Experimental release*) 

#### Prerequisites
* Python 3.5
* Pytorch 0.3.0
#### Running


```
python main-run-vr.py --numEpochs 100 \
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
--trainRatio 0.8 \
--datasetDir ./dataset
```

