import torch
import os
import glob
from spatial_transforms import (Compose, ToTensor, FiveCrops, Scale, Normalize, MultiScaleCornerCrop,
                                RandomHorizontalFlip, TenCrops, FlippedImagesTest, CenterCrop)
from makeDataset import *
from createModel import *
from tensorboardX import SummaryWriter
import sys
import argparse
import matplotlib.pyplot as plt


def sampleFromClass(ds,classCount, k):
    class_counts = {}
    train_data = []
    train_label = []
    test_data = []
    test_label = []
    data, label= ds
    ds = zip(data, label)
    for data, label in ds:
        c = label
        class_counts[c] = class_counts.get(c, 0) + 1
        if class_counts[c] <= k*classCount[label]:
            train_data.append(data)
            train_label.append(label)
        else:
            test_data.append(data)
            test_label.append(label)

    return (train_data, train_label), (test_data, test_label)



def make_split(dir):
    Dataset=[]
    Labels=[]
    classCount=[]
    i=0
    count=0
    for directory in sorted(os.listdir(os.path.abspath(dir))):
        for target in sorted(os.listdir(os.path.join(dir, directory))):
            d = os.path.join(dir,directory, target)
            if os.path.isdir(d):
                continue
            Dataset.append(d)
            Labels.append(i)
            count+=1
        i+=1
        classCount.append(count)
        count=0
    return (Dataset, Labels),classCount

def main_run(numEpochs, lr, stepSize, decayRate, trainBatchSize, seqLen,
             evalInterval, evalMode, numWorkers, outDir,modelUsed,pretrained,minTrainEx,directory):



    compDataset,classCount = make_split(directory)
    (trainDataset,trainLabels),(testDataset,testLabels)=sampleFromClass(compDataset,classCount,minTrainEx)
    

    mean=[0.485, 0.456, 0.406]
    std=[0.229, 0.224, 0.225]
    normalize = Normalize(mean=mean, std=std)
    spatial_transform = Compose([Scale(256), RandomHorizontalFlip(), MultiScaleCornerCrop([1, 0.875, 0.75, 0.65625], 224),
                                 ToTensor(), normalize])

    vidSeqTrain = makeDataset(trainDataset, trainLabels, spatial_transform=spatial_transform,
                                seqLen=seqLen)

    trainLoader = torch.utils.data.DataLoader(vidSeqTrain, batch_size=trainBatchSize,
                            shuffle=True, num_workers=numWorkers, pin_memory=True, drop_last=True)

    if evalMode == 'centerCrop':
        test_spatial_transform = Compose([Scale(256), CenterCrop(224), ToTensor(), normalize])
        testBatchSize = 1
    elif evalMode == 'tenCrops':
        test_spatial_transform = Compose([Scale(256), TenCrops(size=224, mean=mean, std=std)])
        testBatchSize = 1
    elif evalMode == 'fiveCrops':
        test_spatial_transform = Compose([Scale(256), FiveCrops(size=224, mean=mean, std=std)])
        testBatchSize = 1
    elif evalMode == 'horFlip':
        test_spatial_transform = Compose([Scale(256), CenterCrop(224), FlippedImagesTest(mean=mean, std=std)])
        testBatchSize = 1

    vidSeqTest = makeDataset(testDataset, testLabels, seqLen=seqLen,
    spatial_transform=test_spatial_transform)


    testLoader = torch.utils.data.DataLoader(vidSeqTest, batch_size=testBatchSize,
                            shuffle=False, num_workers=int(numWorkers/2), pin_memory=True)


    numTrainInstances = vidSeqTrain.__len__()
    numTestInstances = vidSeqTest.__len__()

    print('Number of training samples = {}'.format(numTrainInstances))
    print('Number of testing samples = {}'.format(numTestInstances))

    modelFolder = './experiments_' + outDir+'_'+modelUsed # Dir for saving models and log files
    # Create the dir
    if os.path.exists(modelFolder):
        print(modelFolder + ' exists!!!')
        sys.exit()
    else:
        os.makedirs(modelFolder)
    # Log files
    writer = SummaryWriter(modelFolder)
    trainLogLoss = open((modelFolder + '/trainLogLoss.txt'), 'w')
    trainLogAcc = open((modelFolder + '/trainLogAcc.txt'), 'w')
    testLogLoss = open((modelFolder + '/testLogLoss.txt'), 'w')
    testLogAcc = open((modelFolder + '/testLogAcc.txt'), 'w')


    model = ViolenceModel(modelUsed,pretrained)


    trainParams = []
    for params in model.parameters():
        params.requires_grad = True
        trainParams += [params]
    model.train(True)
    if(torch.cuda.is_available()):
        model.cuda()

    lossFn = nn.CrossEntropyLoss()
    optimizerFn = torch.optim.RMSprop(trainParams, lr=lr)
    optimScheduler = torch.optim.lr_scheduler.StepLR(optimizerFn, stepSize, decayRate)

    minAccuracy = 50
    train_loss=[]
    val_loss=[]
    train_acc=[]
    val_acc=[]

    for epoch in range(numEpochs):
        optimScheduler.step()
        epochLoss = 0
        numCorrTrain = 0
        iterPerEpoch = 0
        model.train(True)
        print('Epoch = {}'.format(epoch + 1))
        writer.add_scalar('lr', optimizerFn.param_groups[0]['lr'], epoch+1)
        for i, (inputs, targets) in enumerate(trainLoader):
            iterPerEpoch += 1
            optimizerFn.zero_grad()
            if(torch.cuda.is_available()):
                inputVariable1 = Variable(inputs.permute(1, 0, 2, 3, 4).cuda())
                labelVariable = Variable(targets.cuda())
            else:
                inputVariable1=Variable(inputs.permute(1,0,2,3,4))
                labelVariable=Variable(targets)
            outputLabel = model(inputVariable1)
            loss = lossFn(outputLabel, labelVariable)
            loss.backward()
            optimizerFn.step()
            outputProb = torch.nn.Softmax(dim=1)(outputLabel)
            _, predicted = torch.max(outputProb.data, 1)
            if(torch.cuda.is_available()):
                numCorrTrain += (predicted == targets.cuda()).sum()
            else:
                numCorrTrain+=(predicted==targets).sum()
            epochLoss += loss.data[0]
        avgLoss = epochLoss/iterPerEpoch
        trainAccuracy = (float(numCorrTrain) * 100)/float(numTrainInstances)
        train_loss.append(avgLoss)
        train_acc.append(trainAccuracy)
        print('Training: Loss = {} | Accuracy = {}% '.format(avgLoss, trainAccuracy))
        writer.add_scalar('train/epochLoss', avgLoss, epoch+1)
        writer.add_scalar('train/accuracy', trainAccuracy, epoch+1)
        trainLogLoss.write('Training loss after {} epoch = {}\n'.format(epoch+1, avgLoss))
        trainLogAcc.write('Training accuracy after {} epoch = {}\n'.format(epoch+1, trainAccuracy))

        if (epoch+1) % evalInterval == 0:
            model.train(False)
            print('Evaluating...')
            testLossEpoch = 0
            testIter = 0
            numCorrTest = 0
            for j, (inputs, targets) in enumerate(testLoader):
                testIter += 1
                if evalMode == 'centerCrop':
                    if(torch.cuda.is_available()):
                        inputVariable1 = Variable(inputs.permute(1, 0, 2, 3, 4).cuda(), requires_grad=False)
                        labelVariable = Variable(targets.cuda(async=True), requires_grad=False)
                    else:
                        inputVariable1 = Variable(inputs.permute(1, 0, 2, 3, 4), requires_grad=False)
                        labelVariable = Variable(targets, requires_grad=False)
                else:
                    if(torch.cuda.is_available()):
                        inputVariable1 = Variable(inputs[0].permute(1, 0, 2, 3, 4).cuda(), requires_grad=False)
                        labelVariable = Variable(targets.cuda(async=True), requires_grad=False)
                    else:
                        inputVariable1 = Variable(inputs[0].permute(1, 0, 2, 3, 4), requires_grad=False)
                        labelVariable = Variable(targets, requires_grad=False)
                outputLabel = model(inputVariable1)
                outputLabel_mean = torch.mean(outputLabel, 0, True)
                testLoss = lossFn(outputLabel_mean, labelVariable)
                testLossEpoch += testLoss.data[0]
                _, predicted = torch.max(outputLabel_mean.data, 1)
                if(torch.cuda.is_available()):
                    numCorrTest += (predicted == targets[0].cuda()).sum()
                else:
                    numCorrTest += (predicted == targets[0]).sum()
            testAccuracy = (float(numCorrTest) * 100)/float(numTestInstances)
            avgTestLoss = testLossEpoch / testIter
            val_loss.append(avgTestLoss)
            val_acc.append(testAccuracy)
            print('Testing: Loss = {} | Accuracy = {}% '.format(avgTestLoss, testAccuracy))
            writer.add_scalar('test/epochloss', avgTestLoss, epoch + 1)
            writer.add_scalar('test/accuracy', testAccuracy, epoch + 1)
            testLogLoss.write('Test Loss after {} epochs = {}\n'.format(epoch + 1, avgTestLoss))
            testLogAcc.write('Test Accuracy after {} epochs = {}%\n'.format(epoch + 1, testAccuracy))
            if testAccuracy > minAccuracy:
                savePathClassifier = (modelFolder + '/bestModel.pth')
                torch.save(model, savePathClassifier)
                minAccuracy = testAccuracy
    xc=range(1,numEpochs+1)
    xv=[]
    for i in xc:
        if(i%evalInterval==0):
            xv.append(i)
    plt.figure(1,figsize=(7,5))
    plt.plot(xc,train_loss)
    plt.plot(xv,val_loss)
    plt.xlabel('num of Epochs')
    plt.ylabel('loss')
    plt.title('train_loss vs val_loss')
    plt.grid(True)
    plt.legend(['train','val'])
    #print plt.style.available # use bmh, classic,ggplot for big pictures
    plt.style.use(['classic'])
    plt.savefig(modelFolder+"/lossCurve.png")

    plt.figure(2,figsize=(7,5))
    plt.plot(xc,train_acc)
    plt.plot(xv,val_acc)
    plt.xlabel('num of Epochs')
    plt.ylabel('accuracy')
    plt.title('train_acc vs val_acc')
    plt.grid(True)
    plt.legend(['train','val'],loc=4)
    #print plt.style.available # use bmh, classic,ggplot for big pictures
    plt.style.use(['classic'])
    plt.savefig(modelFolder+"/accuracyCurve.png")
    #plt.show()
    trainLogAcc.close()
    testLogAcc.close()
    trainLogLoss.close()
    testLogLoss.close()
    writer.export_scalars_to_json(modelFolder + "/all_scalars.json")
    writer.close()
    return True

def __main__():
    parser = argparse.ArgumentParser()
    parser.add_argument('--numEpochs', type=int, default=100, help='Number of epochs')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--stepSize', type=int, default=25, help='Learning rate decay step')
    parser.add_argument('--decayRate', type=float, default=0.5, help='Learning rate decay rate')
    parser.add_argument('--seqLen', type=int, default=20, help='Length of sequence')
    parser.add_argument('--trainBatchSize', type=int, default=6, help='Training batch size')
    parser.add_argument('--evalInterval', type=int, default=5, help='Evaluation interval')
    parser.add_argument('--evalMode', type=str, default='centerCrop', help='Evaluation mode', choices=['centerCrop', 'horFlip', 'fiveCrops', 'tenCrops'])
    parser.add_argument('--numWorkers', type=int, default=4, help='Number of workers for dataloader')
    parser.add_argument('--outDir', type=str, default='violence', help='Output directory')
    parser.add_argument('--modelUsed', type=str, default='alexnet', help='Output directory')
    parser.add_argument('--pretrained', type=bool, default=True, help='Output directory')
    parser.add_argument('--minTrainEx', type=float, default=0.8, help='Output directory')
    parser.add_argument('--datasetDir', type=str, default='/floyd/input/violencedataset', help='Output directory')
    args = parser.parse_args()

    numEpochs = args.numEpochs
    lr = args.lr
    stepSize = args.stepSize
    decayRate = args.decayRate
    seqLen = args.seqLen
    trainBatchSize = args.trainBatchSize
    evalInterval = args.evalInterval
    evalMode = args.evalMode
    numWorkers = args.numWorkers
    outDir = args.outDir
    modelUsed=args.modelUsed
    pretrained=args.pretrained
    minTrainEx=args.minTrainEx
    datasetDir=args.datasetDir
    main_run(numEpochs, lr, stepSize, decayRate, trainBatchSize, seqLen,
             evalInterval, evalMode, numWorkers, outDir,modelUsed,pretrained,minTrainEx,datasetDir)
if __name__=='__main__':
    __main__()
