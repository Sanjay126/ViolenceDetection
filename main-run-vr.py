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
from random import shuffle
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import KFold
import numpy as np
import itertools

def sampleFromClass(ds,classCount, k):   #test,train and validation split
    class_counts = {}
    train_data = []
    train_label = []
    valid_data=[]
    valid_label=[]
    test_data = []
    test_label = []
    data, label= ds
    ds = zip(data, label)
    ds=list(ds)
    shuffle(ds)
    for data, label in ds:
        c = label
        class_counts[c] = class_counts.get(c, 0) + 1
        if class_counts[c] <= k*classCount[label]:
            train_data.append(data)
            train_label.append(label)
        elif(class_counts[c]<=(k+1)/2*classCount[label]):
            valid_data.append(data)
            valid_label.append(label)
        else:
            test_data.append(data)
            test_label.append(label)

    return (train_data, train_label),(valid_data,valid_label) ,(test_data, test_label)



def make_split(dir): # reads inp dir and takes inputs of diff classes
    Dataset=[]
    Labels=[]
    classCount=[]
    class_names=[]
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
        class_names.append(directory)
        i+=1
        classCount.append(count)
        count=0
    return (Dataset, Labels),classCount,class_names
def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):    # plots confusion matrix
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
def kFoldCrossValid(folds,data,label,Epochs,evalMode,numWorkers,lr, stepSize, decayRate, trainBatchSize, seqLen):
    models=['vgg16','vgg16_bn','vgg19','vgg19_bn','alexnet','resnet50']
    kf=KFold(folds,shuffle=True)
    accuracyList=[0,0,0,0,0,0]
    for j in range(len(models)):
        for train_index,test_index in kf.split(data,label):
            trainDataset=[]
            trainLabels=[]
            testDataset=[]
            testLabels=[]
            for i in train_index:
                trainDataset.append(data[i])
                trainLabels.append(label[i])
            for i in test_index:
                testDataset.append(data[i])
                testLabels.append(label[i])
            _,accuracy=modelTrain(models[j],True,trainDataset,trainLabels,testDataset,testLabels,Epochs,Epochs,evalMode,'/crossvalid',numWorkers,lr, stepSize, decayRate, trainBatchSize, seqLen,False)
            accuracyList[j]+=accuracy
        accuracyList[j]/=kf.get_n_splits()
    plot_bar_x(models,accuracyList)
    sys.exit()

def plot_bar_x(models,accuracyList):
    # this is for plotting bar graph 
    index = np.arange(len(models))
    plt.bar(index, accuracyList)
    plt.xlabel('models', fontsize=5)
    plt.ylabel('accuracy', fontsize=5)
    plt.xticks(index, models, fontsize=5, rotation=30)
    plt.title('K-fold  cross validation results')
    plt.savefig('./crossvalidation.png')

def modelTrain(modelUsed,pretrained,trainDataset,trainLabels,validationDataset,validationLabels,numEpochs,evalInterval,evalMode,outDir,numWorkers,lr, stepSize, decayRate, trainBatchSize, seqLen,plotting):
    mean=[0.485, 0.456, 0.406]
    std=[0.229, 0.224, 0.225]
    normalize = Normalize(mean=mean, std=std)
    spatial_transform = Compose([Scale(256), RandomHorizontalFlip(), MultiScaleCornerCrop([1, 0.875, 0.75, 0.65625], 224),
                                 ToTensor(), normalize])

    vidSeqTrain = makeDataset(trainDataset, trainLabels, spatial_transform=spatial_transform,
                                seqLen=seqLen)
    # torch iterator to give data in batches of specified size
    trainLoader = torch.utils.data.DataLoader(vidSeqTrain, batch_size=trainBatchSize,
                            shuffle=True, num_workers=numWorkers, pin_memory=True, drop_last=True)

    if evalMode == 'centerCrop':
        test_spatial_transform = Compose([Scale(256), CenterCrop(224), ToTensor(), normalize])
    elif evalMode == 'tenCrops':
        test_spatial_transform = Compose([Scale(256), TenCrops(size=224, mean=mean, std=std)])
    elif evalMode == 'fiveCrops':
        test_spatial_transform = Compose([Scale(256), FiveCrops(size=224, mean=mean, std=std)])
    elif evalMode == 'horFlip':
        test_spatial_transform = Compose([Scale(256), CenterCrop(224), FlippedImagesTest(mean=mean, std=std)])
    
    vidSeqValid = makeDataset(validationDataset, validationLabels, seqLen=seqLen,
    spatial_transform=test_spatial_transform)


    validationLoader = torch.utils.data.DataLoader(vidSeqValid, batch_size=1,
                            shuffle=False, num_workers=int(numWorkers/2), pin_memory=True)

    numTrainInstances = vidSeqTrain.__len__()
    numValidationInstances = vidSeqValid.__len__()

    print('Number of training samples = {}'.format(numTrainInstances))
    print('Number of validation samples = {}'.format(numValidationInstances))

    modelFolder = './experiments_' + outDir+'_'+modelUsed+'_'+str(pretrained) # Dir for saving models and log files
    # Create the dir
    if os.path.exists(modelFolder):
        pass
    else:
        os.makedirs(modelFolder)
    # Log files
    writer = SummaryWriter(modelFolder)
    trainLogLoss = open((modelFolder + '/trainLogLoss.txt'), 'a')
    trainLogAcc = open((modelFolder + '/trainLogAcc.txt'), 'a')
    validationLogLoss = open((modelFolder + '/validLogLoss.txt'), 'a')
    validationLogAcc = open((modelFolder + '/validLogAcc.txt'), 'a')


    model = ViolenceModel(modelUsed,pretrained)


    trainParams = []
    for params in model.parameters():
        if params.requires_grad:
            trainParams += [params]
    model.train(True)
    if(torch.cuda.is_available()):
        model.cuda()

    lossFn = nn.CrossEntropyLoss()
    optimizerFn = torch.optim.RMSprop(trainParams, lr=lr)
    optimizerFn.zero_grad()
    optimScheduler = torch.optim.lr_scheduler.StepLR(optimizerFn, stepSize, decayRate)

    minAccuracy = 50
    train_loss=[]
    val_loss=[]
    train_acc=[]
    val_acc=[]
    bestmodel=None

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
            epochLoss += loss.item()
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
            validationLossEpoch = 0
            validationIter = 0
            numCorrTest = 0
            for j, (inputs, targets) in enumerate(validationLoader):
                validationIter += 1
                #if evalMode == 'centerCrop':
                if(torch.cuda.is_available()):
                    inputVariable1 = Variable(inputs.permute(1, 0, 2, 3, 4).cuda(), requires_grad=False)
                    labelVariable = Variable(targets.cuda(async=True), requires_grad=False)
                else:
                    inputVariable1 = Variable(inputs.permute(1, 0, 2, 3, 4), requires_grad=False)
                    labelVariable = Variable(targets, requires_grad=False)
                # else:
                #     if(torch.cuda.is_available()):
                #         inputVariable1 = Variable(inputs[0].permute(1, 0, 2, 3, 4).cuda(), requires_grad=False)
                #         labelVariable = Variable(targets.cuda(async=True), requires_grad=False)
                #     else:
                #         inputVariable1 = Variable(inputs[0].permute(1, 0, 2, 3, 4), requires_grad=False)
                #         labelVariable = Variable(targets, requires_grad=False)
                outputLabel = model(inputVariable1)
                validationLoss = lossFn(outputLabel, labelVariable)
                validationLossEpoch += validationLoss.item()
                outputProb = torch.nn.Softmax(dim=1)(outputLabel)
                _, predicted = torch.max(outputProb.data, 1)
                if(torch.cuda.is_available()):
                    numCorrTest += (predicted == targets[0].cuda()).sum()
                else:
                    numCorrTest += (predicted == targets[0]).sum()
            validationAccuracy = (float(numCorrTest) * 100)/float(numValidationInstances)
            avgValidationLoss = validationLossEpoch / validationIter
            val_loss.append(avgValidationLoss)
            val_acc.append(validationAccuracy)
            print('Testing: Loss = {} | Accuracy = {}% '.format(avgValidationLoss, validationAccuracy))
            writer.add_scalar('test/epochloss', avgValidationLoss, epoch + 1)
            writer.add_scalar('test/accuracy', validationAccuracy, epoch + 1)
            validationLogLoss.write('valid Loss after {} epochs = {}\n'.format(epoch + 1, avgValidationLoss))
            validationLogAcc.write('valid Accuracy after {} epochs = {}%\n'.format(epoch + 1, validationAccuracy))
            if validationAccuracy > minAccuracy:
                bestmodel=model
                minAccuracy = validationAccuracy
    '''plotting the accuracy and loss curves'''
    if plotting:
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
    validationLogAcc.close()
    trainLogLoss.close()
    validationLogLoss.close()
    writer.export_scalars_to_json(modelFolder + "/all_scalars.json")
    writer.close()
    return bestmodel,validationAccuracy 

def main_run(numEpochs, lr, stepSize, decayRate, trainBatchSize, seqLen,
             evalInterval, evalMode, numWorkers, outDir,modelUsed,pretrained,train_test_split,directory,crossValidation,folds):



    compDataset,classCount,class_names = make_split(directory)
    

    if crossValidation:
        data,label=compDataset
        kFoldCrossValid(folds,data,label,numEpochs,evalMode,numWorkers,lr, stepSize, decayRate, trainBatchSize, seqLen)

    else:
        (trainDataset,trainLabels),(validationDataset,validationLabels),(testDataset,testLabels)=sampleFromClass(compDataset,classCount,train_test_split)
        model,accuracy=modelTrain(modelUsed,pretrained,trainDataset,trainLabels,validationDataset,validationLabels,numEpochs,evalInterval,evalMode,outDir,numWorkers,lr, stepSize, decayRate, trainBatchSize, seqLen,True)
        '''for printing confusion matrix'''
        mean=[0.485, 0.456, 0.406]
        std=[0.229, 0.224, 0.225]
        normalize = Normalize(mean=mean, std=std)
        if evalMode == 'centerCrop':
            test_spatial_transform = Compose([Scale(256), CenterCrop(224), ToTensor(), normalize])
        elif evalMode == 'tenCrops':
            test_spatial_transform = Compose([Scale(256), TenCrops(size=224, mean=mean, std=std)])
        elif evalMode == 'fiveCrops':
            test_spatial_transform = Compose([Scale(256), FiveCrops(size=224, mean=mean, std=std)])
        elif evalMode == 'horFlip':
            test_spatial_transform = Compose([Scale(256), CenterCrop(224), FlippedImagesTest(mean=mean, std=std)])
        
        vidSeqTest = makeDataset(testDataset, testLabels, seqLen=seqLen,
        spatial_transform=test_spatial_transform)

        testLoader = torch.utils.data.DataLoader(vidSeqTest, batch_size=1,
                            shuffle=False, num_workers=int(numWorkers/2), pin_memory=True)

        
        numTestInstances = vidSeqTest.__len__()

        print('Number of test samples = {}'.format(numTestInstances))

        modelFolder = './experiments_' + outDir+'_'+modelUsed+'_'+str(pretrained) # Dir for saving models and log files
        
        
        savePathClassifier = (modelFolder + '/bestModel.pth')
        torch.save(model.state_dict(), savePathClassifier)
        '''running test samples and printing confusion matrix'''
        model.train(False)
        print('Testing...')
        LossEpoch = 0
        testIter = 0
        pred=None
        targ=None
        numCorrTest = 0
        for j, (inputs, targets) in enumerate(testLoader):
            testIter += 1
            #if evalMode == 'centerCrop':
            if(torch.cuda.is_available()):
                inputVariable1 = Variable(inputs.permute(1, 0, 2, 3, 4).cuda(), requires_grad=False)
                labelVariable = Variable(targets.cuda(async=True), requires_grad=False)
            else:
                inputVariable1 = Variable(inputs.permute(1, 0, 2, 3, 4), requires_grad=False)
                labelVariable = Variable(targets, requires_grad=False)
            # else:
            #     if(torch.cuda.is_available()):
            #         inputVariable1 = Variable(inputs[0].permute(1, 0, 2, 3, 4).cuda(), requires_grad=False)
            #         labelVariable = Variable(targets.cuda(async=True), requires_grad=False)
            #     else:
            #         inputVariable1 = Variable(inputs[0].permute(1, 0, 2, 3, 4), requires_grad=False)
            #         labelVariable = Variable(targets, requires_grad=False)
            outputLabel = model(inputVariable1)
            outputProb = torch.nn.Softmax(dim=1)(outputLabel)
            _, predicted = torch.max(outputProb.data, 1)
            if pred is None:
                pred=predicted.cpu().numpy()
                targ=targets[0].cpu().numpy()
            else:
                pred=np.append(pred,predicted.cpu().numpy())
                targ=np.append(targ,targets[0].cpu().numpy())
            # if(torch.cuda.is_available()):
            #     numCorrTest += (predicted == targets[0].cuda()).sum()
            # else:
            #     numCorrTest += (predicted == targets[0]).sum()
        
        # Compute confusion matrix
        cnf_matrix = confusion_matrix(targ, pred)
        np.set_printoptions(precision=2)
        # Plot non-normalized confusion matrix
        plt.figure()
        plot_confusion_matrix(cnf_matrix, classes=class_names,
                              title='Confusion matrix, without normalization')
        plt.savefig(modelFolder+"/no_norm_confusion_matrix.png")
        # Plot normalized confusion matrix
        plt.figure()
        plot_confusion_matrix(cnf_matrix, classes=class_names, normalize=True,
                              title='Normalized confusion matrix')

        plt.savefig(modelFolder+"/confusion_matrix.png")    
        return True

def __main__():
    parser = argparse.ArgumentParser()
    parser.add_argument('--numEpochs', type=int, default=1, help='Number of epochs')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--stepSize', type=int, default=25, help='Learning rate decay step')
    parser.add_argument('--decayRate', type=float, default=0.05, help='Learning rate decay rate')
    parser.add_argument('--seqLen', type=int, default=20, help='Length of sequence')
    parser.add_argument('--trainBatchSize', type=int, default=6, help='Training batch size')
    parser.add_argument('--evalInterval', type=int, default=5, help='Evaluation interval')
    parser.add_argument('--evalMode', type=str, default='centerCrop', help='Evaluation mode', choices=['centerCrop', 'horFlip', 'fiveCrops', 'tenCrops'])
    parser.add_argument('--numWorkers', type=int, default=20, help='Number of workers for dataloader')
    parser.add_argument('--outDir', type=str, default='violence', help='Output directory')
    parser.add_argument('--modelUsed', type=str, default='alexnet', help='CNN model to be used',choices=['vgg16','vgg16_bn','vgg19','vgg19_bn','alexnet','resnet50','resnet101'])
    parser.add_argument('--pretrained', type=bool, default=True, help='Use pretrained model')
    parser.add_argument('--train_test_split', type=float, default=0.6, help='train test validation split ratio')
    parser.add_argument('--datasetDir', type=str, default='./dataset', help='Input directory')
    parser.add_argument('--crossValidation', type=bool, default=False, help='Enable cross validation')
    parser.add_argument('--nFolds', type=int, default=5, help='number of folds for cross validation')
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
    train_test_split=args.train_test_split
    datasetDir=args.datasetDir
    crossValidation=args.crossValidation
    nFolds=args.nFolds
    main_run(numEpochs, lr, stepSize, decayRate, trainBatchSize, seqLen,
             evalInterval, evalMode, numWorkers, outDir,modelUsed,pretrained,train_test_split,datasetDir,crossValidation,nFolds)
if __name__=='__main__':
    __main__()
