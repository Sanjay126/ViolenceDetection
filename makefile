numEpochs=10
Model='alexnet'
folds=5
input='./dataset'
get-project:
	make install-dependencies
	git clone https://github.com/SaNjAy-143-u/google_colab
	cd ./google_colab
install-dependencies: 
	#sudo apt-get install python3-pip python3-dev build-essential 
	pip3 install torch torchvision
	pip3 install tensorboardX
	pip3 install matplotlib sklearn numpy opencv-python
	echo "install CUDA for faster training \n instuctions: https://gist.github.com/zhanwenchen/e520767a409325d9961072f666815bb8 "
crossvalidate:
	python3 main-run-vr.py --numEpochs $(numEpochs) --crossValidation True --nFolds $(folds) --datasetDir $(input)

train:
	python3 main-run-vr.py --numEpochs $(numEpochs) --modelUsed $(Model) --datasetDir $(input)

