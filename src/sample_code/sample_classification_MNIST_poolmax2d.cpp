// sample MNIST classification with a poolmax2D
// 96% accuracy after 20 epochs, 1s/epochs

#include <iostream>
#include <chrono>
using namespace std;

#include "Net.h"
#include "NetTrain.h"
#include "MNISTReader.h"
#include "ConfusionMatrix.h"

Net net;
NetTrain netTrain;
MatrixFloat mRefImages, mRefLabels, mTestImages, mTestLabels;
int iEpoch;
chrono::steady_clock::time_point start;

//////////////////////////////////////////////////////////////////////////////
void epoch_callback()
{
	//compute epoch time
	chrono::steady_clock::time_point next = chrono::steady_clock::now();
	auto delta = chrono::duration_cast<std::chrono::milliseconds>(next - start).count();
	start = next;

    iEpoch++;
    cout << "Epoch: " << iEpoch << " duration: " << delta << " ms" << endl;
	cout << "TrainLoss: " << netTrain.get_current_train_loss() << " TrainAccuracy: " << netTrain.get_current_train_accuracy() << " %" ;
	cout << " TestAccuracy: " << netTrain.get_current_test_accuracy() << " %" << endl;

	cout << endl;
}
//////////////////////////////////////////////////////////////////////////////
int main()
{
    iEpoch = 0;

	//load and normalize MNIST data
    cout << "Loading MNIST database..." << endl;
    MNISTReader mr;
    if(!mr.read_from_folder(".",mRefImages,mRefLabels, mTestImages,mTestLabels))
    {
        cout << "MNIST samples not found, please check the *.ubyte files are in the executable folder" << endl;
        return -1;
    }
	mTestImages/= 256.f;
	mRefImages/= 256.f;
  
	//create simple net:
	net.add_poolmax2D_layer(28,28,1, 2, 2); //input rows, input cols,input nbplanes, factor rows, factor cols
	net.add_dense_layer(784/4, 32); // new size is 4x smaller
	net.add_activation_layer("Relu");
	net.add_dense_layer(32, 10);
	net.add_softmax_layer();

	//setup train options
	netTrain.set_net(net);
	netTrain.set_epochs(20);
	netTrain.set_batchsize(64);
	netTrain.set_loss("SparseCategoricalCrossEntropy");
	netTrain.set_epoch_callback(epoch_callback); //optional, show progress
	netTrain.set_train_data(mRefImages, mRefLabels);
	netTrain.set_test_data(mTestImages, mTestLabels); //optional, not used for training, helps to keep the final best model

	// train net
	cout << "Training..." << endl << endl;
	start = chrono::steady_clock::now();
	netTrain.train();

	// show train results
	MatrixFloat mClassPredicted;
	net.classify(mRefImages, mClassPredicted);
	ConfusionMatrix cmRef;
	ClassificationResult crRef = cmRef.compute(mRefLabels, mClassPredicted);
	cout << "Ref accuracy: " << crRef.accuracy << " %" << endl;

	MatrixFloat mClassTest;
	net.classify(mTestImages, mClassTest);
	ConfusionMatrix cmTest;
	ClassificationResult crTest = cmTest.compute(mTestLabels, mClassTest);
	cout << "Test accuracy: " << crTest.accuracy << " %" << endl;
	cout << "Test confusion matrix:" << endl << crTest.mConfMat << endl;
	
	//testu function
	if (crTest.accuracy < 96.f)
	{
		cout << "Test failed! accuracy=" << crTest.accuracy << endl;
		return -1;
	}


	cout << "Test succeded." << endl;
    return 0;
}