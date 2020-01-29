#include <CL\cl.hpp>
#include <fstream>

#include "NeuralNetwork.h"
#include "TestOperations.h"
#include "BatchManager.h"

using namespace DeepCL;

int main(void)
{
	const int BATCH_SIZE = 100;

	// LeNet:


	NNSystem::NeuralNetwork nnTest;
	DeepCLError error = nnTest.InitSystem();
	if (error != 0) {
		std::cout << "Error encountered in Init System!" << std::endl << "Error Code: " << error;
		return 0;
	}
	// Parameters:
	//The input variables of the Neural Network
	//Input image
	NNBufferIdx i = nnTest.CreateInputBuffer(28, 28, 1);
	//Label
	NNBufferIdx l = nnTest.CreateInputBuffer(1);

	//The parameters of the Convolutional Neural Network
	NNBufferIdx wc1 = nnTest.CreateParameterBuffer(3, 3, 1, 32);
	NNBufferIdx bc1 = nnTest.CreateParameterBuffer(32);
	NNBufferIdx wc2 = nnTest.CreateParameterBuffer(3, 3, 32, 64);
	NNBufferIdx bc2 = nnTest.CreateParameterBuffer(64);

	//The parameters of the fully connected layers.
	//The complete output volume must be reduced to a fixed number of Neurons. 
	//This results in a relatively big matrix multiplication if the image is bigger.
	NNBufferIdx wf1 = nnTest.CreateParameterBuffer(1024, 7*7*64);
	NNBufferIdx bf1 = nnTest.CreateParameterBuffer(1024);
	NNBufferIdx wf2 = nnTest.CreateParameterBuffer(10, 1024);
	NNBufferIdx bf2 = nnTest.CreateParameterBuffer(10);
	
	// Modell:
	
	//First convolutional Layer with ReLU activation function
	NNBufferIdx hPreBias = OP::Conv2d(i, wc1, 1);
	NNBufferIdx hconv1 = OP::ReLU(OP::AddBiasConv(hPreBias, bc1));
	//First maxpooling
	NNBufferIdx hpool1 = OP::MaxPooling(hconv1, 0, 0, 2, 2);
	
	//Second convolutional Layer with ReLU activation function
	NNBufferIdx hPreBias2 = OP::Conv2d(hpool1, wc2, 1);
	NNBufferIdx hconv2Tmp = OP::AddBiasConv(hPreBias2, bc2);
	NNBufferIdx hconv2 = OP::ReLU(hconv2Tmp);
	//Second maxpooling
	NNBufferIdx hpool2 = OP::MaxPooling(hconv2, 0, 0, 2, 2);

	//First fully connected layer. The image must be flattened too an 1D array. This is done automatically in MultiplyFlattened.
	NNBufferIdx hf1tmp = OP::MultiplyFlattened(hpool2, wf1);
	NNBufferIdx hf1tmptmp = OP::AddBias(hf1tmp, bf1);
	NNBufferIdx hf1 = OP::ReLU(hf1tmptmp);

	//Second fully connected layer
	NNBufferIdx hf2tmp = OP::Multiply(hf1, wf2);
	NNBufferIdx hf2tmptmp = OP::AddBias(hf2tmp, bf2);
	NNBufferIdx hf2 = hf2tmptmp;

	//Softmax function
	NNBufferIdx soft = OP::Softmax(hf2);

	//Application of the Cross Entropy loss.
	NNBufferIdx loss = OP::CrossEntropy(soft, l);

	//Used for storing the model (Unnecessary for the current model)
	std::map<NNBufferIdx, char*> parameterBufferMap;
	//Add the index of the buffer and a name which should be used to store/load the model.
	parameterBufferMap.insert(std::pair<NNBufferIdx, char*>(wc1, "wc1"));
	parameterBufferMap.insert(std::pair<NNBufferIdx, char*>(wc2, "wc2"));
	parameterBufferMap.insert(std::pair<NNBufferIdx, char*>(wf1, "wf1"));
	parameterBufferMap.insert(std::pair<NNBufferIdx, char*>(wf1, "wf1"));

	parameterBufferMap.insert(std::pair<NNBufferIdx, char*>(bc1, "bc1"));
	parameterBufferMap.insert(std::pair<NNBufferIdx, char*>(bc1, "bc2"));

	parameterBufferMap.insert(std::pair<NNBufferIdx, char*>(bf1, "bf1"));
	parameterBufferMap.insert(std::pair<NNBufferIdx, char*>(bf1, "bf2"));

	
	//Create a loader for the MNIST dataset.
	//This must contain the relative or absolute path of the training files of the MNIST dataset.
	DataSystem::IDXReader idxReader("./train-images.idx3-ubyte", "./train-labels.idx1-ubyte", BATCH_SIZE);
	//Chck for an error during initalization.
	//If an error occured stop executing the program!
	if (!idxReader.Initalized())
		return -1;
	//Create a Transformer object for the MNIST dataset
	DataSystem::MNISTTransformer idxTransformer(BATCH_SIZE);
	//Create the BatchManager for the before created loader and transformer.
	DataSystem::BatchManager<2, int, float> batchManager(BATCH_SIZE, 1, 25, &idxReader, &idxTransformer);
	
	//Do the same to load test examples which will be used to evaluate the performance of the system.
	//This must contain the relative or absolute path of the test files of the MNIST dataset.
	DataSystem::IDXReader idxReader2("./t10k-images.idx3-ubyte", "./t10k-labels.idx1-ubyte", BATCH_SIZE);
	//Chck for an error during initalization.
	//If an error occured stop executing the program!
	if (!idxReader2.Initalized())
		return -1;
	DataSystem::MNISTTransformer idxTransformer2(BATCH_SIZE);
	DataSystem::BatchManager<2, int, float> testManager(BATCH_SIZE, 1, 5, &idxReader2, &idxTransformer2);
	
	//Use the Adam optimizer as optimizer.
	nnTest.AddOptimizer(new NNSystem::NNAdam(0.0001f*0.7f, 0.9f, 0.999f, 10e-8f));
	

	//Init the parameters using truncated normal distributions
	OP::InitWeightTruncatedNormalRnd(wc1, 0.0f, 0.1f);
	OP::InitWeightTruncatedNormalRnd(wc2, 0.0f, 0.1f);
	OP::InitWeightTruncatedNormalRnd(wf1, 0.0f, 0.1f);
	OP::InitWeightTruncatedNormalRnd(wf2, 0.0f, 0.1f);
	
	//Init the bias parameters with zero.
	OP::InitWeightUniform(bc1, 0.0f);
	OP::InitWeightUniform(bc2, 0.0f);
	OP::InitWeightUniform(bf1, 0.0f);
	OP::InitWeightUniform(bf2, 0.0f);
	
	//Initalize the graph.
	nnTest.InitliazeGraph(BATCH_SIZE);
	
	//Temporary variable for loading the softmax results of a batch into host memory.
	float* results = new float[10 * BATCH_SIZE];
	
	
	//List of buffers into which the data should be loaded.
	std::vector<NNBufferIdx> buffer;
	buffer.push_back(l);
	buffer.push_back(i);

	std::cout << "Start training" << std::endl;
	
	//Training should reach something around 99%
	for (int c = 0; c <= 40000; ++c)
	{
		//Query batch
		DataSystem::Batch<int, float>* batch = batchManager.GetBatch();
		
		//Perform the forward path using the data in the batch->data element.
		nnTest.Forward<int, float>(BackendSystem::get<0>(batch->data), BackendSystem::get<1>(batch->data), buffer, batch->sizes, batch->batchSize);
		//Perform the backward path calculating the gradients with respect to the parameters. Uses the results of the last call of forward.
		nnTest.Backward();

		//Perform parameter update step and perform clean up.
		nnTest.BatchDone();

		//Calculates the accuracy on a test set every 20 steps.
		if (c % 20 == 0 && c != 0)
		{

			size_t numTest = 1000;

			//At step 500 the model will be tested on 10000 examples.
			size_t pointOfFullTest = 500;
			if ((c % pointOfFullTest) == 0 && c != 0)
			{
				numTest = 10000;
			}

			size_t numCorrect = 0;

			//Perform all necessary steps.
			for (size_t j = 0; j < numTest; j += BATCH_SIZE)
			{
				//Read test batch.
				DataSystem::Batch<int, float>* testBatch = testManager.GetBatch();
				//Perform only the forward pass.
				nnTest.Forward<int, float>(BackendSystem::get<0>(testBatch->data), BackendSystem::get<1>(testBatch->data), buffer, testBatch->sizes, testBatch->batchSize);

				//Store the softmax results and the label.
				int* label = BackendSystem::get<0>(testBatch->data).data();
				nnTest.ReadDataBuffer(soft, results);

				//Calculate if the NN computed the right result.
				for (int i = 0; i < BATCH_SIZE; ++i)
				{
					int maxProp = 0;
					float currentProp = 0;
					for (int x = 0; x < 10; ++x)
					{
						if (currentProp < results[i * 10 + x])
						{
							maxProp = x;
							currentProp = results[i * 10 + x];
						}
					}

					//If the result of the NN is correct increase the number of correct examples by one.
					if (label[i] == maxProp)
						++numCorrect;
				}
			}

			//Calculate the accuracy by divinding the number of correct examples by the number of total examples.
			int realNum = ((numTest + BATCH_SIZE - 1) / BATCH_SIZE)*BATCH_SIZE;
			if (c % pointOfFullTest == 0 && c != 0)
			{
				std::cout << std::endl << "Test Correct(Full Test Batch): " << static_cast<float>(numCorrect) / static_cast<float>(realNum) << " at Step " << c << std::endl << std::endl;
			}
			else
				std::cout << "Test Correct: " << static_cast<float>(numCorrect) / static_cast<float>(realNum) << " at Step " << c << std::endl;
		}
	}

	delete[] results;

	system("PAUSE");
	return 0;
}