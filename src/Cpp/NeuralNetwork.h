#pragma once

#include <cmath>

#include "OPManager.h"

namespace DeepCL
{
	namespace NNSystem
	{
		class NeuralNetwork
		{
		public:
			NeuralNetwork();
			~NeuralNetwork();

			DeepCLError InitSystem();


			void AddWeightInitializer(InitOp* initOp);
			void AddOptimizer(NNOptimizer* optimizer);

			NNBufferIdx AddOperation(NNOp* operation, const size_t timeOffset);
			NNBufferIdx AddOperation(NNOp* operation, const NNBufferIdx result, const size_t timeOffset);

			//Create Buffer functions create add Buffer to the Graph
			//Returns the index of a Buffer used for Input
			NNBufferIdx CreateInputBuffer(const size_t sizeX, const size_t sizeY = 1, const size_t sizeZ = 1, const size_t sizeW = 1, const size_t timeSteps = 1);
			//Returns the index of a Buffer used for the State of a RNN
			NNBufferIdx CreateStateBuffer(const size_t sizeX, const size_t sizeY = 1, const size_t sizeZ = 1, const size_t sizeW = 1, const size_t timeSteps = 1);
			//Returns the index of a Buffer used for Parameters (Trained by the System)
			NNBufferIdx CreateParameterBuffer(const size_t sizeX, const size_t sizeY = 1, const size_t sizeZ = 1, const size_t sizeW = 1);

			//Write Data Buffer Functions directly write the Data in data into the specified buffer.
			template<typename T>
			void WriteDataBuffer(NNBufferIdx buffer, const T* data, const size_t sizeX, const size_t sizeY = 1, const size_t sizeZ = 1, const size_t sizeW = 1, const size_t offset = 0);
			template<typename T>
			void WriteDataBuffer(NNBufferIdx buffer, const T* data, const size_t sizeX, const size_t sizeY, const size_t sizeZ, const size_t sizeW, const size_t offset, const size_t numSubBuffer);
#ifdef _DEBUG
			void WriteDataBufferGrad(NNBufferIdx buffer, const void* data, const size_t sizeX, const size_t sizeY = 1, const size_t sizeZ = 1, const size_t sizeW = 1, const size_t offset = 0);
#endif
			//Saves the model to the File with the name fileName. The Buffers must be parameter buffers and they must be contained in the map. The map is used to map indices 
			//to names which will then be stored in the file. The names are also used to map the loaded parameters into the specific buffer object.
			void SaveModel(const std::string& fileName, const std::map<NNBufferIdx, char*>& names);
			void LoadModel(const std::string& fileName, const std::map<NNBufferIdx, char*>& names);

			//Read the content of the OpenCL buffer specified by buffer into host memory. The functions can be used to load different time steps or the gradient of the NNBuffer specified by buffer.
			void ReadDataBuffer(NNBufferIdx buffer, void* data, const size_t sizeX, const size_t sizeY, const size_t sizeZ = 1, const size_t sizeW = 1, const size_t offset = 0, const size_t = 0);
			void ReadDataBuffer(NNBufferIdx buffer, void* data, const size_t time = 0);

			void ReadDataBufferGrad(NNBufferIdx buffer, void* data, const size_t sizeX, const size_t sizeY, const size_t sizeZ = 1, const size_t sizeW = 1, const size_t offset = 0, const size_t = 0);
			void ReadDataBufferGrad(NNBufferIdx buffer, void* data, const size_t time = 0);

			//Functions similar to the Read functions but this time printing the buffer into the ostream stream. This allows the output of buffers into files and to the console
			template<typename T>
			void PrintBuffer(NNBufferIdx buffer, const char* msg, std::ostream& stream = std::cout, const size_t time = 0);
			template<typename T>
			void PrintBufferGrad(NNBufferIdx buffer, const char* msg, std::ostream& stream = std::cout, const size_t time = 0);

			template<typename T>
			void PrintBuffer(NNBufferIdx buffer, const char* msg, const size_t numImages, const size_t numBatches, const size_t imgOffset = 0, const size_t batchOffset = 0, std::ostream& stream = std::cout, const size_t time = 0);
			template<typename T>
			void PrintBufferGrad(NNBufferIdx buffer, const char* msg, const size_t numImages, const size_t numBatches, const size_t imgOffset = 0, const size_t batchOffset = 0, std::ostream& stream = std::cout, const size_t time = 0);

			template<typename T>
			void ReadBufferFile(NNBufferIdx buffer, const char* name, std::fstream& file);
			template<typename T>
			void WriteBufferBinFile(NNBufferIdx buffer, const char* name, std::fstream& file);

			template<typename T>
			void PrintMomentumBuffer(NNBufferIdx buffer, const char* msg, const size_t num, const size_t numImages, const size_t numBatches, const size_t imgOffset = 0, const size_t batchOffset = 0, std::ostream& stream = std::cout);

			template<typename T>
			void PrintMomentumBuffer(NNBufferIdx buffer, const char* msg, const size_t num, std::ostream& stream = std::cout);

			//Function Initalizes Graph. It must be called before the training can be performed and after the model was completely created.
			//It creates all OpenCL objects via the backend. Before no OpenCL objects where created.
			DeepCLError InitliazeGraph(const size_t batchSize);

			//Forward functions allowing different types and number of inputs to be used. It executes the forward pass of the NN.
			template<typename T1, typename T2>
			DeepCLError Forward(T1* input, NNBufferIdx inputBuffer, T2* output, NNBufferIdx outputBuffer, const SizeVec& inputSize, const SizeVec& outputSize, const size_t curBatchSize);

			template<typename T1, typename T2>
			DeepCLError Forward(T1* input, NNBufferIdx inputBuffer, T2* output, NNBufferIdx outputBuffer, const SizeVec& inputSize, const SizeVec& outputSize, const size_t curBatchSize, const size_t numTimeSteps);

			//Allows a variable number of inputs using Variadic Templates. The Buffer indices specify the NNInputBuffer in which the corresponding buffer data is loaded into. 
			//The bufferIndice vector must therefore have the same number of elements as there are template arguments.
			template<typename... T1>
			DeepCLError Forward(std::vector<T1>&... buffer, std::vector<NNBufferIdx>& bufferIndices, std::vector<SizeVec>& sizes, const size_t curBatchSize)
			{
				if (!graphInitiliazed)
				{
					std::cout << "Error command queue was not build!" << std::endl;
					return NN_GRAPH_NOT_INITIALIZED;
				}

				UnrollFwd<0, T1...>::apply(buffer..., bufferIndices, sizes, curBatchSize, *this);

				backend.Run(BackendSystem::OpenCLBackend::OperationType::FORWARD);

				return 0;
			}



			DeepCLError Forward();

			//Performs the backward pass of the Nn
			DeepCLError Backward();

			//Updates Parameters and sets all gradients to zero.
			DeepCLError BatchDone();

			//template<typename T1, typename T2>
			//float* CalculateGradError(const NNBufferIdx a, const float epsilon, const NNBufferIdx error, const Batch<T1, T2>& batch, const size_t timeStep, float** gradBuffer, const size_t sX = 0, const size_t sY = 0, const size_t sZ = 0, const size_t sW = 0);

			SizeVec GetSize(const NNBufferIdx a) const;

#ifdef PROFILING_ENABLED
			unsigned long long GetTimeForward(const NNBufferIdx input, const NNBufferIdx output);
			unsigned long long GetTimeBackward(const NNBufferIdx input, const NNBufferIdx output);
#endif // PROFILING_ENABLED

			//Function applied on tuple elements
			template<typename T1>
			void UnrollForward(std::vector<T1>& curBuffer, std::vector<NNBufferIdx>& bufferIndices, std::vector<SizeVec>& sizes, const size_t curBatchSize, size_t i)
			{
				SizeVec tmpSize = sizes[i];
				size_t timeSteps = 1;

				if (tmpSize.sizeW > 1)
					timeSteps = tmpSize.sizeW;

				tmpSize.sizeW = curBatchSize;

				if (timeSteps > 1)
					WriteDataBuffer<T1>(bufferIndices[i], curBuffer.data(), tmpSize.sizeX, tmpSize.sizeY, tmpSize.sizeZ, tmpSize.sizeW, 0, timeSteps);
				else
					WriteDataBuffer<T1>(bufferIndices[i], curBuffer.data(), tmpSize.sizeX, tmpSize.sizeY, tmpSize.sizeZ, tmpSize.sizeW);
			}

		private:
			BackendSystem::OpenCLBackend backend;

			std::vector<NNBuffer*> nnBufferList; //NNBuffers contained in the Graph
			std::vector<NNOp*> nnOperationList;  //NNOps contained in the Graph
			std::vector<NNBufferIdx> parameterBuffer; //ParameterBuffer contained in the Graph(Intersects with nnBufferList)
			std::vector<InitOp*> initOpList; //InitOps used to initalize the parameters

			NNOptimizer* optimizer;
			size_t numAuxBuffer; //Attitional Buffers of the optimizer(Momentum etc.)
			size_t maxSteps; //Maximal number of time steps the RNN runs.

			static NeuralNetwork* activeNN;

			bool initialized; //True when InitSystem was called
			bool graphInitiliazed;//True when the Graph was initalized

			float* tmpDataMemory;
			size_t maxSize;

			NNBufferIdx CreateBuffer(const size_t sizeX, const size_t sizeY = 1, const size_t sizeZ = 1, const size_t sizeW = 1, const size_t timeSteps = 1);
			NNBufferIdx CreateBuffer(const SizeVec size, const size_t timeSteps = 1);

			void NeuralNetwork::ReadDataBufferDirect(BufferIdx buffer, void* data, const size_t totalSize, const size_t offset);

			//Basic buffer for prinitng buffers
			template<typename T>
			void PrintBufferBase(const char* msg, const size_t numImages, const size_t numBatches, const size_t imgOffset, const size_t batchOffset, const SizeVec& size, std::ostream& stream)
			{
				stream << msg << std::endl;

				for (size_t i = batchOffset; i < numBatches + batchOffset; ++i)
				{
					for (size_t l = imgOffset; l < numImages + imgOffset; ++l)
					{
						for (size_t j = 0; j < size.sizeY; ++j)
						{
							for (size_t k = 0; k < size.sizeX; ++k)
							{
								stream << reinterpret_cast<T*>(tmpDataMemory)[k + size.sizeX * (j + size.sizeY * (l + size.sizeZ * i))] << " ";
							}
							stream << std::endl;
						}
						stream << std::endl;
					}
					stream << std::endl;
				}
				stream << std::endl;
			}

			//Sets the w component in each buffer to the batch size where it is necessary (Not in parameter buffers since they are independent of the batch size.)
			void SetBatchSize(const size_t batchSize);
			//Creates the acutal OpenCL hardware buffers.
			void InstantiateBuffer();
			//Calcualtes the number and size of necessary temporary buffers and creates them. Than each temporary buffer is added to operations which need them.
			void CreateTmpBuffer();

			//Creates and adds the implemented operations to the backend. The operations are added to the correct pass in the backend.
			//It also takes care of the unrolling of RNNs in time.
			void InstantiateOperations();
			//Perform the weight initalization operations which create some values for the parameters and transfer them to the backend.
			void InitalizeWeights();

			//Updates the timeStep variable in the buffers by one if necessary
			void UpdateBufferTime();
			//sets the timeStep variable in the buffers to zero.
			void ResetBufferTime();

			//Set all backward openCL hardware buffers to zero.
			void ClearBackwardBuffer();
			//Set all openCL hardware buffers to zero.
			void ClearAllBuffer();
		};

		// Functions necessary to apply transfer the data into the input buffer when variadic templates are used to insert add the data to the forward pass.
		//Terminal/base case
		template<size_t from, class... Ts>
		struct UnrollFwd{
		public:
			inline static void apply(std::vector<Ts>&... buffer, std::vector<NNBufferIdx>& bufferIndices, std::vector<SizeVec>& sizes, const size_t curBatchSizee, NeuralNetwork& nn)
			{
			}
		};
		//Splits of one element of the variadic template and transfers the data to the into the input buffer.
		template<size_t from, class T1, class... Ts>
		struct UnrollFwd < from, T1, Ts... >
		{
		public:
			inline static void apply(std::vector<T1>& curBuffer, std::vector<Ts>&... buffer, std::vector<NNBufferIdx>& bufferIndices, std::vector<SizeVec>& sizes, const size_t curBatchSize, NeuralNetwork& nn)
			{
				nn.UnrollForward<T1>(curBuffer, bufferIndices, sizes, curBatchSize, from);
				UnrollFwd<from + 1, Ts...>::apply(buffer..., bufferIndices, sizes, curBatchSize, nn);
			}
		};

		//Reads a buffer with name name out of the file file and stores it into buffer.
		template<typename T>
		void NeuralNetwork::ReadBufferFile(NNBufferIdx buffer, const char* name, std::fstream& file)
		{
			//Search for the beginning of the buffer with the name name.
			std::string line;
			bool found = false;
			while (getline(file, line))
			{
				if (line.find(name)!=std::string::npos)
				{
					found = true;
					break;
				}
			}
			//Error if a buffer with the name was not found.
			if (!found)
			{
				std::cerr << "name: " << name << " not found in file" << std::endl;
				return;
			}

			//Read the buffer out of the file and transfer it into the OpenCL buffer buffer.
			NNBuffer* nnBuffer = nnBufferList[buffer];
			SizeVec bufferSizes = nnBuffer->size;

			float* values = new float[bufferSizes.sizeX * bufferSizes.sizeY * bufferSizes.sizeZ * bufferSizes.sizeW];

			file.read(reinterpret_cast<char*>(values), bufferSizes.sizeX * bufferSizes.sizeY * bufferSizes.sizeZ * bufferSizes.sizeW * sizeof(float));

			WriteDataBuffer<float>(buffer, values, bufferSizes.sizeX, bufferSizes.sizeY, bufferSizes.sizeZ, bufferSizes.sizeW);
			delete[] values;
		}

		//Writes the contents of the forward part of buffer into the file name under the specified name name.
		template<typename T>
		void NeuralNetwork::WriteBufferBinFile(NNBufferIdx buffer, const char* name, std::fstream& file)
		{
			//Read the buffer content.
			SizeVec size = nnBufferList[buffer]->size;
			ReadDataBuffer(buffer, reinterpret_cast<void*>(tmpDataMemory), 0);
			//Write the name into the file.
			file <<  name << std::endl;

			NNBuffer* nnBuffer = nnBufferList[buffer];
			SizeVec bufferSizes = nnBuffer->size;

			char* dataPointer = reinterpret_cast<char*>(tmpDataMemory);
			//Store the read buffer data into the file.
			file.write(dataPointer, sizeof(T) * bufferSizes.sizeX* bufferSizes.sizeY* bufferSizes.sizeZ* bufferSizes.sizeW);

			file << std::endl;
		}

		//Call the forward pass using two input buffers.
		template<typename T1, typename T2>
		DeepCLError NeuralNetwork::Forward(T1* input, NNBufferIdx inputBuffer, T2* output, NNBufferIdx outputBuffer, const SizeVec& inputSize, const SizeVec& outputSize, const size_t curBatchSize)
		{
			if (!graphInitiliazed)
			{
				std::cout << "Error command queue was not build!" << std::endl;
				return NN_GRAPH_NOT_INITIALIZED;
			}

			//Write the input into the inputBuffer
			SizeVec tmpSize = inputSize;
			if (tmpSize.sizeW == 1)
				tmpSize.sizeW = curBatchSize;

			WriteDataBuffer<T1>(inputBuffer, input, tmpSize.sizeX, tmpSize.sizeY, tmpSize.sizeZ, tmpSize.sizeW);

			//Write the output(label) into the outputBuffer.
			tmpSize = outputSize;
			if (tmpSize.sizeW == 1)
				tmpSize.sizeW = curBatchSize;

			WriteDataBuffer<T2>(outputBuffer, output, tmpSize.sizeX, tmpSize.sizeY, tmpSize.sizeZ, tmpSize.sizeW);
			
			//Perform the forward pass.
			backend.Run(BackendSystem::OpenCLBackend::OperationType::FORWARD);


			return 0;
		}

		//Call the forward pass using two input buffers. The label no contains results for the specified number of time steps.
		template<typename T1, typename T2>
		DeepCLError NeuralNetwork::Forward(T1* input, NNBufferIdx inputBuffer, T2* output, NNBufferIdx outputBuffer, const SizeVec& inputSize, const SizeVec& outputSize, const size_t curBatchSize, const size_t numTimeSteps)
		{
			if (!graphInitiliazed)
			{
				std::cout << "Error command queue was not build!" << std::endl;
				return NN_GRAPH_NOT_INITIALIZED;
			}

			SizeVec tmpSize = inputSize;
			if (tmpSize.sizeW == 1)
				tmpSize.sizeW = curBatchSize;

			WriteDataBuffer<T1>(inputBuffer, input, tmpSize.sizeX, tmpSize.sizeY, tmpSize.sizeZ, tmpSize.sizeW);


			tmpSize = outputSize;
			if (tmpSize.sizeW == 1)
				tmpSize.sizeW = curBatchSize;

			WriteDataBuffer<T2>(outputBuffer, output, tmpSize.sizeX, tmpSize.sizeY, tmpSize.sizeZ, tmpSize.sizeW, numTimeSteps);

			backend.Run(BackendSystem::OpenCLBackend::OperationType::FORWARD);


			return 0;
		}

		//Write some data into the by buffer specified forward buffer.
		template<typename T>
		void NeuralNetwork::WriteDataBuffer(NNBufferIdx buffer, const T* data, const size_t sizeX, const size_t sizeY, const size_t sizeZ, const size_t sizeW, const size_t offset)
		{
			//Query the buffer and check if the sizes match.
			size_t totalSize = sizeW * sizeZ * sizeY * sizeX;
			NNBuffer* bufferData = nnBufferList[buffer];
			size_t bufferSize = bufferData->size.sizeW * bufferData->size.sizeZ * bufferData->size.sizeY * bufferData->size.sizeX;
			if (totalSize + offset > bufferSize)
			{
				std::cout << "Error WriteDataBuffer: Out of Range" << std::endl;
				return;
			}

			//Load the data into the corresponding buffer.
			BufferIdx fwdBuffer = bufferData->ForwardBuffer();

			backend.WriteDataBuffer(bufferData->ForwardBuffer(), data, offset, totalSize * sizeof(T));
		}

		//Writes data into multiple time steps of the forward buffer of buffer.
		//Data must contain the data for the different time steps behind each other.
		template<typename T>
		void NeuralNetwork::WriteDataBuffer(NNBufferIdx buffer, const T* data, const size_t sizeX, const size_t sizeY, const size_t sizeZ, const size_t sizeW, const size_t offset, const size_t numSubBuffer)
		{
			//Query the buffer and check if the sizes match.
			size_t totalSize = sizeW * sizeZ * sizeY * sizeX;
			NNBuffer* bufferData = nnBufferList[buffer];
			size_t bufferSize = bufferData->size.sizeW * bufferData->size.sizeZ * bufferData->size.sizeY * bufferData->size.sizeX;

			if (totalSize + offset > bufferSize)
			{
				std::cout << "Error WriteDataBuffer: Out of Range" << std::endl;
				return;
			}
			//Load the data into the different corresponding buffers.
			totalSize = sizeX * sizeY * sizeZ;
			T* tmpData = new T[totalSize * sizeW];
			std::vector<BufferIdx> localSubBuffer = bufferData->GetCompleteForwardBuffer();
			const T* realPointer = reinterpret_cast<const T*>(data);
			for (size_t i = 0; i < numSubBuffer; ++i)
			{
				for (size_t j = 0; j < sizeW; ++j)
					for (size_t k = 0; k < totalSize; ++k)
					{
						tmpData[j * totalSize + k] = realPointer[k + i * totalSize + j * totalSize * numSubBuffer];
					}
				backend.WriteDataBuffer(localSubBuffer[i], tmpData, offset, totalSize *sizeW* sizeof(T));
			}
			delete[] tmpData;
		}
		
		//Checks the gradient by calculating the numerical gradient and comparing the results.
		//After a change to the framework an error occurs sometimes which must be fixed before this can be used relably again.
		/*
		template <typename T1, typename T2>
		float* NeuralNetwork::CalculateGradError(const NNBufferIdx a, const float epsilon, const NNBufferIdx error, const Batch<T1, T2>& batch, const size_t timeStep, float** gradBuffer, const size_t sX, const size_t sY, const size_t sZ, const size_t sW)
		{
			NNBuffer buffer = nnBufferList[a];
			NNBuffer errorBuffer = nnBufferList[error];

			size_t size =  buffer.size.sizeX * buffer.size.sizeY * buffer.size.sizeZ * buffer.size.sizeW;
			size_t chosenSize = (sX != 0 ? sX : buffer.size.sizeX) * (sY != 0 ? sY : buffer.size.sizeY) * (sZ != 0 ? sZ : buffer.size.sizeZ) * (sW != 0 ? sW : buffer.size.sizeW);

			size_t errorSize = errorBuffer.size.sizeX * errorBuffer.size.sizeY * errorBuffer.size.sizeZ * errorBuffer.size.sizeW;

			float* errorBufferFloat = new float[errorSize];

			float* orgBuffer = new float[size];
			float* changedBuffer = new float[size];

			float* resultBuffer = new float[size];

			ReadDataBuffer(a, orgBuffer);

			size_t i, j, k, l, sum;

			float sumarg;

			for (size_t point = 0; point < chosenSize; ++point)
			{
				for (i = 0; i < buffer.size.sizeW; ++i)
					for (j = 0; j < buffer.size.sizeZ; ++j)
						for (k = 0; k < buffer.size.sizeY; ++k)
							for (l = 0; l < buffer.size.sizeX; ++l)
							{
								changedBuffer[l + buffer.size.sizeX * (k + buffer.size.sizeY * (j + buffer.size.sizeZ * i))] = orgBuffer[l + buffer.size.sizeX * (k + buffer.size.sizeY * (j + buffer.size.sizeZ * i))] + ((l + buffer.size.sizeX * (k + buffer.size.sizeY * (j + buffer.size.sizeZ * i))) == point ? epsilon : 0);
							}

				

				WriteDataBuffer(a, changedBuffer, buffer.size.sizeX, buffer.size.sizeY, buffer.size.sizeZ, buffer.size.sizeW);

				Forward<T1, T2>(batch);
				ReadDataBuffer(error, errorBufferFloat, timeStep);


				sumarg = 0.f;
				for (sum = 0; sum < errorBuffer.size.sizeW; ++sum)
					sumarg += errorBufferFloat[sum];
				resultBuffer[point] = sumarg / static_cast<float>(errorBuffer.size.sizeW);

				for (i = 0; i < buffer.size.sizeW; ++i)
					for (j = 0; j < buffer.size.sizeZ; ++j)
						for (k = 0; k < buffer.size.sizeY; ++k)
							for (l = 0; l < buffer.size.sizeX; ++l)
							{
								changedBuffer[l + buffer.size.sizeX * (k + buffer.size.sizeY * (j + buffer.size.sizeZ * i))] = orgBuffer[l + buffer.size.sizeX * (k + buffer.size.sizeY * (j + buffer.size.sizeZ * i))] - ((l + buffer.size.sizeX * (k + buffer.size.sizeY * (j + buffer.size.sizeZ * i))) == point ? epsilon : 0);
							}

				WriteDataBuffer(a, changedBuffer, buffer.size.sizeX, buffer.size.sizeY, buffer.size.sizeZ, buffer.size.sizeW);

				Forward<T1, T2>(batch);
				ReadDataBuffer(error, errorBufferFloat, timeStep);

				sumarg = 0.f;
				for (sum = 0; sum < errorBuffer.size.sizeW; ++sum)
					sumarg += errorBufferFloat[sum];
				resultBuffer[point] -= sumarg / static_cast<float>(errorBuffer.size.sizeW);


				if(point%100==0)
					std::cout << point << " ";
			}

			WriteDataBuffer(a, orgBuffer, buffer.size.sizeX, buffer.size.sizeY, buffer.size.sizeZ, buffer.size.sizeW);

			Forward<T1, T2>(batch);
			Backward();

			ReadDataBufferGrad(a, changedBuffer);



			for (i = 0; i < buffer.size.sizeW; ++i)
				for (j = 0; j < buffer.size.sizeZ; ++j)
					for (k = 0; k < buffer.size.sizeY; ++k)
						for (l = 0; l < buffer.size.sizeX; ++l)
						{
							resultBuffer[l + buffer.size.sizeX * (k + buffer.size.sizeY * (j + buffer.size.sizeZ * i))] = resultBuffer[l + buffer.size.sizeX * (k + buffer.size.sizeY * (j + buffer.size.sizeZ * i))] / (2.f*epsilon);
						}

			for (i = 0; i < buffer.size.sizeW; ++i)
				for (j = 0; j < buffer.size.sizeZ; ++j)
					for (k = 0; k < buffer.size.sizeY; ++k)
						for (l = 0; l < buffer.size.sizeX; ++l)
						{
							resultBuffer[l + buffer.size.sizeX * (k + buffer.size.sizeY * (j + buffer.size.sizeZ * i))] = abs(changedBuffer[l + buffer.size.sizeX * (k + buffer.size.sizeY * (j + buffer.size.sizeZ * i))] - resultBuffer[l + buffer.size.sizeX * (k + buffer.size.sizeY * (j + buffer.size.sizeZ * i))]) / max(abs(changedBuffer[l + buffer.size.sizeX * (k + buffer.size.sizeY * (j + buffer.size.sizeZ * i))]), abs(resultBuffer[l + buffer.size.sizeX * (k + buffer.size.sizeY * (j + buffer.size.sizeZ * i))]));
						}

			delete[] errorBufferFloat;
			delete[] orgBuffer;

			*gradBuffer = changedBuffer;

			std::cout << std::endl;

			return resultBuffer;
		}
		*/

		//Functions for printing buffers like the normal buffers for different time steps using time, the Momentum buffers, gradient buffers an so on.
		//All functions read the data and then print it using a basic print function called printBufferBase.
		template<typename T>
		void NeuralNetwork::PrintBuffer(NNBufferIdx buffer, const char* msg, std::ostream& stream, const size_t time)
		{
			SizeVec size = nnBufferList[buffer]->size;
			ReadDataBuffer(buffer, reinterpret_cast<void*>(tmpDataMemory), time);

			PrintBufferBase<T>(msg, size.sizeZ, size.sizeW, 0, 0, size, stream);
		}

		template<typename T>
		void NeuralNetwork::PrintMomentumBuffer(NNBufferIdx buffer, const char* msg, const size_t num, std::ostream& stream)
		{
			SizeVec size = nnBufferList[buffer]->size;
			NNBuffer* locBuffer= reinterpret_cast<NNBuffer*>(nnBufferList[buffer]);
			size_t totalSize = size.sizeX * size.sizeY * size.sizeZ * size.sizeW;

			ReadDataBufferDirect(locBuffer->ForwardBuffer(num), reinterpret_cast<void*>(tmpDataMemory), totalSize, 0);

			PrintBufferBase<T>(msg, size.sizeZ, size.sizeW, 0, 0, size, stream);
		}

		template<typename T>
		void NeuralNetwork::PrintMomentumBuffer(NNBufferIdx buffer, const char* msg, const size_t num, const size_t numImages, const size_t numBatches, const size_t imgOffset, const size_t batchOffset, std::ostream& stream)
		{
			SizeVec size = nnBufferList[buffer]->size;
			NNBuffer* locBuffer = reinterpret_cast<NNBuffer*>(nnBufferList[buffer]);
			size_t totalSize = size.sizeX * size.sizeY * size.sizeZ * size.sizeW;

			ReadDataBufferDirect(locBuffer->ForwardBuffer(num), reinterpret_cast<void*>(tmpDataMemory), totalSize, 0);

			PrintBufferBase<T>(msg, numImages, numBatches, imgOffset, batchOffset, size, stream);
		}

		template<typename T>
		void NeuralNetwork::PrintBufferGrad(NNBufferIdx buffer, const char* msg, std::ostream& stream, const size_t time)
		{
			SizeVec size = nnBufferList[buffer]->size;
			ReadDataBufferGrad(buffer, reinterpret_cast<void*>(tmpDataMemory), time);

			PrintBufferBase<T>(msg, size.sizeZ, size.sizeW, 0, 0, size, stream);
		}

		template<typename T>
		void NeuralNetwork::PrintBuffer(NNBufferIdx buffer, const char* msg, const size_t numImages, const size_t numBatches, const size_t imgOffset, const size_t batchOffset, std::ostream& stream, const size_t time)
		{
			SizeVec size = nnBufferList[buffer]->size;
			ReadDataBuffer(buffer, reinterpret_cast<void*>(tmpDataMemory), time);

			PrintBufferBase<T>(msg, numImages, numBatches, imgOffset, batchOffset, size, stream);
		}

		template<typename T>
		void NeuralNetwork::PrintBufferGrad(NNBufferIdx buffer, const char* msg, const size_t numImages, const size_t numBatches, const size_t imgOffset, const size_t batchOffset, std::ostream& stream, const size_t time)
		{
			SizeVec size = nnBufferList[buffer]->size;
			ReadDataBufferGrad(buffer, reinterpret_cast<void*>(tmpDataMemory), time);

			PrintBufferBase<T>(msg, numImages, numBatches, imgOffset, batchOffset, size, stream);
		}
	}
}

