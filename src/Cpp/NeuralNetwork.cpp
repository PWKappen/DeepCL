#include "NeuralNetwork.h"

#include <fstream>

namespace DeepCL
{
	namespace NNSystem
	{
		typedef NNOp NNOperation;

		NeuralNetwork* NeuralNetwork::activeNN = nullptr;

		NeuralNetwork::NeuralNetwork() :nnOperationList(), parameterBuffer(), initialized(false), graphInitiliazed(false), tmpDataMemory(nullptr), maxSize(0), optimizer(nullptr), numAuxBuffer(0),
			nnBufferList(), maxSteps(1)
		{
			if (activeNN == nullptr)
			{
				activeNN = this;
				OPManager::SetActiveNN(this);
			}
		}

		NeuralNetwork::~NeuralNetwork()
		{
			size_t size = nnOperationList.size();
			size_t i;
			for (i = 0; i < size; ++i)
				delete nnOperationList[i];
			size = nnBufferList.size();
			for (i = 0; i < size; ++i)
				delete nnBufferList[i];
			if (tmpDataMemory != nullptr)
				delete tmpDataMemory;
			if (optimizer != nullptr)
				delete optimizer;
			size = initOpList.size();
			for (i = 0; i < size; ++i)
				delete initOpList[i];
		}

		DeepCLError NeuralNetwork::InitSystem()
		{
			//Device Creation, etc...
			DeepCLError err;
			err = backend.InitGPU();
			if (err != 0)
			{
				return err;
			}

			//Loade the different Kernel Files specified in KernelConfig
			//This function call must contain the absolute or relativ path of the KernelConfig.txt file and the folder 
			//which contains all kernels. In this case the KernelConfig file is contained in the kernel folder which contains all kernel files.
			err = backend.LoadKernelFromConfig("./Kernel/KernelConfig.txt", "./Kernel/");
			if (err != 0)
			{
				return err;
			}
		
			initialized = true;

			return 0;
		}



		NNBufferIdx NeuralNetwork::AddOperation(NNOp* operation, const size_t timeOffset)
		{
			NNBufferIdx c = CreateBuffer(operation->GetOutputType(nnBufferList), operation->GetTimeTransform(nnBufferList));

			AddOperation(operation, c, timeOffset);

			return c;
		}

		//Adds operation to the graph, given a target buffer for the result of the operation
		NNBufferIdx NeuralNetwork::AddOperation(NNOp* operation, const NNBufferIdx result, const size_t timeOffset)
		{
			NNBufferIdx c = result;

			operation->timeOffset = timeOffset;
			operation->output.push_back(c);
			operation->SetBuffer(nnBufferList, nnOperationList.size());

			nnBufferList[result]->timeOffset = timeOffset;

			nnOperationList.push_back(operation);

			return c;
		}

		void NeuralNetwork::AddOptimizer(NNOptimizer* optimizer)
		{
			this->optimizer = optimizer;
		}

		NNBufferIdx NeuralNetwork::CreateInputBuffer(const size_t sizeX, const size_t sizeY, const size_t sizeZ, const size_t sizeW, const size_t timeSteps)
		{
			//Creates a NNInputBuffer object used for inserting data into the graph
			nnBufferList.push_back(new NNInputBuffer(sizeX, sizeY, sizeZ, sizeW, timeSteps));

			//The maximal number the NN must be unrolled is set to the highest number of time steps necessary
			if (timeSteps > maxSteps)
				maxSteps = timeSteps;

			NNBufferIdx idx = nnBufferList.size() - 1;
			return idx;
		}

		
		NNBufferIdx NeuralNetwork::CreateStateBuffer(const size_t sizeX, const size_t sizeY, const size_t sizeZ, const size_t sizeW, const size_t timeSteps)
		{
			//Creates a state buffer used for storing the state of an RNN Cell
			nnBufferList.push_back(new NNStateBuffer(sizeX, sizeY, sizeZ, sizeW, timeSteps));

			if (timeSteps > maxSteps)
				maxSteps = timeSteps;

			NNBufferIdx idx = nnBufferList.size() - 1;
			return idx;
		}

		
		NNBufferIdx NeuralNetwork::CreateParameterBuffer(const size_t sizeX, const size_t sizeY, const size_t sizeZ, const size_t sizeW)
		{
			//Creates a buffer to store the trainable parameters
			nnBufferList.push_back(new NNParamBuffer(sizeX, sizeY, sizeZ, sizeW));

			NNBufferIdx idx = nnBufferList.size() - 1;
			parameterBuffer.push_back(idx);
			return idx;
		}

		NNBufferIdx NeuralNetwork::CreateBuffer(const size_t sizeX, const size_t sizeY, const size_t sizeZ, const size_t sizeW, const size_t timeSteps)
		{
			//Determines if the currentn umber of maximal steps is lower than timeSteps. If this is the case update the maximal number of time steps
			if (timeSteps > maxSteps)
				maxSteps = timeSteps;

			nnBufferList.push_back(new NNIntBuffer(sizeX, sizeY, sizeZ, sizeW, timeSteps));
			return nnBufferList.size() - 1;
		}

		NNBufferIdx NeuralNetwork::CreateBuffer(const SizeVec size, const size_t timeSteps)
		{
			if (timeSteps > maxSteps)
				maxSteps = timeSteps;

			nnBufferList.push_back(new NNIntBuffer(size.sizeX, size.sizeY, size.sizeZ, size.sizeW, timeSteps));
			return nnBufferList.size() - 1;
		}

#ifdef _DEBUG
		void NeuralNetwork::WriteDataBufferGrad(NNBufferIdx buffer, const void* data, const size_t sizeX, const size_t sizeY, const size_t sizeZ, const size_t sizeW, const size_t offset)
		{
			//Writes data into the gradient part of an NNBuffer specified by buffer
			size_t totalSize = sizeW * sizeZ * sizeY * sizeX;
			NNBuffer* bufferData = nnBufferList[buffer];
			size_t bufferSize = bufferData->size.sizeW * bufferData->size.sizeZ * bufferData->size.sizeY * bufferData->size.sizeX;

			if (totalSize + offset > bufferSize)
			{
				std::cout << "Error WriteDataBuffer: Out of Range" << std::endl;
				return;
			}

			//uses backend to write into buffer
			backend.WriteDataBuffer(bufferData->BackwardBuffer(), data, offset, totalSize * sizeof(float));
		}
#endif

		//Functions for reading data out of buffers (forward or gradient buffer and at specific time points) using the backend
		void NeuralNetwork::ReadDataBuffer(NNBufferIdx buffer, void* data, const size_t sizeX, const size_t sizeY, const size_t sizeZ, const size_t sizeW, const size_t offset, const size_t time)
		{
			//Reads data from buffer and stores it in data using the Backend
			size_t totalSize = sizeW * sizeZ * sizeY * sizeX;
			NNBuffer* bufferData = nnBufferList[buffer];
			size_t bufferSize = bufferData->size.sizeW * bufferData->size.sizeZ * bufferData->size.sizeY * bufferData->size.sizeX;

			if (totalSize + offset > bufferSize || time >= bufferData->sequenceSize)
			{
				std::cout << "Error ReadDataBuffer: Out of Range" << std::endl;
				return;
			}
			auto locBuffer = bufferData->GetCompleteForwardBuffer();
			backend.ReadDataBuffer(locBuffer[time], data, offset, totalSize * sizeof(float));
		}

		void NeuralNetwork::ReadDataBufferDirect(BufferIdx buffer, void* data, const size_t totalSize, const size_t offset)
		{
			NNBuffer* bufferData = nnBufferList[buffer];

			backend.ReadDataBuffer(buffer, data, offset, totalSize * sizeof(float));
		}


		void NeuralNetwork::ReadDataBuffer(NNBufferIdx buffer, void* data, const size_t time)
		{
			NNBuffer* bufferData = nnBufferList[buffer];
			ReadDataBuffer(buffer, data, bufferData->size.sizeX, bufferData->size.sizeY, bufferData->size.sizeZ, bufferData->size.sizeW, 0, time);
		}

		void NeuralNetwork::ReadDataBufferGrad(NNBufferIdx buffer, void* data, const size_t sizeX, const size_t sizeY, const size_t sizeZ, const size_t sizeW, const size_t offset, const size_t time)
		{
			size_t totalSize = sizeW * sizeZ * sizeY * sizeX;
			NNBuffer* bufferData = nnBufferList[buffer];
			size_t bufferSize = bufferData->size.sizeW * bufferData->size.sizeZ * bufferData->size.sizeY * bufferData->size.sizeX;

			if (totalSize + offset > bufferSize || time >= bufferData->sequenceSize)
			{
				std::cout << "Error ReadDataBuffer: Out of Range" << std::endl;
				return;
			}
			auto locBwdBuffer = bufferData->GetCompleteBackwardBuffer();
			backend.ReadDataBuffer(locBwdBuffer[time], data, offset, totalSize * sizeof(float));
		}

		void NeuralNetwork::ReadDataBufferGrad(NNBufferIdx buffer, void* data, const size_t time)
		{
			NNBuffer* bufferData = nnBufferList[buffer];
			ReadDataBufferGrad(buffer, data, bufferData->size.sizeX, bufferData->size.sizeY, bufferData->size.sizeZ, bufferData->size.sizeW, 0, time);
		}

		//Saves model into the file with path fileName
		//Stores all buffers with indices listed in the map names.
		void NeuralNetwork::SaveModel(const std::string& fileName, const std::map<NNBufferIdx, char*>& names)
		{
			std::fstream file; 
			file.open(fileName.c_str(), std::ios::out | std::ios::binary);

			//The buffers are stored in binary format into a file where each buffer is preceedet by the name choosen for the buffer
			//Thereby each buffer can be loaded by using specified name. Only the forward version of a buffer is stored into the file
			for (auto it = names.begin(); it != names.end(); ++it)
				WriteBufferBinFile<float>(it->first, it->second, file);

			file.close();
		}

		void NeuralNetwork::LoadModel(const std::string& fileName, const std::map<NNBufferIdx, char*>& names)
		{
			std::fstream file;
			size_t lastPos = 0;
			file.open(fileName.c_str(), std::ios::in | std::ios::binary);

			//Loads the buffers with name in names into the corresponding buffers.
			for (auto it = names.begin(); it != names.end(); ++it)
				ReadBufferFile<float>(it->first, it->second, file);
		}

		SizeVec NeuralNetwork::GetSize(const NNBufferIdx a) const
		{
			//Returns the size of a buffer object
			return nnBufferList[a]->size;
		}

#ifdef PROFILING_ENABLED
		unsigned long long NeuralNetwork::GetTimeForward(const NNBufferIdx input, const NNBufferIdx output)
		{
			if (!graphInitiliazed)
			{
				std::cout << "Error command queue was not build!" << std::endl;
				return NN_GRAPH_NOT_INITIALIZED;
			}


			//Calculates the time needed for all forward kernel to calculate the output using input
			NNBuffer* bufferIn = nnBufferList[input];
			NNBuffer* bufferOut = nnBufferList[output];

			if (bufferIn->to != bufferOut->from)
				return 0;

			NNOp* operation = nnOperationList[bufferIn->to];
			unsigned long long result = 0;

			const size_t numOutputBuffer = operation->output.size();

			bool correct = false;
			for (size_t i = 0; i < numOutputBuffer; ++i)
				if (operation->output[i] == output)
					correct = true;

			if (!correct)
				return false;

			//All forward operations needed to perform the forward pass of this operations are executed and the results are added

			size_t operations = operation->forwardOpIdx.size();
			for (size_t i = 0; i < operations; ++i)
				result += backend.GetTime(operation->forwardOpIdx[i], BackendSystem::OpenCLBackend::OperationType::FORWARD);
			return result;
		}

		unsigned long long NeuralNetwork::GetTimeBackward(const NNBufferIdx input, const NNBufferIdx output)
		{
			if (!graphInitiliazed)
			{
				std::cout << "Error command queue was not build!" << std::endl;
				return NN_GRAPH_NOT_INITIALIZED;
			}


			//Calculates the time needed for all forward kernel to calculate the output using input

			NNBuffer* bufferIn = nnBufferList[input];
			NNBuffer* bufferOut = nnBufferList[output];

			if (bufferIn->to != bufferOut->from)
				return 0;

			NNOp* operation = nnOperationList[bufferIn->to];
			unsigned long long result = 0;

			const size_t numOutputBuffer = operation->output.size();

			bool correct = false;
			for (size_t i = 0; i < numOutputBuffer; ++i)
				if (operation->output[i] == output)
					correct = true;

			if (!correct)
				return false;

			//All backend operations needed to perform the backward pass of this operations are executed and the results are added

			size_t operations = operation->backwardOpIdx.size();
			for (size_t i = 0; i < operations; ++i)
				result += backend.GetTime(operation->backwardOpIdx[i], BackendSystem::OpenCLBackend::OperationType::BACKWARD);
			return result;
		}
#endif // PROFILING_ENABLED

		DeepCLError NeuralNetwork::InitliazeGraph(const size_t batchSize)
		{
			if (!initialized)
			{
				std::cout << "Error neural network system not initialized!" << std::endl;
				return NN_SYSTEM_NOT_INITIALIZED;
			}

			//Adjust the w component of each size to be equal to the number of batch elements if necessary. (Trainable parameters are not changed by this but input, state, intermediate, etc. buffers are effected by it)
			SetBatchSize(batchSize);

			//Calculates the maximal number of needed temporary buffers and the required size. Then the necessary number of tmpBuffers is created.
			//Operations, which need a temporary buffer will have handels to the required temporary buffers passed to them. The handles are indices into the nnBufferList vector.
			CreateTmpBuffer();

			//Creates actual OpenCL buffer objects by calling instantiate on each Buffer object
			InstantiateBuffer();

			//Create backend Operations by calling instantiate on each operations in the correct order multiple times. 
			//(The order depends on offsets in the time and the number of times instantiate is called on the number of time steps the operation runs)
			
			InstantiateOperations();
			
			//Sets all buffers to zero.
			ClearAllBuffer();

			//Initalizes all weights using the initalize operations
			InitalizeWeights();

			//The Neural Network may no be used for inference/trainig
			graphInitiliazed = true;

			return 0;
		}

		void NeuralNetwork::SetBatchSize(const size_t batchSize)
		{
			size_t size = nnBufferList.size();
			size_t i, tmpSize;

			//Loop over all buffers and call SetBatchSize on them which sets the w component of the size to be equal to the batch size if it is necessary for the buffer.
			//The parameter buffers won't do anything
			for (i = 0; i < size; ++i)
			{
				NNBuffer* buffer = nnBufferList[i];
				SizeVec sizeVec = buffer->size;
				

				buffer->SetBatchSize(batchSize);

				tmpSize = sizeVec.sizeX * sizeVec.sizeY * sizeVec.sizeZ * sizeVec.sizeW;
				if (tmpSize > maxSize)
					maxSize = tmpSize;
			}

			tmpDataMemory = new float[maxSize];
		}


		void NeuralNetwork::InstantiateBuffer()
		{
			//Retrieve information of the needed number of auxilary buffers for the optimizer.
			//Adam for example needs to auxilary buffers for the momentum and adaptive learning rate.
			if (optimizer != nullptr)
				numAuxBuffer = optimizer->GetNumBuffer();

			//The number of auxilary buffers is static because for each parameter the number of auxilary buffer is equal since the number depends only on the optimizer
			NNParamBuffer::SetNumAuxBuffer(numAuxBuffer);

			size_t size = nnBufferList.size();
			size_t i;

			//Let each buffer in nnBufferList create the required number of OpenCL buffer and subbuffer objects.
			for (i = 0; i < size; ++i)
			{
				nnBufferList[i]->Instantiate(backend);
			}
		}


		void NeuralNetwork::CreateTmpBuffer()
		{
			size_t size = nnOperationList.size();
			size_t tmpSize;
			size_t currentTmpBufferSize = 0;
			size_t i, j;

			NNOperation* op;

			std::vector<size_t> tmpBufferSize;

			//Calculate the necessary number of buffers for each operation
			for (i = 0; i < size; ++i)
			{
				op = nnOperationList[i];
				op->SetTmpBuffer(nnBufferList, i);

				tmpSize = op->tmpSizes.size();

				//Check if the required number of tmpBuffers is bigger as the current number of tmpBuffers
				if (currentTmpBufferSize < tmpSize)
				{
					//Add the necessary number of tmpBuffers
					size_t numToAdd = tmpSize - currentTmpBufferSize;
					for (j = 0; j < numToAdd ; ++j)
					{
						SizeVec opSize = op->tmpSizes[currentTmpBufferSize++];
						tmpBufferSize.push_back(opSize.sizeX * opSize.sizeY * opSize.sizeZ * opSize.sizeW);
					}
				}

				//Calculate if the currently available tmp buffers are big enough if not increase the size
				for (j = 0; j < tmpSize; ++j)
				{
					size_t totalSize = tmpBufferSize[j];
					size_t totalTmpSize = op->tmpSizes[j].sizeX * op->tmpSizes[j].sizeY * op->tmpSizes[j].sizeZ * op->tmpSizes[j].sizeW;
					tmpBufferSize[j] = totalSize < totalTmpSize ? totalTmpSize : totalSize;
				}
			}

			size = tmpBufferSize.size();

			//Add the tmpBuffers to the nnBufferList
			for (i = 0; i < size; ++i)
			{
				nnBufferList.push_back(new NNTmpBuffer(tmpBufferSize[i], 1, 1, 1));
			}
			size_t tmpBufferIdx = nnBufferList.size() - size;

			size = nnOperationList.size();

			//Pass the indices of the tmpBuffers to each operation
			for (i = 0; i < size; ++i)
			{
				op = nnOperationList[i];

				tmpSize = op->tmpSizes.size();
				for (j = 0; j < tmpSize; ++j)
				{
					op->tmpBuffer.push_back(tmpBufferIdx + j);
				}
			}
		}

		void NeuralNetwork::InstantiateOperations()
		{
			size_t size = nnOperationList.size();
			size_t i, j, k;


			//The maximal number of steps is determined by the buffer with the longest length.
			//Iterate over the maximal number of time steps to instantiate each operation the number of times necessary.
			size_t outputTime;
			NNOp* operation;
			for (j = 0; j < maxSteps; ++j)
			{
				//Iterate over all operations in the graph
				for (i = 0; i < size; ++i)
				{
					//The number of time steps an operation runs is determined by the number of time steps of the output.
					//In most cases(all) operations only have one output
					//In principle all time steps in the output must be equal.
					//At the moment there are no exceptions.

					//Determine the number of time steps the operation must be executed.
					outputTime = 0;
					operation = nnOperationList[i];
					for (k = 0; k < operation->output.size(); ++k)
					{
						if (nnBufferList[operation->output[k]]->sequenceSize > outputTime)
							outputTime = nnBufferList[operation->output[k]]->sequenceSize;
					}

					//Each operation has an associated offset allowing, for example a loss, to be computed at an specific time step. The operation is only executed when the offset is smaller  or equal than the current time step.
					if (j < outputTime && operation->timeOffset <= j)
						nnOperationList[i]->Instantiate(backend, nnBufferList);
				}
				//Each buffer contains a time step variable, which allows the hardware sub buffer of the current time step to be automatically returned.
				UpdateBufferTime();
			}

			//Set all Buffer times to zero.
			ResetBufferTime();

			//The optimizer must be instantiate as well.
			if (optimizer != nullptr)
			{
				size = parameterBuffer.size();

				for (i = 0; i < size; ++i)
				{
					//The optimizer will add auxilary buffers to the parameter buffers.
					optimizer->Instantiate(backend, nnBufferList, parameterBuffer[i]);
				}
			}
		}

		void NeuralNetwork::AddWeightInitializer(InitOp* initOp)
		{
			//The initOps stored in initOpList will be used in IntializeWeights
			initOpList.push_back(initOp);
		}

		void NeuralNetwork::InitalizeWeights()
		{

			//Calls instantiate on each initOperation therby allowing each init operation to initalize its buffers.
			size_t size = initOpList.size();

			for (size_t i = 0; i < size; ++i)
				initOpList[i]->Instantiate(nnBufferList, backend);
		}

		DeepCLError NeuralNetwork::Backward()
		{
			//Calculate the backward pass of the Neural Network.
			//Results only in non-zero results if forward pass was performed beforehand
			if (!graphInitiliazed)
			{
				std::cout << "Error command queue was not build!" << std::endl;
				return NN_GRAPH_NOT_INITIALIZED;
			}

			backend.Run(BackendSystem::OpenCLBackend::OperationType::BACKWARD);
			
			
			return 0;
		}

		DeepCLError NeuralNetwork::BatchDone()
		{
			//Calcualte update operations (Performing optimizer update etc.)
			if (!graphInitiliazed)
			{
				std::cout << "Error command queue was not build!" << std::endl;
				return NN_GRAPH_NOT_INITIALIZED;
			}

			backend.Run(BackendSystem::OpenCLBackend::OperationType::UPDATE);

			//Set the backward buffers to zero.
			ClearBackwardBuffer();
			
			return 0;
		}

		DeepCLError NeuralNetwork::Forward()
		{
			//Calculate the forward pass of the NN
			if (!graphInitiliazed)
			{
				std::cout << "Error command queue was not build!" << std::endl;
				return NN_GRAPH_NOT_INITIALIZED;
			}
			backend.Run(BackendSystem::OpenCLBackend::OperationType::FORWARD);
			return 0;
		}

		void NeuralNetwork::ClearBackwardBuffer()
		{
			size_t size = nnBufferList.size();

			//Set all backward buffers to zero.
			for (size_t i = 0; i < size; ++i)
				nnBufferList[i]->Reset(backend);
		}

		void NeuralNetwork::UpdateBufferTime()
		{
			size_t size = nnBufferList.size();
			
			//Increase the buffer time. This way the next sub buffer is retrieved when accessing the current forward and backward buffer in the buffer. Only increased if necessary (Example for state buffers)
			for (size_t i = 0; i < size; ++i)
			{
				nnBufferList[i]->UpdateStep();
			}
		}

		void NeuralNetwork::ResetBufferTime()
		{
			//Set the time variable in each buffer to zero (This determines which sub buffer will be returned by some buffers)
			size_t size = nnBufferList.size();

			//The reset function all sets the backward buffer to zero
			for (size_t i = 0; i < size; ++i)
			{
				nnBufferList[i]->Reset();
			}
		}

		void NeuralNetwork::ClearAllBuffer()
		{
			size_t size = nnBufferList.size();

			std::vector<BufferIdx> bwdBuffer;

			for (size_t i = 0; i < size; ++i)
			{
				NNBuffer* buffer = nnBufferList[i];

				//Set all backward sub buffers to zero
				bwdBuffer = buffer->GetCompleteBackwardBuffer();
				for (size_t j = 0; j < bwdBuffer.size(); ++j)
				{
					if (bwdBuffer[j] != MAX_UNSIGNED_INT)
						backend.ResetBuffer(bwdBuffer[j], buffer->size.sizeX * buffer->size.sizeY * buffer->size.sizeZ * buffer->size.sizeW * sizeof(float));
				}

				//Set all forward sub buffers to zero
				bwdBuffer = buffer->GetCompleteForwardBuffer();
				for (size_t j = 0; j < bwdBuffer.size(); ++j)
					backend.ResetBuffer(bwdBuffer[j], buffer->size.sizeX * buffer->size.sizeY * buffer->size.sizeZ * buffer->size.sizeW * sizeof(float));
			}
		}
	}
}