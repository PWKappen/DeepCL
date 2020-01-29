#pragma once

#include <vector>
#include <memory>
#include <map>

#include "Operation.h"

namespace DeepCL
{
	namespace BackendSystem
	{

		//Class for Interacting with OpenCL (Creating OpenCL Buffer, Kernel, etc.)
		//It also handles the execution of each different Pass(Forward, Backward, Update)
		class OpenCLBackend
		{
		public:
			OpenCLBackend();
			~OpenCLBackend();

			//Used to describe the type of the operation
			enum OperationType
			{
				FORWARD, BACKWARD, UPDATE
			};

			//Needs to be run in order to Initalize OpenCL:
			//Quering and selecting a vendor/device.
			//Loading the Code out of kernel files
			DeepCLError InitGPU();

			//Loads a specific kernel File
			DeepCLError LoadKernel(const std::string& kernelFolder);

			//Loads kernels using a kernel config file. The kernels must be contained in the path specified by kernelPath.
			DeepCLError LoadKernelFromConfig(const std::string& configFilePath, const std::string& kernelPath);

			//Return the index of a specific Kernel object stored in kernels.
			KernelIdx GetKernelIdx(const std::string& fileName);
			//Returns the index of a specific Kernel object stored in kernels, allowing additional kernel defines.
			KernelIdx GetKernelIdx(const std::string& fileName, const std::string& compileDefines);

			//Adds an operation to the specific Operation vector defined by opType
			template<size_t Tsize, class... Ts>
			OperationIdx AddOperation(const KernelIdx kernel,
				const Tuple<Ts...> tuple,
				const cl::NDRange offset,
				const cl::NDRange globalSize,
				const cl::NDRange localSize, const OperationType opType);

			//Adds an increment operation to the specific Operation vector defined by opType
			template<size_t idx, size_t Tsize, class... Ts>
			OperationIdx AddOperation(const KernelIdx kernel,
				const Tuple<Ts...> tuple,
				const cl::NDRange offset,
				const cl::NDRange globalSize,
				const cl::NDRange localSize, const OperationType opType);

			//Runs a specific kernel objects using the in tuple defined parameters
			template<size_t Tsize, class... Ts>
			void RunKernel(const KernelIdx kernel, const Tuple<Ts...> tuple, const cl::NDRange offset, const cl::NDRange globalSize, const cl::NDRange localSize);
			
			//Executes one of the three passes specified by opType.
			void Run(const OperationType opType);

			//Returns the time a specific operation takes in the specified pass.
#ifdef PROFILING_ENABLED
			unsigned long long GetTime(const OperationIdx opIdx, const OperationType opType);
#endif // PROFILING_ENABLED

			//Creates a buffer which inclues padding to allow the specified number of sub buffers
			BufferIdx CreateBuffer(const size_t size, const MEM_FLAG memFlag, const size_t numSubBuffer);

			//Creates a subbuffer in the by bufferIdx specified buffer. 
			BufferIdx CreateSubBuffer(const BufferIdx bufferIdx, const size_t size, const MEM_FLAG memFlag, const size_t idxBuffer);
			//Write data into an arbitrary buffer
			void WriteDataBuffer(BufferIdx idx, const void* data, const size_t offset, const size_t size);
			//Read the content of a specified buffer into data
			void ReadDataBuffer(BufferIdx idx, void* data, const size_t offset, const size_t size);
			//Set the specified buffer to zero.
			void ResetBuffer(BufferIdx idx, const size_t size);

			//Returns the alilgnment needed when creating subbuffers
			cl_uint GetBaseAddrAllignment()const { return baseAddrAllign; }

		private:

			//Vector contains pointer on all kernel objects avialable (Some may not be initalized directly)
			std::vector<cl::Kernel*> kernels;

			//Contains the name of a kernel and the corresponding source code
			std::map < std::string, std::string> namesToSources;

			//Allows the retrival of the kernel index using the name of it
			std::map < std::string, KernelIdx> kernelTypesToIdx;

			//Contains kernel objects with the corresponding compile time arguments that must be created
			std::map < KernelIdx, std::string> needsToCreate;

			//List of OpenCL program objects
			std::vector<cl::Program> programList;

			//Checks if kernel needs to be created(contained in needsToCreate) and does so if it is the case using BuildSingleKernel
			void CreateIfNecessary(const KernelIdx kernelIdx);

			//Builds a single kernel from source
			void BuildSingleKernel(const std::string& fileName, const std::string& defineArguments, const KernelIdx kernelIdx);

			//Runs all kernels in opList from start to end
			void RunForward(std::vector<BaseOperation*>* opList, 
#ifdef PROFILING_ENABLED
				std::vector<cl_ulong>* opTimes,
#endif // PROFILING_ENABLED
				 const size_t);

			//Runs all kernels in opList from end to start
			void RunBackward(std::vector<BaseOperation*>* opList, 
#ifdef PROFILING_ENABLED
				std::vector<cl_ulong>* opTimes,
#endif // PROFILING_ENABLED
				const size_t);

			//Stores the chosen platform
			cl::Platform platform;

			//Stores the chosen device
			cl::Device device;
			
			//Stores the created context
			cl::Context context;

			//Stores the comQueue used by the whole backend.
			cl::CommandQueue comQueue;

			//Vector of operations for the forward pass
			std::vector<BaseOperation*> forwardList;

			//Vector of operations for the backward pass
			std::vector<BaseOperation*> backwardList;

			//Vector of operations for the update pass
			std::vector<BaseOperation*> updateList;

			//Vectors to store operation times
#ifdef PROFILING_ENABLED
			std::vector<cl_ulong> forwardTime;
			std::vector<cl_ulong> backwardTime;
			std::vector<cl_ulong> updateTime;
#endif

			//The necessary alignment for sub buffers
			cl_uint baseAddrAllign;

			//List of all used OpenCL Buffers (Normal and Sub Buffers)
			std::vector<cl::Buffer> bufferList;

			//Event to calculate the timing information
			cl::Event timingEvent;
		};

		template<size_t Tsize, class... Ts>
		OperationIdx OpenCLBackend::AddOperation(const KernelIdx kernel,
			const Tuple<Ts...> tuple,
			const cl::NDRange offset,
			const cl::NDRange globalSize,
			const cl::NDRange localSize, const OperationType opType)
		{

			//If the kernel was not created before it gets created now
			CreateIfNecessary(kernel);

			//Create and Add operation to the vector of the corresponding pass
			Operation<Tsize, Ts...>* operation = new Operation<Tsize, Ts...>((kernels[kernel]), tuple, offset, globalSize, localSize);

			std::vector<BaseOperation*>* opList = opType == OperationType::FORWARD ? &forwardList : (opType == OperationType::BACKWARD ? &backwardList : &updateList);
			opList->push_back(operation);
#ifdef PROFILING_ENABLED
			std::vector<cl_ulong>* opTimes = opType == OperationType::FORWARD ? &forwardTime : (opType == OperationType::BACKWARD ? &backwardTime : &updateTime);
			opTimes->push_back(0);
#endif // PROFILING_ENABLED
			return opList->size() - 1;
		}


		template<size_t idx, size_t Tsize, class... Ts>
		OperationIdx OpenCLBackend::AddOperation(const KernelIdx kernel,
			const Tuple<Ts...> tuple,
			const cl::NDRange offset,
			const cl::NDRange globalSize,
			const cl::NDRange localSize, const OperationType opType)
		{
			CreateIfNecessary(kernel);
			
			//Create an operation that increments the variable at index every time step
			IncrementOperation<idx, Tsize, Ts...>* operation = new IncrementOperation<idx, Tsize, Ts...>((kernels[kernel]), tuple, offset, globalSize, localSize);

			std::vector<BaseOperation*>* opList = opType == OperationType::FORWARD ? &forwardList : (opType == OperationType::BACKWARD ? &backwardList : &updateList);
			opList->push_back(operation);
#ifdef PROFILING_ENABLED
			std::vector<cl_ulong>* opTimes = opType == OperationType::FORWARD ? &forwardTime : (opType == OperationType::BACKWARD ? &backwardTime : &updateTime);
			opTimes->push_back(0);
#endif // PROFILING_ENABLED
			return opList->size() - 1;
		}

		//Run a specific kernel object
		template<size_t Tsize, class... Ts>
		void OpenCLBackend::RunKernel(const KernelIdx kernel, const Tuple<Ts...> tuple, const cl::NDRange offset, const cl::NDRange globalSize, const cl::NDRange localSize)
		{
			Operation<Tsize, Ts...>* operation = new Operation<Tsize, Ts...>(kernel[kernel], tuple, offset, globalSize, localSize);
			operation->Run(comQueue, &timingEvent, bufferList);
		}
	}
}

