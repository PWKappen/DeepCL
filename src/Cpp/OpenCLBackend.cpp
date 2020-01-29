#include "OpenCLBackend.h"
#include "KernelLoader.h"
#include "Defines.h"

#include <algorithm>

namespace DeepCL
{
	namespace BackendSystem
	{
		OpenCLBackend::OpenCLBackend() :
			kernels(), timingEvent(), namesToSources(), kernelTypesToIdx(), needsToCreate()
		{
		}


		OpenCLBackend::~OpenCLBackend()
		{
			//Deletion of the created memory objects.
			//The backend has the task to destroy its operations, kernels, etc.
			size_t i;
			size_t size;
			size = forwardList.size();
			for (i = 0; i < size; ++i)
				delete forwardList[i];
			size = backwardList.size();
			for (i = 0; i < size; ++i)
				delete backwardList[i];
			size = updateList.size();
			for (i = 0; i < size; ++i)
				delete updateList[i];
			size = kernels.size();
			for (i = 0; i < size; ++i)
				delete kernels[i];
		}

		DeepCLError OpenCLBackend::LoadKernel(const std::string& kernelFile)
		{

			//Load kernels from file
			std::string* kernelCode = KernelLoader::LoadKernel(kernelFile);
			if (kernelCode->compare("ERROR") == 0)
			{
				delete kernelCode;
				return -1;
			}

			size_t bracketPos;

			std::vector <size_t> kernelPositions;

			size_t foundPos = kernelCode->find("void kernel", 0);

			//Search for all kernels contained in the file and store the beginning of the source of each kernel in kernelPositions
			while (foundPos != std::string::npos)
			{
				kernelPositions.push_back(foundPos);
				foundPos = kernelCode->find("void kernel", foundPos + 11);
			}

			//Extract the name of each Kernel Object in the file.
			//The name and source code is then stored in namesToSources
			size_t numKernels = kernelPositions.size();
			std::string kernelName;
			std::string kernelSubString;
			for (size_t i = 0; i < numKernels; ++i)
			{
				bracketPos = kernelCode->find('(', kernelPositions[i]);
				kernelName.assign(*kernelCode, (kernelPositions[i] + 12), bracketPos - (kernelPositions[i] + 12));
				kernelSubString = kernelCode->substr(kernelPositions[i], (i + 1 < numKernels ? kernelPositions[i + 1] : kernelCode->length()) - kernelPositions[i]);
				namesToSources.insert(std::pair<std::string, std::string>(kernelName, kernelSubString));
			}

			return 0;

		}

		DeepCLError OpenCLBackend::LoadKernelFromConfig(const std::string& configFilePath, const std::string& kernelPath)
		{
			//Load the names of all kernel files
			std::vector<std::string>* kernelFiles = KernelLoader::LoadKernelConfig(configFilePath);
			size_t size = kernelFiles->size();
			if (size == 0)
				return 1;
			//Iterate over all kernel files and load it
			size_t error;
			for (size_t i = 0; i < size; ++i)
			{
				error = LoadKernel(kernelPath + (*kernelFiles)[i]);
				if (error != 0)
					return error;
			}

			return 0;
		}

		DeepCLError OpenCLBackend::InitGPU()
		{
			std::cout << "OpenCL Deep Learning Project" << std::endl << std::endl;
			
			//Retrieve all avilable Platforms
			std::vector<cl::Platform> platforms;
			cl::Platform::get(&platforms);

			size_t platformsSize = platforms.size();

			if (platformsSize == 0)
			{
				std::cout << "Error no OpenCL platforms found!" << std::endl;
				system("PAUSE");
				return -1;
			}

			std::cout << "Available platforms: " << std::endl;

			for (size_t i = 0; i < platformsSize; ++i)
			{
				std::cout << i << " " << platforms[i].getInfo<CL_PLATFORM_NAME>() << std::endl;
			}		

			std::cout << std::endl;

			std::vector<cl::Device>();

			//Let user choose the Platform
			size_t platformChoosenIdx;
			bool platformChoosen = false;

			while (!platformChoosen)
			{
				std::cout << "Choose platform\t";
				std::cin >> platformChoosenIdx;

				platformChoosen = platformsSize > platformChoosenIdx;
			}

			std::cout << std::endl << std::endl;

			platform = platforms[platformChoosenIdx];

			//query all avialbe devices
			std::vector<cl::Device> devices;

			platform.getDevices(CL_DEVICE_TYPE_ALL, &devices);

			platformsSize = devices.size();

			if (platformsSize == 0)
			{
				std::cout << "Error no OpenCL devices found!" << std::endl;
				system("PAUSE");
				return -1;
			}

			std::cout << "Available devices: "<< std::endl;

			for (size_t i = 0; i < platformsSize; ++i)
			{
				std::cout << i << " " << devices[i].getInfo<CL_DEVICE_NAME>() << std::endl;
			}

			std::cout << std::endl;

			platformChoosen = false;


			//Let the user choose a device
			while (!platformChoosen)
			{
				std::cout << "Choose device \t";
				std::cin >> platformChoosenIdx;

				platformChoosen = platformsSize > platformChoosenIdx;
			}

			std::cout << std::endl;

			device = devices[platformChoosenIdx];

			//Some meta information of the device
			std::cout << "Choosen device: \t" << device.getInfo<CL_DEVICE_NAME>() << std::endl;
			std::cout << "Memory on Device: \t" << device.getInfo<CL_DEVICE_GLOBAL_MEM_SIZE>() << std::endl;
			std::cout << "Local Memory available on device: \t" << device.getInfo<CL_DEVICE_LOCAL_MEM_SIZE>() << std::endl << std::endl;
			
			baseAddrAllign = device.getInfo< CL_DEVICE_MEM_BASE_ADDR_ALIGN >();

			//Create a context for the device
			context = cl::Context({ device });


			//Create a OpenCL commandqueue either with profiling enabled or not
#ifdef PROFILING_ENABLED
			comQueue = cl::CommandQueue(context, device, CL_QUEUE_PROFILING_ENABLE);
#else 
			comQueue = cl::CommandQueue(context, device, 0);
#endif

			return 0;
		}


		KernelIdx OpenCLBackend::GetKernelIdx(const std::string& fileName)
		{
			//Check if the kernel already exists if it does return the kernel
			std::map<std::string, KernelIdx>::iterator it = kernelTypesToIdx.find(fileName);
			if (it == kernelTypesToIdx.end())
			{
				//Check if source code for the kernel is available
				if (namesToSources.find(fileName) == namesToSources.end())
				{
					std::cerr << "Error kernel: " << fileName << " does not exist" << std::endl;
					return MAX_UNSIGNED_INT;
				}

				//create a placeholder kernel and in the kernel vector
				KernelIdx idx = kernels.size();

				//Add the kernel to needsToCreate to create the kernel at an later point
				needsToCreate[idx] = fileName;
				kernels.push_back(nullptr);
				kernelTypesToIdx[fileName] = idx;
				return idx;
			}

			return it->second;
		}

		KernelIdx OpenCLBackend::GetKernelIdx(const std::string& fileName, const std::string& compileDefines)
		{
			//The same as before but this time wiht additional defines for the OpenCL compiler
			//The kernel is stored in combination with the compiler defines
			std::map<std::string, KernelIdx>::iterator it = kernelTypesToIdx.find(fileName+compileDefines);
			if (it == kernelTypesToIdx.end())
			{
				if (namesToSources.find(fileName) == namesToSources.end())
				{
					std::cerr << "Error kernel: " << fileName << " does not exist" << std::endl;
					return MAX_UNSIGNED_INT;
				}

				//Extract all defines
				std::vector<size_t> defines;
				defines.push_back(0);
				size_t spacePos = 0;
				
				while (true)
				{
					spacePos = compileDefines.find(" ", spacePos + 1);
					
					if (spacePos == std::string::npos)
						break;
					else
						++spacePos;
					defines.push_back(spacePos);
				}
				
				std::string fullStrings = fileName+"!";

				//Create a list of defines that can be pussed to the OpenCL compiler
				size_t numDefines = defines.size();
				for (size_t i = 0; i < numDefines; ++i)
					fullStrings += "-D" + compileDefines.substr(defines[i], (i + 1 < numDefines ? defines[i + 1] -1: compileDefines.length()) - defines[i]) + " ";

				//Create placeholder etc.
				KernelIdx idx = kernels.size();
				needsToCreate[idx] = fullStrings;
				kernels.push_back(nullptr);
				kernelTypesToIdx[fileName+compileDefines] = idx;
				return idx;
			}

			return it->second;
		}


		void OpenCLBackend::CreateIfNecessary(const KernelIdx kernelIdx)
		{
			// check if Kernel needs to be created
			std::map<KernelIdx, std::string>::iterator it = needsToCreate.find(kernelIdx);
			if (it != needsToCreate.end())
			{
				//Retrieve kernel name and compile time parameters
				std::string fileNameAdded = it->second;

				size_t idx = fileNameAdded.find("!");
				std::string kernelName;
				std::string arguments;
				if (idx == std::string::npos)
				{
					kernelName = fileNameAdded;
					arguments = "";
				}
				else
				{
					kernelName = fileNameAdded.substr(0, idx);
					arguments = fileNameAdded.substr(idx + 1, fileNameAdded.size() - idx);
				}

				//Compile the kernel object using the arguments
				BuildSingleKernel(kernelName, arguments, kernelIdx);
				//Kernel was created and doesn't need to be created anymore
				needsToCreate.erase(kernelIdx);
			}
		}

		void OpenCLBackend::BuildSingleKernel(const std::string& kernelName, const std::string& defineArguments, const KernelIdx kernelIdx)
		{
			//Create a new program with the sourcecode of kernelName
			size_t idx = programList.size();
			programList.push_back(cl::Program(context, namesToSources[kernelName]));

			//Create Compile time arguments. Allows possible optimization to be enabled.
			std::string compileArguments = "";//"-cl-mad-enable -cl-fast-relaxed-math ";
			compileArguments += defineArguments;

			//Compile program
			if (programList[idx].build({ device }, compileArguments.c_str()) != CL_SUCCESS)
			{
				std::cerr << "Error building: " << programList[idx].getBuildInfo<CL_PROGRAM_BUILD_LOG>(device) << std::endl;
				return;
			}
			//Extract kernel object and store it in kernels at the correct position
			kernels[kernelIdx] = new cl::Kernel(programList[idx], kernelName.c_str());
		}

		
		void OpenCLBackend::Run(const OperationType opType)
		{
			//query the vector specified by opList (FORWARD, BACKWARD or UPDATE)
			std::vector<BaseOperation*>* opList = (opType == OperationType::FORWARD ? &forwardList : (opType == OperationType::BACKWARD ? &backwardList : &updateList));
			size_t size = opList->size();

			//Stores the execution time for each operation in the corresponding pass
#ifdef PROFILING_ENABLED
			std::vector<cl_ulong>* opTimes = (opType == OperationType::FORWARD ? &forwardTime : (opType == OperationType::BACKWARD ? &backwardTime : &updateTime));
#endif		
			
			//Run the operations in the vector starting at the end
			if (opType == OperationType::BACKWARD)
				RunBackward(opList, 
#ifdef PROFILING_ENABLED
					opTimes,
#endif // PROFILING_ENABLED
					size);

			//Run the operations in the vector starting at the begninning
			else
				RunForward(opList, 
#ifdef PROFILING_ENABLED
					opTimes,
#endif // PROFILING_ENABLED 
					size);

			//Wait for operations to finish (Maybe remove this statemend and wait only when absolutely necessary
			timingEvent.wait();
		}


		void OpenCLBackend::RunForward(std::vector<BaseOperation*>* opList, 
#ifdef PROFILING_ENABLED
			std::vector<cl_ulong>* opTimes,
#endif // PROFILING_ENABLED
			const size_t size)
		{
			//Enque each operation in the openCL queue
			for (size_t i = 0; i < size; ++i)
			{
				(*opList)[i]->Run(comQueue, &timingEvent, bufferList);

				//Calculate the execution time of the operation(Reduces spead of execution)
#ifdef PROFILING_ENABLED
				cl_ulong timeStart, timeEnd;
				timingEvent.wait();
				timeStart = timingEvent.getProfilingInfo<CL_PROFILING_COMMAND_START>();
				timeEnd = timingEvent.getProfilingInfo<CL_PROFILING_COMMAND_END>();
				(*opTimes)[i] = timeEnd - timeStart;
#endif
			}
#ifdef _DEBUG
			cl_int err = comQueue.finish();
			if (err != CL_SUCCESS)
				std::cout << "Error in execution of commands in queue: " << err << std::endl;
#endif // DEBUG
		}

		void OpenCLBackend::RunBackward(std::vector<BaseOperation*>* opList, 
#ifdef PROFILING_ENABLED
			std::vector<cl_ulong>* opTimes,
#endif // PROFILING_ENABLED
			const size_t size)
		{
			//Enqueue each operation in the OpenCL queue starting at the end
			for (size_t i = size - 1; i < size; --i)
			{
				(*opList)[i]->Run(comQueue, &timingEvent, bufferList);

				//Calculate the execution time of the operation(Reduces spead of execution)
#ifdef PROFILING_ENABLED
				cl_ulong timeStart, timeEnd;
				timingEvent.wait();
				timeStart = timingEvent.getProfilingInfo<CL_PROFILING_COMMAND_START>();
				timeEnd = timingEvent.getProfilingInfo<CL_PROFILING_COMMAND_END>();
				(*opTimes)[i] = timeEnd - timeStart;
#endif
			}

#ifdef _DEBUG
			cl_int err = comQueue.finish();
			if (err != CL_SUCCESS)
				std::cout << "Error in execution of commands in queue: " << err << std::endl;
#endif
		}

#ifdef PROFILING_ENABLED
		unsigned long long OpenCLBackend::GetTime( const OperationIdx opIdx, const OperationType opType)
		{
			//Return the time of the operation
			std::vector<cl_ulong>* opTimes = (opType == OperationType::FORWARD ? &forwardTime : (opType == OperationType::BACKWARD ? &backwardTime : &updateTime));
			return (*opTimes)[opIdx];
		}
#endif

		BufferIdx OpenCLBackend::CreateBuffer(const size_t size, const MEM_FLAG memFlag, const size_t numSubBuffer)
		{

			cl_mem_flags memFlagCL = memFlag == MEM_FLAG::READ_WRITE ? CL_MEM_READ_WRITE : (memFlag == MEM_FLAG::WRITE_ONLY ? CL_MEM_WRITE_ONLY : CL_MEM_READ_ONLY);
			size_t paddedSize = size;
			//Calculate the correct size of the buffer including the necessary memory alignment
			paddedSize = baseAddrAllign*((size + baseAddrAllign-1) / baseAddrAllign)*numSubBuffer;

			//Create a new OpenCL buffer object and store it in the bufferList. The index of the buffer is returned at the end
#ifdef _DEBUG
			cl_int err;
			bufferList.push_back(cl::Buffer(context, memFlagCL, paddedSize, nullptr, &err));
			if (err != CL_SUCCESS)
				std::cout << "Error create Buffer: " << err << std::endl;
#else
			bufferList.push_back(cl::Buffer(context, memFlagCL, paddedSize));
#endif // DEBUG
			return bufferList.size() - 1;
		}

	
		BufferIdx OpenCLBackend::CreateSubBuffer(const BufferIdx bufferIdx, const size_t size, const MEM_FLAG memFlag, const size_t idxBuffer)
		{
			cl_mem_flags memFlagCL = memFlag == MEM_FLAG::READ_WRITE ? CL_MEM_READ_WRITE : (memFlag == MEM_FLAG::WRITE_ONLY ? CL_MEM_WRITE_ONLY : CL_MEM_READ_ONLY);
			
			//calculate the correct offset incorporating the necessary memory alignment
			size_t paddOffset = baseAddrAllign*((size + baseAddrAllign-1) / baseAddrAllign) * idxBuffer;
			cl_buffer_region region = { paddOffset, size };

			//Create a sub buffer using the by bufferIdx specified OpenCL buffer
#ifdef _DEBUG
			cl_int err;
			bufferList.push_back(bufferList[bufferIdx].createSubBuffer(memFlagCL, CL_BUFFER_CREATE_TYPE_REGION, static_cast<void*>(&region), &err));
			if (err != CL_SUCCESS)
				std::cout << "Error create SubBuffer: " << err << std::endl;
#else
			bufferList.push_back(bufferList[bufferIdx].createSubBuffer(memFlagCL, CL_BUFFER_CREATE_TYPE_REGION, static_cast<void*>(&region)));
#endif // DEBUG
			return bufferList.size() - 1;
		}
		
		void OpenCLBackend::WriteDataBuffer(BufferIdx idx, const void* data, const size_t offset, const size_t size)
		{
#ifdef _DEBUG
			cl_int err = comQueue.enqueueWriteBuffer((bufferList[idx]), CL_TRUE, offset, size, data);
			if (err != CL_SUCCESS)
				std::cout << "Error write buffer: " << err << std::endl;
#else
			comQueue.enqueueWriteBuffer((bufferList[idx]), CL_FALSE, offset, size, data);
#endif // DEBUG
		}

		void OpenCLBackend::ReadDataBuffer(BufferIdx idx, void* data, const size_t offset, const size_t size)
		{
#ifdef _DEBUG
			cl_int err = comQueue.enqueueReadBuffer(bufferList[idx], CL_TRUE, offset, size, data);
			if (err != CL_SUCCESS)
				std::cout << "Error read buffer: " << err << std::endl;
#else
			comQueue.enqueueReadBuffer(bufferList[idx], CL_FALSE, offset, size, data);
#endif // DEBUG
		}

		void OpenCLBackend::ResetBuffer(BufferIdx idx, const size_t size)
		{
			cl_float pattern = 0;
			//Fill Buffer with zeros. if necessary second function with arbitrary pattern can be created
#ifdef _DEBUG
			cl_int err = comQueue.enqueueFillBuffer<cl_float>(bufferList[idx], pattern, 0, size);
			if (err != CL_SUCCESS)
				std::cout << "Error read buffer: " << err << std::endl;
#else
			comQueue.enqueueFillBuffer(bufferList[idx], pattern, 0, size);
#endif // DEBUG
		}
	}
}
