#include "DataReader.h"
//#include "lodepng.h"

#include <Windows.h>

namespace DeepCL
{
	namespace DataSystem
	{
		//Reads an IDX file of type float.
		float* IDXReader::ReadIDXFileConvertFloat(const std::string& fileName, NNSystem::SizeVec& resultSize, size_t& numData)
		{
			unsigned char* result = ReadIDXFile<unsigned char>(fileName, resultSize, numData);
			if (result == nullptr)
				return nullptr;
			size_t totalSize = resultSize.sizeX*resultSize.sizeY*resultSize.sizeZ*resultSize.sizeW*numData;
			float* floatResult = new float[totalSize];
			for (size_t i = 0; i < totalSize; ++i)
				floatResult[i] = static_cast<float>(result[i]) / 255.f;

			delete[] result;

			return floatResult;
		}

		//Reads an IDX file of type int
		int* IDXReader::ReadIDXFileConvertInt(const std::string& fileName, NNSystem::SizeVec& resultSize, size_t& numData)
		{
			unsigned char* result = ReadIDXFile<unsigned char>(fileName, resultSize, numData);
			if (result == nullptr)
				return nullptr;
			size_t totalSize = resultSize.sizeX*resultSize.sizeY*resultSize.sizeZ*resultSize.sizeW*numData;
			int* floatResult = new int[totalSize];
			for (size_t i = 0; i < totalSize; ++i)
				floatResult[i] = static_cast<int>(result[i]);

			delete[] result;

			return floatResult;
		}

		void IDXReader::Init()
		{
			//Load all images and labels into memory.
			dataLocal = ReadIDXFileConvertFloat(fileNameData, dataSize, numData);
			labelLocal = ReadIDXFileConvertInt(fileNameLabel, labelSize, numData);

			if (dataLocal == nullptr || labelLocal == nullptr) {
				initalized = false;
				return;
			}
			//Compute size of one element
			unrolledDataSize = dataSize.sizeX * dataSize.sizeY * dataSize.sizeZ * dataSize.sizeW;
			sizeData =  unrolledDataSize * numData;
			unrolledLabelSize = labelSize.sizeX * labelSize.sizeY * labelSize.sizeZ * labelSize.sizeW;
			sizeLabel = unrolledLabelSize * numData;

			//Start with example zero.
			currentDataPos = 0;
			currentLabelPos = 0;
			initalized = true;
		}

		BaseDataReader<int, float>* IDXReader::AllocateCopy()
		{
			IDXReader* reader = new IDXReader(fileNameData, fileNameLabel, batchSize);

			return reader;
		}

		void IDXReader::GetNextData(BackendSystem::Tuple<std::vector<int>, std::vector<float>>& data, std::vector<std::vector<size_t>>& offsets, std::vector<std::vector<NNSystem::SizeVec>>& sizes)
		{
			float* tmpDataPos = dataLocal + currentDataPos;
			int* tmpLabelPos = labelLocal + currentLabelPos;

			float* tmpDataEnd = dataLocal + (currentDataPos + batchSize * unrolledDataSize) % (numData*unrolledDataSize);
			int* tmpLabelEnd = labelLocal + (currentLabelPos + batchSize * unrolledLabelSize) % (numData*unrolledLabelSize);

			BackendSystem::get<1>(data).resize(unrolledDataSize * batchSize);

			//Copy batch size number of images into the data tuple
			if (tmpDataPos >= tmpDataEnd)
			{
				size_t copiedSize = unrolledDataSize * numData - currentDataPos;
				void* tmpPointer = reinterpret_cast<void*> (tmpDataPos);
				std::memcpy(BackendSystem::get<1>(data).data(), tmpPointer, (copiedSize)* sizeof(float));
				tmpPointer = reinterpret_cast<void*> (dataLocal);
				std::memcpy(&(BackendSystem::get<1>(data).data()[copiedSize]), tmpPointer, (unrolledDataSize * batchSize - copiedSize) * sizeof(float));
			}
			else
			{
				void* tmpPointer = reinterpret_cast<void*> (tmpDataPos);
				std::memcpy(BackendSystem::get<1>(data).data(), tmpPointer, unrolledDataSize * batchSize * sizeof(float));
			}


			BackendSystem::get<0>(data).resize(unrolledLabelSize * batchSize);
			//Copy batch size number of label into the data tuple.
			if (tmpDataPos >= tmpDataEnd)
			{
				size_t copiedSize = unrolledLabelSize * numData - currentLabelPos;
				void* tmpPointer = reinterpret_cast<void*> (tmpLabelPos);
				std::memcpy(BackendSystem::get<0>(data).data(), tmpPointer, (copiedSize)* sizeof(int));
				tmpPointer = reinterpret_cast<void*> (labelLocal);
				std::memcpy(&BackendSystem::get<0>(data).data()[copiedSize], tmpPointer, (unrolledLabelSize * batchSize - copiedSize) * sizeof(int));
			}
			else
			{
				void* tmpPointer = reinterpret_cast<void*> (tmpLabelPos);
				std::memcpy(BackendSystem::get<0>(data).data(), tmpPointer, unrolledLabelSize * batchSize * sizeof(int));
			}

			currentDataPos = (currentDataPos + batchSize*unrolledDataSize) % (numData*unrolledDataSize);
			currentLabelPos = (currentLabelPos + batchSize*unrolledLabelSize) % (numData*unrolledLabelSize);
			sizes[0].push_back(labelSize);
			sizes[1].push_back(dataSize);
		}

		//Add offset to the readers.
		void IDXReader::AddOffset(const size_t offset)
		{
			currentDataPos = (currentDataPos + offset * unrolledDataSize) % (numData*unrolledDataSize);
			currentLabelPos = (currentLabelPos + offset * unrolledLabelSize) % (numData*unrolledLabelSize);
		}
	}
}