#pragma once

#include <vector>
#include <string>
#include <set>
#include <map>
#include <fstream>
#include <cassert>
#include "VariadicTuple.h"
#include "NNOperations.h"

namespace DeepCL
{
	namespace DataSystem
	{
		//Class, from which all data loaders, which shall use the BatchManager should derive.
		//The variadic template defines the number of different data parts that should be used.
		//Example for data parts are an integer for the label and an float for an input image.
		//The variadic template will be changed to be a variadic template of vectors. Each vector contains the batch size number of data points.
		template<typename... varT>
		class BaseDataReader
		{
		public:
			BaseDataReader(const size_t batchSize);
			virtual ~BaseDataReader();
			
			//Function to create a copy of this object for each thread.
			virtual BaseDataReader<varT...>* AllocateCopy() = 0;

			//Fills the tuple data with a full batch. The different data points are allowed to have different sizes (The transformer has the purpose to change this).
			//Since every data point has potentialy a different size the offsets are stored in the vector of vectors offsets. The vector contains number of different data parts vectors.
			//The size of each data point is stores in the vector of vectors sizes.
			virtual void GetNextData(BackendSystem::Tuple<std::vector<varT>...>& data, std::vector<std::vector<size_t>>& offsets, std::vector<std::vector<NNSystem::SizeVec>>& sizes) = 0;

			//Changes a current position from which data is read. This is necessary when multiple threads are used since different threads should load data from different positions.
			virtual void AddOffset(const size_t offset) = 0;

			//Returns the total number of data points in the data set. May not always be correctly defined.
			size_t GetNumData() const { return numData; }
			
			//Returns true if object was initalized correctly.
			bool Initalized() const { return initalized; }

		protected:
			size_t batchSize;
			size_t numData;
			bool initalized;
		};

		template<typename... varT>
		BaseDataReader<varT...>::BaseDataReader(const size_t batchSize) :
			batchSize(batchSize), initalized(false)
		{
		}
		template<typename... varT>
		BaseDataReader<varT...>::~BaseDataReader()
		{
		}

		class IDXReader : public BaseDataReader<int, float>
		{
		public:
			IDXReader(const std::string& fileNameData, const std::string fileNameLabel, const size_t batchSize) : BaseDataReader(batchSize),
				fileNameData(fileNameData), fileNameLabel(fileNameLabel)
			{
				Init();
			}

			~IDXReader()
			{
				delete[] dataLocal;
				delete[] labelLocal;
			}

			virtual BaseDataReader<int, float>* AllocateCopy();
			virtual void GetNextData(BackendSystem::Tuple<std::vector<int>, std::vector<float>>& data, std::vector<std::vector<size_t>>& offsets, std::vector<std::vector<NNSystem::SizeVec>>& sizes);

			virtual void AddOffset(const size_t offset);

		protected:
			//Reads an floating point data point stored in IDX format
			float* ReadIDXFileConvertFloat(const std::string& fileName, NNSystem::SizeVec& resultSize, size_t& numData);

			//Reads an integer point data point stored in IDX format
			int* ReadIDXFileConvertInt(const std::string& fileName, NNSystem::SizeVec& resultSize, size_t& numData);

			//Read an part of an data example of type T stored in an IDX file
			template<typename T>
			T* ReadIDXFile(const std::string& fileName, NNSystem::SizeVec& resultSize, size_t& numData);

			//Reverse the ordering of bytes since the IDX format stores integers in the reverse order
			template<typename T>
			T Reverse(T input);


			void Init();

			//Stores the complete data in local memory since the MNIST dataset is rather small.
			float* dataLocal;
			int* labelLocal;

			//The number of data points
			size_t sizeData;

			size_t sizeLabel;
			size_t currentDataPos;
			size_t currentLabelPos;
			size_t unrolledLabelSize;
			size_t unrolledDataSize;

			NNSystem::SizeVec dataSize;
			NNSystem::SizeVec labelSize;

			std::string fileNameData;
			std::string fileNameLabel;
		};

		template<typename T>
		T* IDXReader::ReadIDXFile(const std::string& fileName, NNSystem::SizeVec& resultSize, size_t& numData)
		{
			std::ifstream file(fileName, std::ios::binary | std::ios::in);

			if (file.is_open())
			{
				int magicNumber;
				//Number at the beginning of an IDXFile.
				file.read(reinterpret_cast<char*>(&magicNumber), sizeof(magicNumber));
				magicNumber = Reverse<int>(magicNumber);

				int type = (magicNumber >> 8) & 255;
				int dimensions = (magicNumber)& 255;

				//Check if the types match.
				switch (type)
				{
				case 0x08:
					assert((std::is_same<T, unsigned char>::value));
					break;
				case 0x09:
					assert((std::is_same<T, char>::value));
					break;
				case 0x0B:
					assert((std::is_same<T, short>::value));
					break;
				case 0x0C:
					assert((std::is_same<T, int>::value));
					break;
				case 0x0D:
					assert((std::is_same<T, float>::value));
					break;
				case 0x0E:
					assert((std::is_same<T, double>::value));
					break;
				default:
					std::cout << "ERROR: type is not specified for IDX" << std::endl;
					file.close();
					return nullptr;
				}

				int* dimensionSizes = new int[dimensions];

				//Read the dimension of data points.
				for (int i = 0; i < dimensions; ++i)
				{
					file.read(reinterpret_cast<char*>(&(dimensionSizes[i])), sizeof(int));
					dimensionSizes[i] = Reverse<int>(dimensionSizes[i]);
				}

				numData = dimensionSizes[0];

				//Check if the number of dimensions match.
				resultSize.sizeX = 1;
				resultSize.sizeY = 1;
				resultSize.sizeZ = 1;
				resultSize.sizeW = 1;
				if (dimensions != 1)
				{
					switch (dimensions - 2)
					{
					case 3:
						resultSize.sizeW = dimensionSizes[4];
					case 2:
						resultSize.sizeZ = dimensionSizes[3];
					case 1:
						resultSize.sizeY = dimensionSizes[2];
					case 0:
						resultSize.sizeX = dimensionSizes[1];
						break;
					default:
						std::cout << "ERROR: to many dimensions in File!" << std::endl;
						file.close();
						return nullptr;
					}
				}

				size_t totalSize = static_cast<size_t>(dimensionSizes[0]);

				for (int i = 1; i < dimensions; ++i)
					totalSize *= static_cast<size_t>(dimensionSizes[i]);

				T* result = new T[totalSize];

				//Read all data points into local memory.
				file.read(reinterpret_cast<char*> (result), totalSize * sizeof(T));

				//Perform inversing if an integral is loaded.
				if (std::is_integral<T>::value && sizeof(T) > 1)
				{
					for (size_t i = 0; i < totalSize; ++i)
						result[i] = Reverse<T>(result[i]);
				}

				delete[] dimensionSizes;
				file.close();

				return result;
			}
			else 
				std::cout << "ERROR: File " << fileName << " could not be opened!" << std::endl;

			file.close();
			return nullptr;
		}

		template<typename T>
		T IDXReader::Reverse(T input)
		{
			static_assert(std::is_integral<T>::value, "ERROR: In Reverse type is not integral!");

			//Change the order of bytes in read integers. Should only be used if the example is of integral type.
			const size_t size = sizeof(T);
			unsigned char ch[size];
			size_t i;
			for (i = 0; i < size; ++i)
				ch[i] = (input >> (i * 8)) & 255;
			T result = 0;

			for (i = 0; i < size; ++i)
				result += static_cast<T>(ch[size - i - 1]) << (i * 8);

			return result;
		}
	}
}