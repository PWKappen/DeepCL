void kernel Convolution(global read_only const float* restrict A, const global read_only float* restrict K, global write_only float* restrict C, const int wA, const int hA, const int wK, const int hK, const int dK, const int numK, const int pad, const int batchSize)
{

#define HALF_WIDTH (WIDTH_KERNEL/2)
#define HALF_HEIGHT (HEIGHT_KERNEL/2)
#define TILE_SIZE_X (TILE_WIDTH*STRIDE_X + HALF_WIDTH * 2)
#define TILE_SIZE_Y (TILE_HEIGHT*STRIDE_Y + HALF_HEIGHT * 2)

	const int kernelImageSize = wK * hK;
	const int kernelVolume = kernelImageSize*dK;

	const int imageOffset = wA * hA;
	const int batchOffset = imageOffset * dK;

	const int tx = get_local_id(0);
	const int ty = get_local_id(1);
	const int tz = get_local_id(2);

	const int i = get_global_id(0);
	const int j = get_global_id(1);
	const int k = get_global_id(2); 

	const int lSizeX = get_local_size(0);
	const int lSizeY = get_local_size(1);
	const int lSizeZ = get_local_size(2);

	const int groupIdX = get_group_id(0);
	const int groupIdY = get_group_id(1);
	const int groupIdZ = get_group_id(2);


	const int numTimes = (numK+lSizeZ-1)/lSizeZ;

	const int startZGroupSp = (groupIdZ % numTimes);
	const int batchIdx = (groupIdZ - startZGroupSp)/numTimes;
	const int startZ = startZGroupSp*lSizeZ;

	__local float kern[WIDTH_KERNEL * HEIGHT_KERNEL*TILE_DEPTH];
	__local float imageTile[TILE_SIZE_X * TILE_SIZE_Y];

	float sum = 0;

	const int outputXSize = ((wA - wK + 2 * pad) + STRIDE_X) / STRIDE_X;
	const int outputYSize = ((hA - hK + 2 * pad) + STRIDE_Y) / STRIDE_Y;

	const int startX = (groupIdX * lSizeX) * STRIDE_X - pad;
	const int startY = (groupIdY * lSizeY) * STRIDE_Y - pad;

	const int unrolledPos = tx + ty * lSizeX + tz *lSizeX*lSizeY;
	const int totalSize = lSizeX*lSizeY*lSizeZ;


	const int times = (TILE_SIZE_X * TILE_SIZE_Y + totalSize-1) / totalSize;

	bool loadKern = tx < wK && ty < hK && (startZ + tz) < numK;

	int posX = 0;
	int posY = 0;
	int tmp = 0;
	int currentY = 0;
	int currentX = 0;

	for (int l = 0; l <dK; ++l)
	{


		if (loadKern)
			kern[tx + ty * WIDTH_KERNEL + tz * kernelImageSize] = K[l * kernelImageSize + tx + ty * wK + (startZ + tz)*kernelVolume];


		for (int h = 0; h < times; ++h)
		{
			tmp = h * totalSize + unrolledPos;

			currentY = tmp / TILE_SIZE_X;
			currentX = tmp % TILE_SIZE_X;
			posX = currentX + startX;
			posY = currentY + startY;
			if (currentY < TILE_SIZE_Y)
			{

				if (posX >= 0 && posX < wA && posY >= 0 && posY < hA && batchIdx < batchSize)
				{
					imageTile[currentX + currentY * TILE_SIZE_X] = A[posX + posY * wA + l * imageOffset + batchIdx * batchOffset];
				}
				else
				{
					imageTile[currentX + currentY * TILE_SIZE_X] = 0;
				}
			}

			//if (groupIdX == 0 && groupIdY == 0 && groupIdZ == 1 &&  l == 0 &&tz==0)
			//	C[currentX + currentY * TILE_SIZE_X] = A[posX + posY * wA + l * imageOffset + batchIdx * batchOffset];

			barrier(CLK_LOCAL_MEM_FENCE);

			for (int m = 0; m < HEIGHT_KERNEL; ++m)
			{
#pragma unroll WIDTH_KERNEL
				for (int n = 0; n < WIDTH_KERNEL; ++n)
				{
					sum += imageTile[tx * STRIDE_X + n + m * TILE_SIZE_X + ty *TILE_SIZE_X*STRIDE_Y] * kern[m * WIDTH_KERNEL + n + tz *kernelImageSize];
				}
			}
		}
	}

	if (i < outputXSize && j < outputYSize && (startZ+tz)<numK && batchIdx < batchSize)
		C[i + j * outputXSize + (startZ +tz) * outputXSize*outputYSize + batchIdx * outputXSize*outputYSize*numK] = sum;
}


void kernel ConvolutionAdd(const read_only global float* restrict A, const global read_only float* restrict K, global float* restrict C, const int wA, const int hA, const int wK, const int hK, const int dK, const int numK, const int pad, const int batchSize)
{

#define WIDTH_KERNEL 5
#define HEIGHT_KERNEL 5

#define TILE_WIDTH 8
#define TILE_HEIGHT 8

#define ROW_SIZE (TILE_WIDTH + (WIDTH_KERNEL-1))

#define MEM_SIZE ((TILE_HEIGHT * TILE_WIDTH)/ROW_SIZE)

#define ROWS_PER_LOAD 5//(MEM_SIZE * WIDTH_KERNEL)

#define ROWS (ROWS_PER_LOAD / WIDTH_KERNEL)
#define TOTAL_KERNEL_SIZE (WIDTH_KERNEL * HEIGHT_KERNEL)


	const int imageSize = wA * hA;
	const int kernelImageSize = wK * hK;
	const int kernelVolume = kernelImageSize*dK;

	int imageOffset = imageSize;
	int batchOffset = imageSize * dK;

	const int tx = get_local_id(0);
	const int ty = get_local_id(1);

	const int i = get_global_id(0);
	const int j = get_global_id(1);
	const int k = get_global_id(2);

	const int lSizeX = get_local_size(0);
	const int lSizeY = get_local_size(1);

	const int groupIdX = get_group_id(0);
	const int groupIdY = get_group_id(1);

	__local float kern[TILE_WIDTH * TILE_HEIGHT];
	__local float imageTile[ROW_SIZE * ROWS];

	float sum = 0;

	const int outputXSize = (wA - wK + 2 * pad) + 1;
	const int outputYSize = (hA - hK + 2 * pad) + 1;

	const int imgRemainder = (wA + lSizeX - 1) / lSizeX;
	const int startX = (groupIdX%imgRemainder) * lSizeX - pad;
	const int startY = (groupIdX / imgRemainder) - pad;

	const int unrolledPos = tx + ty * lSizeX;

	const int posX = ((unrolledPos)) % (ROW_SIZE);
	const int posY = ((unrolledPos)-posX) / (ROW_SIZE);

	int imgRow;
	int img;
	int imgCol = startX + posX;
	int tmp;

	for (int l = 0; l <kernelVolume; l += ROWS_PER_LOAD)
	{
		if (j < numK && l + tx < kernelVolume)
			kern[unrolledPos] = K[j * kernelVolume + l + tx];

		tmp = posY + l / WIDTH_KERNEL;
		img = (tmp) / WIDTH_KERNEL;
		//imgRow = startY + (posY + l/WIDTH_KERNEL)%WIDTH_KERNEL;
		imgRow = startY + tmp - img * WIDTH_KERNEL;

		if (unrolledPos < (ROW_SIZE * ROWS))
		{
			if (imgCol >= 0 && imgCol < wA && imgRow >= 0 && imgRow < hA && k < batchSize)
				imageTile[unrolledPos] = A[imgCol + img * imageOffset + imgRow * wA + batchOffset * k];
			else
				imageTile[unrolledPos] = 0;
		}

		barrier(CLK_LOCAL_MEM_FENCE);

		for (int m = 0; m < ROWS; ++m)
		{
#pragma unroll WIDTH_KERNEL
			for (int n = 0; n < WIDTH_KERNEL; ++n)
			{
				sum += imageTile[tx + n + m * ROW_SIZE] * kern[m * WIDTH_KERNEL + n + ty * lSizeX];
			}
		}
	}

	if (j < numK && tx + startX + pad < outputXSize && startY + pad < outputYSize && k < batchSize)
		C[tx + startX + pad + (startY + pad + outputYSize * (j + numK * k))*outputXSize] += sum;
}

void kernel ConvolutionWeightGrad(const read_only global float* restrict A, const read_only global float* restrict K, global float* restrict C, const int wA, const int hA, const int wK, const int hK, const int dK, const int numK, const int pad, const int batchSize)
{
#define OUTPUT_WIDTH 5
#define OUTPUT_HEIGHT 5

#define TILE_WIDTH 8
#define TILE_HEIGHT 8

#define ROWS 3

#define ROW_SIZE (TILE_WIDTH + OUTPUT_WIDTH - 1)
#define LOCAL_LOAD_THREADS (ROWS * ROW_SIZE)

	const int tx = get_local_id(0);
	const int ty = get_local_id(1);

	const int i = get_global_id(0);
	const int j = get_global_id(1);

	const int lSizeX = get_local_size(0);
	const int lSizeY = get_local_size(1);

	const int gSizeX = get_global_size(0);
	const int gSizeY = get_global_size(1);

	const int groupIdX = get_group_id(0);
	const int groupIdY = get_group_id(1);

	const int unrolledLocal = tx + ty * lSizeX;

	const int groupGlobalX = lSizeX * groupIdX;

	const int gStartZ = i / (OUTPUT_WIDTH * OUTPUT_HEIGHT);
	const int gStartY = (i - gStartZ * (OUTPUT_WIDTH * OUTPUT_HEIGHT)) / OUTPUT_WIDTH;
	const int gStartX = (i - gStartY * (OUTPUT_WIDTH)-gStartZ * (OUTPUT_WIDTH * OUTPUT_HEIGHT));

	const int groupStartZ = groupGlobalX / (OUTPUT_WIDTH * OUTPUT_HEIGHT);
	const int groupStartY = (groupGlobalX - groupStartZ * (OUTPUT_WIDTH * OUTPUT_HEIGHT)) / OUTPUT_WIDTH;
	const int groupStartX = (groupGlobalX - groupStartY * (OUTPUT_WIDTH)-groupStartZ * (OUTPUT_WIDTH * OUTPUT_HEIGHT));

	const int nextStartX = gStartX / OUTPUT_WIDTH;

	const int diffZ = gStartZ - groupStartZ;
	const int diffY = (gStartY - groupStartY + diffZ * OUTPUT_HEIGHT);
	const int diffX = (diffY) > 0 ? gStartX - (nextStartX*	OUTPUT_WIDTH) : (gStartX - groupStartX);


	const int kernelVolume = wK * hK;

	__local float kern[TILE_WIDTH * TILE_HEIGHT];
	__local float imageTile[LOCAL_LOAD_THREADS];

	float sum = 0;

	const bool load = (diffX == 0 || diffX > 0 && ty == (lSizeY - 1));
	const int loadLocalPos = (diffY)* ROW_SIZE + ty + gStartX;

	const int tmpPosX = ty + gStartX - pad;
	const int tmpPosY = gStartY - pad;
	const int totalPosZ = gStartZ * wA * hA;

	const int totalWidth = wA + 2 * pad;
	const int totalHeight = hA + 2 * pad;
	const int mod = wK + 2 * pad;

	int a;
	int x;
	int y;

	const int imgVolumeSize = wA*hA*dK;

	int batchPos;
	int batchPosKernel;

	const int complete = (wK / TILE_WIDTH) * TILE_WIDTH;
	const int remainder = wK - complete;

	int realY;

	for (int l = 0; l < batchSize; ++l)
	{
		batchPos = l * imgVolumeSize;
		batchPosKernel = l * kernelVolume * numK;

		for (int m = 0; m < hK; ++m)
		{
			realY = m * wK;

			for (int n = 0; n < complete; n += TILE_WIDTH)
			{
				if (j < numK)
					kern[unrolledLocal] = K[j * kernelVolume + realY + n + tx + batchPosKernel];

				x = n + tmpPosX;
				y = m + tmpPosY;

				if (load)
				{
					if (x >= 0 && x < wA && y >= 0 && y < hA && gStartZ < dK)
						imageTile[loadLocalPos] = A[x + y * wA + totalPosZ + batchPos];
					else
						imageTile[loadLocalPos] = 0;
				}

				barrier(CLK_LOCAL_MEM_FENCE);


				for (int o = 0; o < TILE_WIDTH; ++o)
					sum += imageTile[gStartX + o + diffY * ROW_SIZE] * kern[o + ty * lSizeX];
			}

			if (j < numK && tx < remainder)
				kern[unrolledLocal] = K[j * kernelVolume + realY + complete + tx + batchPosKernel];

			x = complete + tmpPosX;
			y = m + tmpPosY;

			if (load)
			{
				if (x >= 0 && x < wA && y >= 0 && y < hA && gStartZ < dK)
					imageTile[loadLocalPos] = A[x + y * wA + totalPosZ + batchPos];
				else
					imageTile[loadLocalPos] = 0;
			}

			barrier(CLK_LOCAL_MEM_FENCE);

			for (int o = 0; o < remainder; ++o)
				sum += imageTile[gStartX + o + diffY * ROW_SIZE] * kern[o + ty * lSizeX];
		}
	}

	if (gStartZ < dK && j < numK)
		C[gStartX + gStartY * OUTPUT_WIDTH + gStartZ * OUTPUT_WIDTH * OUTPUT_HEIGHT + j * dK * OUTPUT_WIDTH * OUTPUT_HEIGHT] += sum;
}