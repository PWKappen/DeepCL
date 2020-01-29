void kernel MaxPooling(global read_only const float* restrict A, global write_only float* restrict B, const int n, const int m, const int l, const int stride, const int padX, const int padY, const int size)
{
#define TILE_SIZE_X_2D 8
#define TILE_SIZE_Y_2D 8

#define SIZE 2
#define STRIDE 2

#define LOCAL_MEM_WIDTH (TILE_SIZE_X_2D*STRIDE+SIZE)
#define LOCAL_MEM_HEIGHT (TILE_SIZE_Y_2D * STRIDE + SIZE)

	
	const int i = get_global_id(0);
	const int j = get_global_id(1);
	const int k = get_global_id(2);

	const int wB = (n - size + padX) / stride + 1;
	const int hB = (m - size + padY) / stride + 1;

	int startX = i * STRIDE - padX;
	int startY = j * STRIDE - padY;
	int xMax = startX + size;
	int yMax = startY + size;

	const int posZ = k * m * n;

	if (startX < 0)
		startX = 0;
	if (startY < 0)
		startY = 0;
	if (xMax > n)
		xMax = n;
	if (yMax > m)
		yMax = m;

	float max = -FLT_MAX;

	if (k < l)
	{
		float tmp;

		for (int idxI = startX; idxI < xMax; ++idxI)
		{
			for (int idxJ = startY; idxJ < yMax; ++idxJ)
			{
				tmp = A[idxI + idxJ * n + posZ];
				if (max < tmp)
					max = tmp;
			}
		}
	}

	if (i < wB && j < hB && k < l)
		B[i + j * wB + k * wB * hB] = max;
}

void kernel MaxPoolingGrad(global read_only const float* restrict A, global read_only const float* restrict gradB, global float* restrict gradA, const int n, const int m, const int l, const int stride, const int padX, const int padY, const int size)
{
#define TILE_SIZE_X_2D 8
#define TILE_SIZE_Y_2D 8

#define SIZE 2
#define STRIDE 2

#define LOCAL_MEM_WIDTH (TILE_SIZE_X_2D*STRIDE+SIZE)
#define LOCAL_MEM_HEIGHT (TILE_SIZE_Y_2D * STRIDE + SIZE)

	
	const int i = get_global_id(0);
	const int j = get_global_id(1);
	const int k = get_global_id(2);

	const int wB = (n - size + padX) / stride + 1;
	const int hB = (m - size + padY) / stride + 1;

	int startX = i * STRIDE - padX;
	int startY = j * STRIDE - padY;
	int xMax = startX + size;
	int yMax = startY + size;

	const int posZ = k * m * n;

	if (startX < 0)
		startX = 0;
	if (startY < 0)
		startY = 0;
	if (xMax > n)
		xMax = n;
	if (yMax > m)
		yMax = m;

	int posX = startX;
	int posY = startY;

	if (k < l)
	{
		float max = -FLT_MAX;
		float tmp;

		for (int idxI = startX; idxI < xMax; ++idxI)
		{
			for (int idxJ = startY; idxJ < yMax; ++idxJ)
			{
				tmp = A[idxI + idxJ * n + posZ];
				if (max < tmp)
				{
					max = tmp;
					posX = idxI;
					posY = idxJ;
				}
			}
		}
	}

	if (i < wB && j < hB && k < l)
		gradA[posX + posY * n + k * n * m] += gradB[i + j * wB + k * wB * hB];
}