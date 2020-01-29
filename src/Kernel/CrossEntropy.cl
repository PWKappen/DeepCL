void kernel CrossEntropy(global read_only const float* restrict A, global read_only const int* restrict Y, global write_only float* restrict L, const int m, const int n)
{
	const int i = get_global_id(0);
	
	if(i >= n)
		return;
	
	L[i] = -log(A[i*m + Y[i]]);
}

void kernel CrossEntropyGrad(global read_only const float* restrict A, global read_only const int* restrict Y, global float* restrict gradL, const int m, const int n, const int batchN)
{
	#define TILE_SIZE_X_2D 8
	#define TILE_SIZE_Y_2D 8
	
	const int tx = get_local_id(0);
	const int ty = get_local_id(1);
	
	const int i = get_global_id(0);
	const int j = get_global_id(1);
	
	__local float localTileA[TILE_SIZE_X_2D * TILE_SIZE_Y_2D];
	__local float localTileB[TILE_SIZE_Y_2D];
	
	if(i >= m || j >= n)
		return;

	localTileA[tx + TILE_SIZE_X_2D * ty] = A[i + j * m];
	
	if(tx == 0)
		localTileB[ty] = Y[j];
	
	barrier(CLK_LOCAL_MEM_FENCE);
	
	gradL[i + j * m] += (localTileB[ty] == i ? -1.f/(localTileA[tx + TILE_SIZE_X_2D * ty] == 0 ? 1 : localTileA[tx + TILE_SIZE_X_2D * ty]) : 0)/batchN;
}

void kernel CrossEntropyTTime(global read_only const float* restrict A, global read_only const int* restrict Y, global write_only float* restrict L, const int m, const int n, const int time, const int offsetMemA)
{
	const int i = get_global_id(0);

	if (i >= n)
		return;

	for (int t = 0; t < time; ++t)
	{
		int y = Y[t+i * n] - 1;

		if (y == -1)
			break;
		L[i] += -log(A[i*m + y + t * offsetMemA]);
	}
}

void kernel CrossEntropyTTimeGrad(global read_only const float* restrict A, global read_only const int* restrict Y, global float* restrict gradL, const int m, const int n, const int batchN, const int time, const int offsetMemA)
{
#define TILE_SIZE_X_2D 8
#define TILE_SIZE_Y_2D 8

	const int tx = get_local_id(0);
	const int ty = get_local_id(1);

	const int i = get_global_id(0);
	const int j = get_global_id(1);

	__local float localTileA[TILE_SIZE_X_2D * TILE_SIZE_Y_2D];
	__local float localTileB[TILE_SIZE_Y_2D];

	if (i >= m || j >= n)
		return;

	localTileA[tx + TILE_SIZE_X_2D * ty] = A[i + j * m];

	if (tx == 0)
		localTileB[ty] = Y[j];

	barrier(CLK_LOCAL_MEM_FENCE);

	gradL[i + j * m] += (localTileB[ty] == i ? -1.f / (localTileA[tx + TILE_SIZE_X_2D * ty] == 0 ? 1 : localTileA[tx + TILE_SIZE_X_2D * ty]) : 0) / batchN;
}