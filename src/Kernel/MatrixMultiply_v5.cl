void kernel MatrixMul(global read_only const float* restrict A, global read_only const float* restrict B, global write_only float* restrict C, const int hA, const int wB, const int wA)
{

//The size of a tile from the output that gets calculated by this work group
#define TILE_SIZE_X_2D 8
#define TILE_SIZE_Y_2D 8

//Each thread can potentially compute the result of multiple outputs
//This increases the total tile size.
#define WPTY 1
#define WPTX 1


#define TOTAL_WPT (WPTX * WPTY)

	const int tx = get_local_id(0);
	const int ty = get_local_id(1);
	const int lSizeX = get_local_size(0);
	const int lSizeY = get_local_size(1);
	
	const int gIdX = get_group_id(0);
	const int gIdY = get_group_id(1);

	const int i = get_global_id(0);
	const int j = get_global_id(1);
	
	//Registers for all results.
	float sum[TOTAL_WPT];

	for (int w = 0; w < TOTAL_WPT; ++w)
		sum[w] = 0;

	//Local memory for loading tiles into memory
	__local float localTileA[TILE_SIZE_X_2D * TILE_SIZE_Y_2D * WPTY];
	__local float localTileB[TILE_SIZE_X_2D * TILE_SIZE_Y_2D * WPTX];
	
	//Multiple tiles in the width of A are necessary.
	//The number of those tiles might not be a multiple of the work group.
	//Two parts are therefore used. One calculates the results for the complete loads
	//and the second calculates the results for the remaining elements.
	int reminder = wA%lSizeX;
	int numTiles =  (wA-reminder) / lSizeX;

	//Iterate over the x direction of the input.
	for(int t = 0; t < numTiles; ++t)
	{
		//Load the tile of A into local memory.
		for (int x = 0; x < WPTY; ++x)
		{
			//Calculate the y index of the loading position.
			int yIdx = gIdY * (WPTY*TILE_SIZE_Y_2D) + x*lSizeY + ty;
			if (yIdx < hA)
				localTileA[tx + (ty + lSizeY * x) * lSizeX] = A[t*lSizeX + tx + (yIdx)* wA];
			else
				localTileA[tx + (ty + lSizeY * x) * lSizeX] = 0;
		}
		
		//Load the tile of B into local memory.
		for (int x = 0; x < WPTX; ++x)
		{
			//Calculate the specific x index of the loading position.
			int xIdx = gIdX * (WPTX*TILE_SIZE_X_2D) + x*lSizeX + tx;
			if (xIdx < wB)
				localTileB[tx + lSizeX * x + ty * (lSizeX*WPTX)] = B[(t * lSizeY + ty)*wB + (xIdx)];
			else
				localTileB[tx + lSizeX * x + ty * (lSizeX*WPTX)] = 0;
		}

		//Compute the results of each output that is computed by this thread
		barrier(CLK_LOCAL_MEM_FENCE);
		
		for (int k = 0; k < lSizeX; ++k)
		{
			for (int wY = 0; wY < WPTY; ++wY)
			{
				for (int wX = 0; wX < WPTX; ++wX)
				{
					sum[wX + wY * WPTX] += localTileA[k + (ty + wY * lSizeY)*lSizeX] * localTileB[(tx + wX * lSizeX) + k*(lSizeX*WPTX)];
				}
			}
		}

		barrier(CLK_LOCAL_MEM_FENCE);
	}
	//The following behaves the same as before but this time the reminding pixels are used to calculate the result.
	for (int x = 0; x < WPTY; ++x)
	{
		int yIdx = gIdY * (WPTY*TILE_SIZE_Y_2D) + x*lSizeY + ty;
		if (yIdx < hA && tx < reminder)
			localTileA[tx + (ty + lSizeY * x) * lSizeX] = A[numTiles*lSizeX + tx + (yIdx)* wA];
		else
			localTileA[tx + (ty + lSizeY * x) * lSizeX] = 0;
	}
	for (int x = 0; x < WPTX; ++x)
	{
		int xIdx = gIdX * (WPTX*TILE_SIZE_X_2D) + x*lSizeX + tx;
		if (xIdx < wB && ty < reminder)
			localTileB[tx + lSizeX * x + ty * (lSizeX*WPTX)] = B[(numTiles * lSizeY + ty)*wB + (xIdx)];
		else
			localTileB[tx + lSizeX * x + ty * (lSizeX*WPTX)] = 0;
	}

	barrier(CLK_LOCAL_MEM_FENCE);

	for (int k = 0; k < reminder; ++k)
	{
		for (int wY = 0; wY < WPTY; ++wY)
		{
			for (int wX = 0; wX < WPTX; ++wX)
			{
				sum[wX + wY * WPTX] += localTileA[k + (ty + wY * lSizeY)*lSizeX] * localTileB[(tx + wX * lSizeX) + k*(lSizeX*WPTX)];
			}
		}
	}
	barrier(CLK_LOCAL_MEM_FENCE);

	//Store the results into local memory.
	for (int wY = 0; wY < WPTY; ++wY)
	{
		for (int wX = 0; wX < WPTX; ++wX)
		{
			int xIdx = gIdX * (WPTX*TILE_SIZE_X_2D) + wX*lSizeX + tx;
			int yIdx = gIdY * (WPTY*TILE_SIZE_Y_2D) + wY*lSizeY + ty;
			

			if (xIdx < wB && yIdx < hA)
				C[xIdx + (yIdx)* wB] = sum[wX + wY * WPTX];
		}
	}
}

//The same function as before but this time the result is added to the current content of the buffer.
//This is necessary for the backward pass.(Implicit copies)
void kernel MatrixMulAdd(global read_only const float* restrict A, global read_only const float* restrict B, global float* restrict C, const int hA, const int wB, const int wA)
{
#define TILE_SIZE_X_2D 8
#define TILE_SIZE_Y_2D 8

#define WPTY 1
#define WPTX 1
#define TOTAL_WPT (WPTX * WPTY)

	const int tx = get_local_id(0);
	const int ty = get_local_id(1);
	const int lSizeX = get_local_size(0);
	const int lSizeY = get_local_size(1);

	const int gIdX = get_group_id(0);
	const int gIdY = get_group_id(1);

	const int i = get_global_id(0);
	const int j = get_global_id(1);

	float sum[TOTAL_WPT];

	for (int w = 0; w < TOTAL_WPT; ++w)
		sum[w] = 0;

	__local float localTileA[TILE_SIZE_X_2D * TILE_SIZE_Y_2D * WPTY];
	__local float localTileB[TILE_SIZE_X_2D * TILE_SIZE_Y_2D * WPTX];

	int reminder = wA%lSizeX;
	int numTiles = (wA - reminder) / lSizeX;

	for (int t = 0; t < numTiles; ++t)
	{
		for (int x = 0; x < WPTY; ++x)
		{
			int yIdx = gIdY * (WPTY*TILE_SIZE_Y_2D) + x*lSizeY + ty;
			if (yIdx < hA)
				localTileA[tx + (ty + lSizeY * x) * lSizeX] = A[t*lSizeX + tx + (yIdx)* wA];
			else
				localTileA[tx + (ty + lSizeY * x) * lSizeX] = 0;
		}
		for (int x = 0; x < WPTX; ++x)
		{
			int xIdx = gIdX * (WPTX*TILE_SIZE_X_2D) + x*lSizeX + tx;
			if (xIdx < wB)
				localTileB[tx + lSizeX * x + ty * (lSizeX*WPTX)] = B[(t * lSizeY + ty)*wB + (xIdx)];
			else
				localTileB[tx + lSizeX * x + ty * (lSizeX*WPTX)] = 0;
		}

		barrier(CLK_LOCAL_MEM_FENCE);

		for (int k = 0; k < lSizeX; ++k)
		{
			for (int wY = 0; wY < WPTY; ++wY)
			{
				for (int wX = 0; wX < WPTX; ++wX)
				{
					sum[wX + wY * WPTX] += localTileA[k + (ty + wY * lSizeY)*lSizeX] * localTileB[(tx + wX * lSizeX) + k*(lSizeX*WPTX)];
				}
			}
		}

		barrier(CLK_LOCAL_MEM_FENCE);
	}

	for (int x = 0; x < WPTY; ++x)
	{
		int yIdx = gIdY * (WPTY*TILE_SIZE_Y_2D) + x*lSizeY + ty;
		if (yIdx < hA && tx < reminder)
			localTileA[tx + (ty + lSizeY * x) * lSizeX] = A[numTiles*lSizeX + tx + (yIdx)* wA];
		else
			localTileA[tx + (ty + lSizeY * x) * lSizeX] = 0;
	}
	for (int x = 0; x < WPTX; ++x)
	{
		int xIdx = gIdX * (WPTX*TILE_SIZE_X_2D) + x*lSizeX + tx;
		if (xIdx < wB && ty < reminder)
			localTileB[tx + lSizeX * x + ty * (lSizeX*WPTX)] = B[(numTiles * lSizeY + ty)*wB + (xIdx)];
		else
			localTileB[tx + lSizeX * x + ty * (lSizeX*WPTX)] = 0;
	}

	barrier(CLK_LOCAL_MEM_FENCE);

	for (int k = 0; k < reminder; ++k)
	{
		for (int wY = 0; wY < WPTY; ++wY)
		{
			for (int wX = 0; wX < WPTX; ++wX)
			{
				sum[wX + wY * WPTX] += localTileA[k + (ty + wY * lSizeY)*lSizeX] * localTileB[(tx + wX * lSizeX) + k*(lSizeX*WPTX)];
			}
		}
	}
	barrier(CLK_LOCAL_MEM_FENCE);

	for (int wY = 0; wY < WPTY; ++wY)
	{
		for (int wX = 0; wX < WPTX; ++wX)
		{
			int xIdx = gIdX * (WPTX*TILE_SIZE_X_2D) + wX*lSizeX + tx;
			int yIdx = gIdY * (WPTY*TILE_SIZE_Y_2D) + wY*lSizeY + ty;


			if (xIdx < wB && yIdx < hA)
				C[xIdx + (yIdx)* wB] += sum[wX + wY * WPTX];
		}
	}
}