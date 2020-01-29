void kernel Transpose(global read_only const float* restrict A, global write_only float* restrict B, const int n, const int m)
{
	#define TILE_SIZE_X_2D 8
	#define TILE_SIZE_Y_2D 8

	const int tx = get_local_id(0);
	const int ty = get_local_id(1);
	
	const int i = get_global_id(0);
	const int j = get_global_id(1);
	
	__local float buffer[TILE_SIZE_X_2D * TILE_SIZE_Y_2D];
	
	if(i < n && j < m)
	{
		buffer[ty*TILE_SIZE_X_2D + tx] = A[j * n + i];
	}
	
	barrier(CLK_LOCAL_MEM_FENCE);
	
	if (i < n && j < m)
		B[j + i *m] = buffer[ty * TILE_SIZE_X_2D + tx];
}