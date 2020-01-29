void kernel RotateAndReorder(global read_only const float* restrict A, global write_only float* restrict B, const int wK, const int hK, const int dK, const int numK)
{
#define TILE_SIZE_X_2D 8
#define TILE_SIZE_Y_2D 8

	const int tx = get_local_id(0);
	const int ty = get_local_id(1);

	const int i = get_global_id(0);
	const int j = get_global_id(1);
	const int k = get_global_id(2);

	const int imgRow = k / dK;
	const int img = k - imgRow * dK;

	const int imgSize = wK * hK;
	const int numImages = dK * numK;

	const int x = (wK - 1) - i;
	const int y = (hK - 1 - j);

	__local float buffer[TILE_SIZE_X_2D * TILE_SIZE_Y_2D];

	if (i < wK && j < hK && k < numImages)
		buffer[ty*TILE_SIZE_X_2D + tx] = A[j * wK + i + k * imgSize];

	barrier(CLK_LOCAL_MEM_FENCE);

	if ( x < wK && x >= 0 && y < hK && y >= 0 && k < numImages)
		B[(wK - 1) - i + (hK - 1 - j)*wK + imgRow * imgSize + img * imgSize * numK] = buffer[ty * TILE_SIZE_X_2D + tx];
}