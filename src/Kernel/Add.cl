void kernel Add(global read_only const float* restrict A, global read_only const float* restrict B, global write_only float* restrict Y, const int n)
{
	const int i = get_global_id(0);

	if (i >= n)
		return;

	Y[i] = A[i] + B[i];
}

void kernel SubtractFromConst(global read_only const float* restrict A, global write_only float* restrict C, const float co, const int n)
{
	const int i = get_global_id(0);

	if (i >= n)
		return;

	C[i] = co - A[i];
}

void kernel SubtractFromConstGrad(global read_only const float* restrict A, global float* restrict C, const int n)
{
	const int i = get_global_id(0);

	if (i >= n)
		return;

	C[i] += -1.f * A[i];
}
