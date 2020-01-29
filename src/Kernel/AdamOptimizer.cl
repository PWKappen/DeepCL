void kernel Adam(global read_only const float* restrict A, global float* restrict m, global float* restrict v, global float* restrict L, const float alpha, const float beta1, const float beta2, const float epsilon, const int size, const int t)
{
	const int i = get_global_id(0);

	if (i >= size)
		return;

	float a = A[i];
	float mt = beta1 * m[i] + (1.f - beta1) * a;
	float vt = beta2 * v[i] + (1.f - beta2) * a * a;
	float m_ = mt / (1.f - pown(beta1, t));
	float v_ = vt / (1.f - pown(beta2, t));
	float l = L[i];

	barrier(CLK_LOCAL_MEM_FENCE);

	L[i] = l - (alpha * m_ / (sqrt(v_) + epsilon));
	m[i] = mt;
	v[i] = vt;
}