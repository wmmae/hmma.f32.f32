#include <iostream>
#include <wmma_extension/hmma_f32_f32.hpp>

constexpr unsigned warp_size = 32;

template <class T>
std::string get_type_name();
template <> std::string get_type_name<half>() {return "half";}
template <> std::string get_type_name<nvcuda::wmma::precision::tf32>() {return "tf32";}

template <unsigned N, class T>
__global__ void test_elementwise_kernel(float* const ptr) {
	__shared__ float smem[N * N];

	mtk::wmma::fragment_f32<nvcuda::wmma::accumulator, N, N, N, T> frag;
	mtk::wmma::fill_fragment(frag, 0.0f);

	for (unsigned i = 0; i < frag.num_elements; i++) {
		frag.x(i) = threadIdx.x * 100 + i;
	}

	mtk::wmma::store_matrix_sync(smem, frag, N, nvcuda::wmma::mem_col_major);

	for (unsigned i = 0; i < N * N; i += warp_size) {
		const auto index = i + threadIdx.x;
		ptr[index] = smem[index];
	}
}

template <unsigned N, class T>
void test_elementwise() {
	std::printf("[%s, N = %u, T = %s]\n", __func__, N, get_type_name<T>().c_str());
	float* hC;
	cudaMallocHost(&hC, N * N * sizeof(float));

	test_elementwise_kernel<N, T><<<1, warp_size>>>(hC);
	cudaDeviceSynchronize();

	for (unsigned i = 0; i < N; i++) {
		for (unsigned j = 0; j < N; j++) {
			std::printf("%e ", hC[i + j * N]);
		}
		std::printf("\n");
	}
}

int main() {
	test_elementwise<32, half>();
#ifdef TEST_TF32
	test_elementwise<32, nvcuda::wmma::precision::tf32>();
#endif
}
