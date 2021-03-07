#include <iostream>
#include <wmma_extension/hmma_f32_f32.hpp>
#include "utils.hpp"

#ifdef WMMAE_USE_NVCUDA_NAMESPACE
namespace f32_namespace = nvcuda;
#else
namespace f32_namespace = mtk;
#endif

constexpr unsigned warp_size = 32;

template <unsigned N, class T>
__global__ void test_elementwise_kernel(float* const ptr) {
	__shared__ float smem[N * N];

	f32_namespace::wmma::fragment_f32<nvcuda::wmma::accumulator, N, N, N, T> frag;
	f32_namespace::wmma::fill_fragment(frag, 0.0f);

	for (unsigned i = 0; i < frag.num_elements; i++) {
		frag.x(i) = threadIdx.x * 100 + i;
	}

	f32_namespace::wmma::store_matrix_sync(smem, frag, N, nvcuda::wmma::mem_col_major);

	for (unsigned i = 0; i < N * N; i += warp_size) {
		const auto index = i + threadIdx.x;
		ptr[index] = smem[index];
	}
}

template <unsigned N, class T>
void test_elementwise() {
	std::printf("[%s, N = %u, T = %s]\n", __func__, N, mtk::test_utils::to_string<T>().c_str());
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
