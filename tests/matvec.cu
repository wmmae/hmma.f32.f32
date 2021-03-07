#include <iostream>
#include <random>
#include "utils.hpp"

#ifdef WMMAE_USE_NVCUDA_NAMESPACE
namespace f32_namespace = nvcuda;
#else
namespace f32_namespace = mtk;
#endif

template <unsigned N, class T, bool Cor>
__global__ void matvec_kernel(float* const y_ptr, const float* const a_ptr, const float* const x_ptr) {
	__shared__ float smem[N * N];
	mtk::test_utils::fill_zero(smem, N * N);

	typename mtk::test_utils::select_fragemnt<Cor, nvcuda::wmma::matrix_a   , N, N, N, T, nvcuda::wmma::col_major>::type frag_a;
	typename mtk::test_utils::select_fragemnt<Cor, nvcuda::wmma::matrix_b   , N, N, N, T, nvcuda::wmma::col_major>::type frag_x;
	typename mtk::test_utils::select_fragemnt<Cor, nvcuda::wmma::accumulator, N, N, N, T>::type frag_y;

	// Load A
	mtk::test_utils::copy_matrix(smem, N, a_ptr, N, N, N);
	f32_namespace::wmma::load_matrix_sync(frag_a, smem, N);

	// Load X
	mtk::test_utils::copy_matrix(smem, N, x_ptr, N, N, 1);
	f32_namespace::wmma::fill_zero(frag_x);
	f32_namespace::wmma::load_vector(frag_x, smem);

	// mma
	f32_namespace::wmma::mma_sync(frag_y, frag_a, frag_x);

	// Store D
	f32_namespace::wmma::store_vector(smem, frag_y, nvcuda::wmma::mem_col_major);
	mtk::test_utils::copy_matrix(y_ptr, N, smem, N, N, 1);
}

template <unsigned N, class T, bool AddC, bool Cor>
void test_matvec() {
	float *hX, *hY, *hA;
	cudaMallocHost(&hX, N     * sizeof(float));
	cudaMallocHost(&hY, N     * sizeof(float));
	cudaMallocHost(&hA, N * N * sizeof(float));

	std::mt19937 mt(std::random_device{}());
	std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

	for (unsigned i = 0; i < N * N; i++) {
			hA[i] = dist(mt);
	}
	for (unsigned i = 0; i < N; i++) {
			hX[i] = dist(mt);
	}
	cudaDeviceSynchronize();

	matvec_kernel<N, T, Cor><<<1, mtk::test_utils::warp_size>>>(hY, hA, hX);

	cudaDeviceSynchronize();

	double max_error = 0.;
	for (unsigned n = 0; n < N; n++) {
		double cor_d = 0.;
		for (unsigned k = 0; k < N; k++) {
			cor_d += static_cast<double>(hA[k * N + n]) * static_cast<double>(hX[k]);
		}

		max_error = std::max(max_error, std::abs(cor_d - hY[n]));
	}

	std::printf("[N = %2u, T = %4s, AddC = %u, Cor = %u] error = %e\n", N, mtk::test_utils::to_string<T>().c_str(), (AddC ? 1 : 0), (Cor ? 1 : 0), max_error);

	cudaFreeHost(hA);
	cudaFreeHost(hX);
	cudaFreeHost(hY);
}

int main() {
	test_matvec<32, half, true , true >();
	test_matvec<32, half, false, true >();
	test_matvec<32, half, true , false>();
	test_matvec<32, half, false, false>();
#ifdef TEST_TF32
	test_matvec<32, nvcuda::wmma::precision::tf32, true , true >();
	test_matvec<32, nvcuda::wmma::precision::tf32, false, true >();
	test_matvec<32, nvcuda::wmma::precision::tf32, true , false>();
	test_matvec<32, nvcuda::wmma::precision::tf32, false, false>();
#endif
}
