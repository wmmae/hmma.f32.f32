#include <iostream>
#include <random>
#include "utils.hpp"
#include <wmma_extension/hmma_f32_f32.hpp>
#include <wmma_extension/hmma_f32_f32_no_cor.hpp>

template <unsigned N, class T, bool Cor>
__global__ void mma_kernel_abcd(float* const d_ptr, const float* const a_ptr, const float* const b_ptr, const float* const c_ptr) {
	constexpr unsigned LD = 2 * N + 1;
	__shared__ float smem[N * LD];
	mtk::test_utils::fill_zero(smem, N * LD);

	typename mtk::test_utils::select_fragemnt<Cor, nvcuda::wmma::matrix_a   , N, N, N, T, nvcuda::wmma::col_major>::type frag_a;
	typename mtk::test_utils::select_fragemnt<Cor, nvcuda::wmma::matrix_b   , N, N, N, T, nvcuda::wmma::col_major>::type frag_b;
	typename mtk::test_utils::select_fragemnt<Cor, nvcuda::wmma::accumulator, N, N, N, T>::type frag_c, frag_d;

	// Load A
	mtk::test_utils::copy_matrix(smem, LD, a_ptr, N, N, N);
	mtk::wmma::load_matrix_sync(frag_a, smem, LD);

	// Load B
	mtk::test_utils::copy_matrix(smem, LD, b_ptr, N, N, N);
	mtk::wmma::load_matrix_sync(frag_b, smem, LD);

	// Load C
	mtk::test_utils::copy_matrix(smem, LD, c_ptr, N, N, N);
	mtk::wmma::load_matrix_sync(frag_c, smem, LD, nvcuda::wmma::mem_col_major);

	// Fill D
	mtk::wmma::fill_fragment(frag_d, 0.0f);

	// mma
	mtk::wmma::mma_sync(frag_d, frag_a, frag_b, frag_c);

	// Store D
	mtk::wmma::store_matrix_sync(smem, frag_d, LD, nvcuda::wmma::mem_col_major);
	mtk::test_utils::copy_matrix(d_ptr, N, smem, LD, N, N);
}

template <unsigned N, class T, bool Cor>
__global__ void mma_kernel_abd(float* const d_ptr, const float* const a_ptr, const float* const b_ptr) {
	constexpr unsigned LD = 2 * N + 1;
	__shared__ float smem[N * LD];
	mtk::test_utils::fill_zero(smem, N * LD);

	typename mtk::test_utils::select_fragemnt<Cor, nvcuda::wmma::matrix_a   , N, N, N, T, nvcuda::wmma::col_major>::type frag_a;
	typename mtk::test_utils::select_fragemnt<Cor, nvcuda::wmma::matrix_b   , N, N, N, T, nvcuda::wmma::col_major>::type frag_b;
	typename mtk::test_utils::select_fragemnt<Cor, nvcuda::wmma::accumulator, N, N, N, T>::type frag_d;

	// Load A
	mtk::test_utils::copy_matrix(smem, LD, a_ptr, N, N, N);
	mtk::wmma::load_matrix_sync(frag_a, smem, LD);

	// Load B
	mtk::test_utils::copy_matrix(smem, LD, b_ptr, N, N, N);
	mtk::wmma::load_matrix_sync(frag_b, smem, LD);

	// mma
	mtk::wmma::mma_sync(frag_d, frag_a, frag_b);

	// Store D
	mtk::wmma::store_matrix_sync(smem, frag_d, LD, nvcuda::wmma::mem_col_major);
	mtk::test_utils::copy_matrix(d_ptr, N, smem, LD, N, N);
}

template <unsigned N, class T, bool AddC, bool Cor>
void test_mma() {
	float *hA, *hB, *hC, *hD;
	cudaMallocHost(&hA, N * N * sizeof(float));
	cudaMallocHost(&hB, N * N * sizeof(float));
	cudaMallocHost(&hC, N * N * sizeof(float));
	cudaMallocHost(&hD, N * N * sizeof(float));

	std::mt19937 mt(std::random_device{}());
	std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

	for (unsigned i = 0; i < N * N; i++) {
			hA[i] = dist(mt);
			hB[i] = dist(mt);
			hC[i] = dist(mt);
	}
	cudaDeviceSynchronize();

	if (AddC)
		mma_kernel_abcd<N, T, Cor><<<1, mtk::test_utils::warp_size>>>(hD, hA, hB, hC);
	else
		mma_kernel_abd<N, T, Cor><<<1, mtk::test_utils::warp_size>>>(hD, hA, hB);

	cudaDeviceSynchronize();

	double max_error = 0.;
	for (unsigned m = 0; m < N; m++) {
		for (unsigned n = 0; n < N; n++) {
			double cor_d = 0.;
			for (unsigned k = 0; k < N; k++) {
				cor_d += static_cast<double>(hA[k * N + m]) * static_cast<double>(hB[k + n * N]);
			}
			if (AddC)
				cor_d += hC[m + n * N];

			max_error = std::max(max_error, std::abs(cor_d - hD[m + n * N]));
		}
	}

	std::printf("[N = %2u, T = %4s, AddC = %u, Cor = %u] error = %e\n", N, mtk::test_utils::to_string<T>().c_str(), (AddC ? 1 : 0), (Cor ? 1 : 0), max_error);

	cudaFreeHost(hA);
	cudaFreeHost(hB);
	cudaFreeHost(hC);
	cudaFreeHost(hD);
}

int main() {
	test_mma<32, half, true , true >();
	test_mma<32, half, false, true >();
	test_mma<32, half, true , false>();
	test_mma<32, half, false, false>();
#ifdef TEST_TF32
	test_mma<32, nvcuda::wmma::precision::tf32, true , true >();
	test_mma<32, nvcuda::wmma::precision::tf32, false, true >();
	test_mma<32, nvcuda::wmma::precision::tf32, true , false>();
	test_mma<32, nvcuda::wmma::precision::tf32, false, false>();
#endif
}
