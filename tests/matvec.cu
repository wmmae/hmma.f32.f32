#include <iostream>
#include <random>
#include "utils.hpp"

#ifdef WMMAE_USE_NVCUDA_NAMESPACE
namespace f32_namespace = nvcuda;
#else
namespace f32_namespace = mtk;
#endif

template <class T, class ErrorCorrection>
constexpr double error_threshold = 0.0;
template <>
constexpr double error_threshold<half                         , mtk::wmma::op_with_error_correction   > = 1e-5;
template <>
constexpr double error_threshold<nvcuda::wmma::precision::tf32, mtk::wmma::op_with_error_correction   > = 1e-5;
template <>
constexpr double error_threshold<half                         , mtk::wmma::op_without_error_correction> = 1e-2;
template <>
constexpr double error_threshold<nvcuda::wmma::precision::tf32, mtk::wmma::op_without_error_correction> = 1e-2;

template <unsigned N, class T, class Policy>
__global__ void matvec_kernel(float* const y_ptr, const float* const a_ptr, const float* const x_ptr) {
	__shared__ float smem[N * N];
	mtk::test_utils::fill_zero(smem, N * N);

	f32_namespace::wmma::fragment_f32<nvcuda::wmma::matrix_a   , N, N, N, T, nvcuda::wmma::col_major, Policy> frag_a;
	f32_namespace::wmma::fragment_f32<nvcuda::wmma::matrix_b   , N, N, N, T, nvcuda::wmma::col_major, Policy> frag_x;
	f32_namespace::wmma::fragment_f32<nvcuda::wmma::accumulator, N, N, N, T, void                   , Policy> frag_y;
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

template <unsigned N, class T, class Policy>
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

	matvec_kernel<N, T, Policy><<<1, mtk::test_utils::warp_size>>>(hY, hA, hX);

	cudaDeviceSynchronize();

	double max_error = 0.;
	for (unsigned n = 0; n < N; n++) {
		double cor_d = 0.;
		for (unsigned k = 0; k < N; k++) {
			cor_d += static_cast<double>(hA[k * N + n]) * static_cast<double>(hX[k]);
		}

		max_error = std::max(max_error, std::abs(cor_d - hY[n]));
	}

	std::printf(
			"[Type:%5s, N:%3u, Policy<%7s,%9s,%2u,%2u,%2u>] max_error: %e (%6s)\n",
			mtk::test_utils::to_string<T>().c_str(),
			N,
			std::is_same<typename Policy::op, mtk::wmma::op_wmma>::value ? "op_wmma" : "op_mma",
			std::is_same<typename Policy::error_correction, mtk::wmma::op_with_error_correction>::value ? "{w/ ec}" : "{w/o ec}",
			Policy::m,
			Policy::n,
			Policy::k,
			max_error,
			(max_error < error_threshold<T, typename Policy::error_correction> ? "PASSED" : "FAILED")
			);

	cudaFreeHost(hA);
	cudaFreeHost(hX);
	cudaFreeHost(hY);
}

int main() {
	// wmma FP16 test
	test_matvec<32, half, typename mtk::wmma::detail::default_policy<half, mtk::wmma::op_with_error_correction   , mtk::wmma::op_wmma>::type>();
	test_matvec<32, half, typename mtk::wmma::detail::default_policy<half, mtk::wmma::op_without_error_correction, mtk::wmma::op_wmma>::type>();

#ifdef TEST_TF32
	// wmma TF32 test
	test_matvec<32, nvcuda::wmma::precision::tf32, typename mtk::wmma::detail::default_policy<nvcuda::wmma::precision::tf32, mtk::wmma::op_with_error_correction   , mtk::wmma::op_wmma>::type>();
	test_matvec<32, nvcuda::wmma::precision::tf32, typename mtk::wmma::detail::default_policy<nvcuda::wmma::precision::tf32, mtk::wmma::op_without_error_correction, mtk::wmma::op_wmma>::type>();
#endif
}
