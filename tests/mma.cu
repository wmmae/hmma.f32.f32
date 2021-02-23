#include <iostream>
#include <random>
#include <wmma_extension/hmma_f32_f32.hpp>

template <class T>
std::string get_type_name();
template <> std::string get_type_name<half>() {return "half";}
template <> std::string get_type_name<nvcuda::wmma::precision::tf32>() {return "tf32";}

constexpr unsigned warp_size = 32;

__device__ void copy_matrix(
		float* const dst, const unsigned ldd,
		const float* const src, const unsigned lds,
		const unsigned m, const unsigned n) {
	for (unsigned i = 0; i < m * n; i += warp_size) {
		const auto j = i + threadIdx.x;
		if (j >= m * n) return;
		const auto mm = j % m;
		const auto mn = j / m;
		dst[mm + mn * ldd] = src[mm + mn * lds];
	}
}

__device__ void fill_zero(float* const dst, const unsigned size) {
	for (unsigned i = 0; i < size; i += warp_size) {
		const auto j = i + threadIdx.x;
		if (j >= size) return;
		dst[j] = 0.0f;
	}
}

template <unsigned N, class T>
__global__ void mma_kernel(float* const d_ptr, const float* const a_ptr, const float* const b_ptr, const float* const c_ptr) {
	constexpr unsigned LD = 2 * N + 1;
	__shared__ float smem[N * LD];
	fill_zero(smem, N * LD);

	mtk::wmma::fragment_f32<nvcuda::wmma::matrix_a, N, N, N, T, nvcuda::wmma::col_major> frag_a;
	mtk::wmma::fragment_f32<nvcuda::wmma::matrix_b, N, N, N, T, nvcuda::wmma::col_major> frag_b;
	mtk::wmma::fragment_f32<nvcuda::wmma::accumulator, N, N, N, T> frag_c, frag_d;

	// Load A
	copy_matrix(smem, LD, a_ptr, N, N, N);
	mtk::wmma::load_matrix_sync(frag_a, smem, LD);

	// Load B
	copy_matrix(smem, LD, b_ptr, N, N, N);
	mtk::wmma::load_matrix_sync(frag_b, smem, LD);

	// Load C
	copy_matrix(smem, LD, c_ptr, N, N, N);
	mtk::wmma::load_matrix_sync(frag_c, smem, LD, nvcuda::wmma::mem_col_major);

	// Fill D
	mtk::wmma::fill_fragment(frag_d, 0.0f);

	// mma
	mtk::wmma::mma_sync(frag_d, frag_a, frag_b, frag_c);

	// Store D
	mtk::wmma::store_matrix_sync(smem, frag_d, LD, nvcuda::wmma::mem_col_major);
	copy_matrix(d_ptr, N, smem, LD, N, N);
}

template <unsigned N, class T>
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

	mma_kernel<N, T><<<1, warp_size>>>(hD, hA, hB, hC);

	cudaDeviceSynchronize();

	double max_error = 0.;
	for (unsigned m = 0; m < N; m++) {
		for (unsigned n = 0; n < N; n++) {
			double cor_d = 0.;
			for (unsigned k = 0; k < N; k++) {
				cor_d += static_cast<double>(hA[k * N + m]) * static_cast<double>(hB[k + n * N]);
			}
			cor_d += hC[m + n * N];

			max_error = std::max(max_error, std::abs(cor_d - hD[m + n * N]));
		}
	}

	std::printf("[N = %2u, T = %4s] error = %e\n", N, get_type_name<T>().c_str(), max_error);

	cudaFreeHost(hA);
	cudaFreeHost(hB);
	cudaFreeHost(hC);
	cudaFreeHost(hD);
}

int main() {
	test_mma<32, half>();
	test_mma<32, nvcuda::wmma::precision::tf32>();
}
