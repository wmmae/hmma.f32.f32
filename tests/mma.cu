#include <iostream>
#include <random>
#include <wmmae/hmma_f32_f32_f32.hpp>

template <class T>
std::string get_type_name();
template <> std::string get_type_name<half>() {return "half";}
template <> std::string get_type_name<nvcuda::wmma::precision::tf32>() {return "tf32";}

constexpr unsigned warp_size = 32;

template <unsigned N, class T>
__global__ void mma_kernel(float* const d_ptr, const float* const a_ptr, const float* const b_ptr, const float* const c_ptr) {
	__shared__ float smem[N * N];

	mtk::wmma::fragment_f32<nvcuda::wmma::matrix_a, N, N, N, T, nvcuda::wmma::col_major> frag_a;
	mtk::wmma::fragment_f32<nvcuda::wmma::matrix_b, N, N, N, T, nvcuda::wmma::col_major> frag_b;
	mtk::wmma::fragment_f32<nvcuda::wmma::accumulator, N, N, N, T> frag_c, frag_d;

	// Load A
	for (unsigned i = 0; i < N * N; i += warp_size) {
		smem[i + threadIdx.x] = a_ptr[i + threadIdx.x];
	}
	mtk::wmma::load_matrix_sync(frag_a, smem, N);

	// Load B
	for (unsigned i = 0; i < N * N; i += warp_size) {
		smem[i + threadIdx.x] = b_ptr[i + threadIdx.x];
	}
	mtk::wmma::load_matrix_sync(frag_b, smem, N);

	// Load C
	for (unsigned i = 0; i < N * N; i += warp_size) {
		smem[i + threadIdx.x] = c_ptr[i + threadIdx.x];
	}
	mtk::wmma::load_matrix_sync(frag_c, smem, N, nvcuda::wmma::mem_col_major);

	// Fill D
	mtk::wmma::fill_fragment(frag_d, 0.0f);

	// mma
	mtk::wmma::mma_sync(frag_d, frag_a, frag_b, frag_c);

	// Store D
	mtk::wmma::store_matrix_sync(frag_d, smem, N, nvcuda::wmma::mem_col_major);
	for (unsigned i = 0; i < N * N; i += warp_size) {
		d_ptr[i + threadIdx.x] = smem[i + threadIdx.x];
	}
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

	std::printf("[%3u,%4s] error = %e\n", N, get_type_name<T>().c_str(), max_error);

	cudaFreeHost(hA);
	cudaFreeHost(hB);
	cudaFreeHost(hC);
	cudaFreeHost(hD);
}

int main() {
	test_mma<32, half>();
	test_mma<32, nvcuda::wmma::precision::tf32>();
}
