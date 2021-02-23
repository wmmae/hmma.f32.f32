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
__global__ void matvec_kernel(float* const y_ptr, const float* const a_ptr, const float* const x_ptr) {
	__shared__ float smem[N * N];
	fill_zero(smem, N * N);

	mtk::wmma::fragment_f32<nvcuda::wmma::matrix_a, N, N, N, T, nvcuda::wmma::col_major> frag_a;
	mtk::wmma::fragment_f32<nvcuda::wmma::matrix_b, N, N, N, T, nvcuda::wmma::col_major> frag_x;
	mtk::wmma::fragment_f32<nvcuda::wmma::accumulator, N, N, N, T> frag_y;

	// Load A
	copy_matrix(smem, N, a_ptr, N, N, N);
	mtk::wmma::load_matrix_sync(frag_a, smem, N);

	// Load X
	copy_matrix(smem, N, x_ptr, N, N, 1);
	mtk::wmma::fill_zero(frag_x);
	mtk::wmma::load_vector(frag_x, smem);

	// mma
	mtk::wmma::mma_sync(frag_y, frag_a, frag_x);

	// Store D
	mtk::wmma::store_vector(smem, frag_y, nvcuda::wmma::mem_col_major);
	copy_matrix(y_ptr, N, smem, N, N, 1);
}

template <unsigned N, class T, bool AddC>
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

	matvec_kernel<N, T><<<1, warp_size>>>(hY, hA, hX);

	cudaDeviceSynchronize();

	double max_error = 0.;
	for (unsigned n = 0; n < N; n++) {
		double cor_d = 0.;
		for (unsigned k = 0; k < N; k++) {
			cor_d += static_cast<double>(hA[k * N + n]) * static_cast<double>(hX[k]);
		}

		max_error = std::max(max_error, std::abs(cor_d - hY[n]));
	}

	std::printf("[N = %2u, T = %4s, AddC = %u] error = %e\n", N, get_type_name<T>().c_str(), (AddC ? 1 : 0), max_error);

	cudaFreeHost(hA);
	cudaFreeHost(hX);
	cudaFreeHost(hY);
}

int main() {
	test_matvec<32, half, true>();
	test_matvec<32, half, false>();
#ifdef TEST_TF32
	test_matvec<32, nvcuda::wmma::precision::tf32, true>();
	test_matvec<32, nvcuda::wmma::precision::tf32, false>();
#endif
}
