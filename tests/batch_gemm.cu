#include <iostream>
#include <chrono>
#include <wmma_extension/hmma_f32_f32.hpp>

namespace {
constexpr unsigned warp_size = 32;

// SMEM_M * SMEM_N must be larger than or equal to BLOCK_SIZE
template <unsigned SMEM_M, unsigned SMEM_N, unsigned BLOCK_SIZE>
__device__ void dmem2smem(
		float* const dst_smem,
		const unsigned m, const unsigned n,
		const float* const src_dmem, const unsigned ld
		) {
	if (m == SMEM_M && n == SMEM_N) {
		for (unsigned i = 0; i < SMEM_M * SMEM_N; i += BLOCK_SIZE) {
			const auto j = i + threadIdx.x;
			const auto j_m = j % SMEM_M;
			const auto j_n = j / SMEM_M;

			dst_smem[j] = src_dmem[j_m + j_n * ld];
		}
	} else {
		for (unsigned i = 0; i < SMEM_M * SMEM_N; i += BLOCK_SIZE) {
			const auto j = i + threadIdx.x;
			const auto j_m = j % SMEM_M;
			const auto j_n = j / SMEM_M;

			float v = 0.f;
			if (j_m < m && j_n < n) {
				v = src_dmem[j_m + j_n * ld];
			}

			dst_smem[j] = v;
		}
	}
}

// SMEM_M * SMEM_N must be larger than or equal to BLOCK_SIZE
template <unsigned SMEM_M, unsigned SMEM_N, unsigned BLOCK_SIZE>
__device__ void smem2dmem(
		float* const dst_dmem, const unsigned ld,
		const unsigned m, const unsigned n,
		const float* const src_smem,
		const float alpha, const float beta
		) {
	if (beta == 0.f) {
		if (m == SMEM_M && n == SMEM_N) {
			for (unsigned i = 0; i < SMEM_M * SMEM_N; i += BLOCK_SIZE) {
				const auto j = i + threadIdx.x;
				const auto j_m = j % SMEM_M;
				const auto j_n = j / SMEM_M;

				dst_dmem[j_m + j_n * ld] = alpha * src_smem[j];
			}
		} else {
			for (unsigned i = 0; i < SMEM_M * SMEM_N; i += BLOCK_SIZE) {
				const auto j = i + threadIdx.x;
				const auto j_m = j % SMEM_M;
				const auto j_n = j / SMEM_M;

				if (j_m < m && j_n < n) {
					dst_dmem[j_m + j_n * ld] = alpha * src_smem[j];
				}
			}
		}
	} else {
		// beta is not zero
		if (m == SMEM_M && n == SMEM_N) {
			for (unsigned i = 0; i < SMEM_M * SMEM_N; i += BLOCK_SIZE) {
				const auto j = i + threadIdx.x;
				const auto j_m = j % SMEM_M;
				const auto j_n = j / SMEM_M;

				const auto dmem_offset = j_m + j_n * ld;
				dst_dmem[dmem_offset] = alpha * src_smem[j] + beta * dst_dmem[dmem_offset];
			}
		} else {
			for (unsigned i = 0; i < SMEM_M * SMEM_N; i += BLOCK_SIZE) {
				const auto j = i + threadIdx.x;
				const auto j_m = j % SMEM_M;
				const auto j_n = j / SMEM_M;

				if (j_m < m && j_n < n) {
					const auto dmem_offset = j_m + j_n * ld;
					dst_dmem[dmem_offset] = alpha * src_smem[j] + beta * dst_dmem[dmem_offset];
				}
			}
		}
	}
}

// SMEM_M * SMEM_N must be larger than or equal to BLOCK_SIZE
template <unsigned SMEM_M, unsigned SMEM_N, unsigned BLOCK_SIZE>
__device__ void fill_zero(
		float* const dst_smem
		) {
	for (unsigned i = 0; i < SMEM_M * SMEM_N; i += BLOCK_SIZE) {
		const auto j = i + threadIdx.x;
		dst_smem[j] = 0.f;
	}
}

// This kernel function computes batched matrix-matrix multiplication
// A needs to be row major, and B needst to be col major
template <
	unsigned SMEM_M,
	unsigned SMEM_N,
	unsigned SMEM_K,
	unsigned WARP_M,
	unsigned WARP_N,
	unsigned WARP_K,
	unsigned BLOCK_SIZE,
	class FRAGMENT_T,
	class TC_Policy>
__global__ void bgemm_kernel(
		const unsigned m,
		const unsigned n,
		const unsigned k,
		const float alpha,
		const float* const* const a_ptr, const unsigned lda,
		const float* const* const b_ptr, const unsigned ldb,
		const float beta,
		float* const* const c_ptr, const unsigned ldc
		) {
	// Sharedm memory
	extern __shared__ float smem[];
	float* const a_smem = smem;
	float* const b_smem = a_smem + SMEM_M * SMEM_K;
	float* const c_smem = b_smem + SMEM_K * SMEM_N;

	// Device memory
	float* const c_dmem = c_ptr[blockIdx.x];
	const float* const a_dmem = a_ptr[blockIdx.x];
	const float* const b_dmem = b_ptr[blockIdx.x];

	for (unsigned bm = 0; bm < m; bm += SMEM_M) {
		for (unsigned bn = 0; bn < n; bn += SMEM_N) {
			fill_zero<SMEM_M, SMEM_N, BLOCK_SIZE>(c_smem);
			for (unsigned bk = 0; bk < k; bk += SMEM_K) {
				// Load A from device memory to shared memory
				const auto real_bm = min(SMEM_M, m - bm);
				const auto real_bk = min(SMEM_K, k - bk);
				const auto a_dmem_offset = bm * lda + bk;
				dmem2smem<SMEM_M, SMEM_K, BLOCK_SIZE>(a_smem, real_bm, real_bk, a_dmem + a_dmem_offset, lda);

				// Load B from global memory to shared memory
				const auto real_bn = min(SMEM_N, n - bn);
				const auto b_dmem_offset = bn * ldb + bk;
				dmem2smem<SMEM_K, SMEM_N, BLOCK_SIZE>(b_smem, real_bk, real_bn, b_dmem + b_dmem_offset, ldb);

				__syncthreads();

				for (unsigned w = 0; w < (SMEM_M * SMEM_N / (WARP_M * WARP_N)); w += BLOCK_SIZE / warp_size) {
					const auto wi = w + threadIdx.x / warp_size;
					const auto wi_m = (wi % (SMEM_M / WARP_M)) * WARP_M;
					const auto wi_n = (wi / (SMEM_M / WARP_M)) * WARP_N;

					mtk::wmma::mma_f32::fragment<nvcuda::wmma::accumulator, WARP_M, WARP_N, WARP_K, FRAGMENT_T, void, TC_Policy> frag_c;
					const auto c_smem_offset = wi_m + wi_n * SMEM_M;
					mtk::wmma::mma_f32::load_matrix_sync(frag_c, c_smem + c_smem_offset, SMEM_M, nvcuda::wmma::mem_col_major);
					for (unsigned wi_k = 0; wi_k < SMEM_K; wi_k += WARP_K) {
						// Load A
						mtk::wmma::mma_f32::fragment<nvcuda::wmma::matrix_a, WARP_M, WARP_N, WARP_K, FRAGMENT_T, nvcuda::wmma::row_major, TC_Policy> frag_a;
						const auto a_smem_offset = wi_m * SMEM_K + wi_k;
						mtk::wmma::mma_f32::load_matrix_sync(frag_a, a_smem + a_smem_offset, SMEM_K);

						// Load B
						mtk::wmma::mma_f32::fragment<nvcuda::wmma::matrix_b, WARP_M, WARP_N, WARP_K, FRAGMENT_T, nvcuda::wmma::col_major, TC_Policy> frag_b;
						const auto b_smem_offset = wi_n * SMEM_K + wi_k;
						mtk::wmma::mma_f32::load_matrix_sync(frag_b, b_smem + b_smem_offset, SMEM_K);

						// mma
						mtk::wmma::mma_f32::mma_sync(frag_c, frag_a, frag_b, frag_c);
					}
					mtk::wmma::mma_f32::store_matrix_sync(c_smem + c_smem_offset, frag_c, SMEM_M, nvcuda::wmma::mem_col_major);
				}
				__syncthreads();
			} // loop bk
			const auto real_bm = min(SMEM_M, m - bm);
			const auto real_bn = min(SMEM_N, n - bn);
			const auto c_dmem_offset = bm + bn * ldc;
			smem2dmem<SMEM_M, SMEM_N, BLOCK_SIZE>(c_dmem + c_dmem_offset, ldc, real_bm, real_bn, c_smem, alpha, beta);
		} // loop bn
	} // loop bm
}

template <
	unsigned SMEM_M,
	unsigned SMEM_N,
	unsigned SMEM_K,
	unsigned WARP_M,
	unsigned WARP_N,
	unsigned WARP_K,
	unsigned BLOCK_SIZE,
	class FRAGMENT_T,
	class TC_Policy>
void bgemm(
		const unsigned m,
		const unsigned n,
		const unsigned k,
		const float alpha,
		const float* const* const a_ptr, const unsigned lda,
		const float* const* const b_ptr, const unsigned ldb,
		const float beta,
		float* const* const c_ptr, const unsigned ldc,
		const unsigned batch_size
		) {
	// Set shared memory size
	const auto shared_memory_size = (SMEM_M * SMEM_K + SMEM_K * SMEM_N + SMEM_M * SMEM_N) * sizeof(float);
	cudaFuncSetAttribute(&(bgemm_kernel<SMEM_M, SMEM_N, SMEM_K, WARP_M, WARP_N, WARP_K, BLOCK_SIZE, FRAGMENT_T, TC_Policy>), cudaFuncAttributeMaxDynamicSharedMemorySize, shared_memory_size);

	// Launch
	bgemm_kernel<SMEM_M, SMEM_N, SMEM_K, WARP_M, WARP_N, WARP_K, BLOCK_SIZE, FRAGMENT_T, TC_Policy><<<batch_size, BLOCK_SIZE, shared_memory_size>>>(
			m, n, k,
			alpha,
			a_ptr, lda,
			b_ptr, ldb,
			beta,
			c_ptr, ldc
			);
}

template <
	unsigned SMEM_M,
	unsigned SMEM_N,
	unsigned SMEM_K,
	unsigned WARP_M,
	unsigned WARP_N,
	unsigned WARP_K,
	unsigned BLOCK_SIZE>
void test_batched_sgemm(
		const unsigned m,
		const unsigned n,
		const unsigned k,
		const unsigned batch_size
		) {
	std::printf("!-- %s\n", __func__);
	using FRAGMENT_T = half;
	using TC_Policy = mtk::wmma::mma_f32::detail::default_policy<FRAGMENT_T, mtk::wmma::mma_f32::op_with_error_correction, mtk::wmma::mma_f32::op_mma>::type;

	float **d_a_ptr_array;
	float **d_b_ptr_array;
	float **d_c_ptr_array;
	cudaMalloc(&d_a_ptr_array, sizeof(float*) * batch_size);
	cudaMalloc(&d_b_ptr_array, sizeof(float*) * batch_size);
	cudaMalloc(&d_c_ptr_array, sizeof(float*) * batch_size);

	float **h_a_ptr_array;
	float **h_b_ptr_array;
	float **h_c_ptr_array;
	cudaMallocHost(&h_a_ptr_array, sizeof(float*) * batch_size);
	cudaMallocHost(&h_b_ptr_array, sizeof(float*) * batch_size);
	cudaMallocHost(&h_c_ptr_array, sizeof(float*) * batch_size);

	// Host memory for initializing
	float* init_matrix;
	cudaMallocHost(&init_matrix, sizeof(float) * m * n * k / (std::min(m, std::min(n, k))));
	for (unsigned i = 0; i < batch_size; i++) {
		// Allocate device memory and set
		float *d_a_ptr;
		float *d_b_ptr;
		float *d_c_ptr;
		cudaMalloc(&d_a_ptr, sizeof(float) * m * k);
		cudaMalloc(&d_b_ptr, sizeof(float) * k * n);
		cudaMalloc(&d_c_ptr, sizeof(float) * m * n);
		h_a_ptr_array[i] = d_a_ptr;
		h_b_ptr_array[i] = d_b_ptr;
		h_c_ptr_array[i] = d_c_ptr;

		// Initialize matrices
		// A
		for (unsigned j = 0; j < m * k; j++) init_matrix[j] = 1.f;
		cudaMemcpy(d_a_ptr, init_matrix, sizeof(float) * m * k, cudaMemcpyDefault);
		// B
		for (unsigned j = 0; j < k * n; j++) init_matrix[j] = 1.f;
		cudaMemcpy(d_b_ptr, init_matrix, sizeof(float) * k * n, cudaMemcpyDefault);
		// C
		for (unsigned j = 0; j < m * n; j++) init_matrix[j] = 1.f;
		cudaMemcpy(d_c_ptr, init_matrix, sizeof(float) * m * n, cudaMemcpyDefault);
	}
	cudaFreeHost(init_matrix);

	// Copy the pointer array to the device
	cudaMemcpy(d_a_ptr_array, h_a_ptr_array, sizeof(float*) * batch_size, cudaMemcpyDefault);
	cudaMemcpy(d_b_ptr_array, h_b_ptr_array, sizeof(float*) * batch_size, cudaMemcpyDefault);
	cudaMemcpy(d_c_ptr_array, h_c_ptr_array, sizeof(float*) * batch_size, cudaMemcpyDefault);
	std::printf("Start evaluation\n");

	cudaDeviceSynchronize();
	const auto start_clock = std::chrono::system_clock::now();
	bgemm<SMEM_M, SMEM_N, SMEM_K, WARP_M, WARP_N, WARP_K, BLOCK_SIZE, FRAGMENT_T, TC_Policy>(
			m, n, k,
			1.f,
			d_a_ptr_array, m,
			d_b_ptr_array, n,
			1.f,
			d_c_ptr_array, k,
			batch_size
			);
	cudaDeviceSynchronize();
	const auto end_clock = std::chrono::system_clock::now();
	const auto elapsed_time = std::chrono::duration_cast<std::chrono::microseconds>(end_clock - start_clock).count() * 1e-6;
	const auto complexity = 2lu * static_cast<std::size_t>(m) * static_cast<std::size_t>(n) * static_cast<std::size_t>(k) * static_cast<std::size_t>(batch_size);

	std::printf("-------\n");
	std::printf("%15s: (%u, %u, %u)\n", "Size", m, n, k);
	std::printf("%15s: %u\n", "Batch size", batch_size);
	std::printf("%15s: %lu byte\n", "Shared memory", sizeof(float) * (SMEM_M * SMEM_K + SMEM_K * SMEM_N + SMEM_M * SMEM_N));
	std::printf("%15s: %e s\n", "Time", elapsed_time);
	std::printf("%15s: %e TFlop/s\n", "Performance", complexity / elapsed_time / (1lu << 40));

	// Free
	for (unsigned i = 0; i < batch_size; i++) {
		cudaFree(h_a_ptr_array[i]);
		cudaFree(h_b_ptr_array[i]);
		cudaFree(h_c_ptr_array[i]);
	}
	cudaFree(d_a_ptr_array);
	cudaFree(d_b_ptr_array);
	cudaFree(d_c_ptr_array);
	cudaFreeHost(h_a_ptr_array);
	cudaFreeHost(h_b_ptr_array);
	cudaFreeHost(h_c_ptr_array);
}
} // noname napespace

int main() {
	test_batched_sgemm<128, 64, 16, 32, 16, 16, 512>(1024, 1024, 1024, 512);
}
