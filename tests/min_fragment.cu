#include <iostream>
#include <wmma_extension/hmma_f32_f32.hpp>

template <class T>
void print_min_fragment() {
	std::printf("min fragment size : %u, %u, %u\n",
			mtk::wmma::min_fragment_m<T>,
			mtk::wmma::min_fragment_n<T>,
			mtk::wmma::min_fragment_k<T>
			);
}

int main() {
	print_min_fragment<half>();
	print_min_fragment<nvcuda::wmma::precision::tf32>();
}
