#ifndef __HMMA_F32_F32_DETAIL_CONVERTER_HPP__
#define __HMMA_F32_F32_DETAIL_CONVERTER_HPP__
#include <mma.h>
#include "wmma_extension_include.hpp"

namespace mtk {
namespace wmma {
namespace mma_f32 {
namespace detail {

template <class T>
struct Converter {
	virtual __device__ T operator()(const float a) const = 0;
};

struct ConvertToTF32_RNA : public Converter<float> {
	__device__ float operator()(const float v) const {
		return mtk::wmma::detail::common::cast<nvcuda::wmma::precision::tf32>(v);
	}
};

struct ConvertToTF32_RZ : public Converter<float> {
	__device__ float operator()(const float v) const {
		return v;
	}
};

struct ConvertToFP16 : public Converter<half> {
	__device__ half operator()(const float v) const {
		return mtk::wmma::detail::common::cast<half>(v);
	}
};

template <int scale>
struct ConvertToFP16_Scale : public Converter<half> {
	__device__ half operator()(const float v) const {
		return mtk::wmma::detail::common::cast<half>(v * scale);
	}
};

// ------------------------------
// Default Converter Selector
// ------------------------------
// A
template <class T>
struct default_converter_A;

template <>
struct default_converter_A<half> {using type = ConvertToFP16;};
template <>
struct default_converter_A<nvcuda::wmma::precision::tf32> {using type = ConvertToTF32_RNA;};

// B
template <class T>
struct default_converter_B;

template <>
struct default_converter_B<half> {using type = ConvertToFP16_Scale<1024>;};
template <>
struct default_converter_B<nvcuda::wmma::precision::tf32> {using type = ConvertToTF32_RNA;};
} // namespace detail
} // namespace mma_f32
} // namespace wmma
} // namespace mtk
#endif

