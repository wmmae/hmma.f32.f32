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

struct ConvertToFP32_Id : public Converter<float> {
	__device__ float operator()(const float v) const {
		return v;
	}
};

// ------------------------------
// Default Converter Selector
// ------------------------------
// A
template <class T>
struct default_converter_hv;

template <>
struct default_converter_hv<half> {using type = ConvertToFP16;};
template <>
struct default_converter_hv<nvcuda::wmma::precision::tf32> {using type = ConvertToTF32_RNA;};
template <>
struct default_converter_hv<float> {using type = ConvertToFP32_Id;};

// B
template <class T>
struct default_converter_dhv;

template <>
struct default_converter_dhv<half> {using type = ConvertToFP16_Scale<1024>;};
template <>
struct default_converter_dhv<nvcuda::wmma::precision::tf32> {using type = ConvertToTF32_RNA;};
template <>
struct default_converter_dhv<float> {using type = ConvertToFP32_Id;};
} // namespace detail
} // namespace mma_f32
} // namespace wmma
} // namespace mtk
#endif

