// Adated from FasterTransformer, https://github.com/NVIDIA/FasterTransformer/blob/release/v5.3_tag/src/fastertransformer/kernels/decoder_masked_multihead_attention/decoder_masked_multihead_attention_template.hpp
#pragma once

#include <assert.h>
#include <stdint.h>
#include <float.h>
#include <type_traits>
#include "../../attention/attention_dtypes.h"
#include <stdio.h>

namespace vllm {
namespace int8 {


// KV-CACHE int8
static inline __device__ float int8_to_float(uint8_t x, const float scale) {
  int8_t a = x - 128;
  float res = a * scale;

  return res;
}

static inline __device__ uint8_t float_to_int8(float x, const float scale) {
  int8_t fx = roundf(max(-128.f, min(127.f, x / scale)));
  uint8_t res = fx + 128;

  return res;
}

template <typename Tout, typename Tin>
__inline__ __device__ Tout scaled_vec_conversion_int8(const Tin& x,
                                                      const float scale) {
  return x;
}

// int8 -> half
// template <>
// __inline__ __device__ uint16_t scaled_vec_conversion_int8<uint16_t, uint8_t>(
//     const uint8_t& a, const float scale) {
//   float res = int8_to_float(a, scale);
//   return float_to_half(res);
//   // return half(a);__float2half 
// }

// int8x2 -> half2
template <>
__inline__ __device__ uint32_t scaled_vec_conversion_int8<uint32_t, uint16_t>(
    const uint16_t& a, const float scale) {
    union {
        uint8_t  uint8[2];
        uint16_t uint16;
    };
    uint16 = a;

    float2 b;
    b.x = (uint8[0] - 128) * scale;
    b.y = (uint8[1] - 128) * scale;
    union {
        half2    float16;
        uint32_t uint32;
    };

    float16 = __float22half2_rn(b);
    return uint32;
}

template<typename Tout, typename Tin>
__inline__ __device__ Tout vec_conversion(const Tin& x)
{
    return x;
}
template<>
__inline__ __device__ uint32_t vec_conversion<uint32_t, float2>(const float2& a)
{
    union {
        half2    float16;
        uint32_t uint32;
    };

    float16 = __float22half2_rn(a);
    return uint32;
}

template<>
__inline__ __device__ uint2 vec_conversion<uint2, Float4_>(const Float4_& a)
{
    uint2  b;
    float2 val;
    val.x = a.x.x;
    val.y = a.x.y;
    b.x   = vec_conversion<uint32_t, float2>(val);

    val.x = a.y.x;
    val.y = a.y.y;
    b.y   = vec_conversion<uint32_t, float2>(val);

    return b;
}
// int8x4 -> half2x2
template <>
__inline__ __device__ uint2 scaled_vec_conversion_int8<uint2, uint32_t>(
    const uint32_t& a, const float scale) {
    union {
        uint8_t  uint8[4];
        uint32_t uint32;
    };
    uint32 = a;
    Float4_ b;
    b.x.x = (uint8[0] - 128) * scale;
    b.x.y = (uint8[1] - 128) * scale;
    b.y.x = (uint8[2] - 128) * scale;
    b.y.y = (uint8[3] - 128) * scale;
    return vec_conversion<uint2, Float4_>(b);
}

inline __device__ float2 dequant(uint16_t a, const float scale)
{
    union {
        uint8_t  uint8[2];
        uint16_t uint16;
    };
    uint16 = a;

    float2 b;
    b.x = (uint8[0] - 128) * scale;
    b.y = (uint8[1] - 128) * scale;
    return b;
}

// int8x8 -> half2x4
template <>
__inline__ __device__ uint4
scaled_vec_conversion_int8<uint4, uint2>(const uint2& a, const float scale) {
// scaled_vec_conversion_int8<uint4, uint64_t>(const uint64_t& a, const float scale) {

    union {
        uint16_t uint16[4];
        uint2 uint64;
    };
    uint64 = a;

    Float8_ b;
    b.x = dequant(uint16[0], scale);
    b.y = dequant(uint16[1], scale);
    b.z = dequant(uint16[2], scale);
    b.w = dequant(uint16[3], scale);
    uint4 c;
    c.x = vec_conversion<uint32_t, float2>(b.x);
    c.y = vec_conversion<uint32_t, float2>(b.y);
    c.z = vec_conversion<uint32_t, float2>(b.z);
    c.w = vec_conversion<uint32_t, float2>(b.w);
    return c;
}

// int8 -> __nv_bfloat16
template <>
__inline__ __device__ __nv_bfloat16
scaled_vec_conversion_int8<__nv_bfloat16, uint8_t>(const uint8_t& a,
                                                   const float scale) {
  // Note there is no direct convert function from int8 to bf16.
  float res = int8_to_float(a, scale);
  return __float2bfloat16(res);
}

// int8x2 -> __nv_bfloat162
template <>
__inline__ __device__ __nv_bfloat162
scaled_vec_conversion_int8<__nv_bfloat162, uint16_t>(const uint16_t& a,
                                                     const float scale) {
  __nv_bfloat162 res;
  res.x = scaled_vec_conversion_int8<__nv_bfloat16, uint8_t>((uint8_t)a, scale);
  res.y = scaled_vec_conversion_int8<__nv_bfloat16, uint8_t>((uint8_t)(a >> 8U),
                                                             scale);
  return res;
}

// int8x4 -> bf16_4_t
template <>
__inline__ __device__ bf16_4_t scaled_vec_conversion_int8<bf16_4_t, uint32_t>(
    const uint32_t& a, const float scale) {
  bf16_4_t res;
  res.x =
      scaled_vec_conversion_int8<__nv_bfloat162, uint16_t>((uint16_t)a, scale);
  res.y = scaled_vec_conversion_int8<__nv_bfloat162, uint16_t>(
      (uint16_t)(a >> 16U), scale);
  return res;
}

// int8x8 -> bf16_8_t
template <>
__inline__ __device__ bf16_8_t
scaled_vec_conversion_int8<bf16_8_t, uint2>(const uint2& a, const float scale) {
// scaled_vec_conversion_int8<bf16_8_t, uint64_t>(const uint64_t& a, const float scale) {
  // bf16_4_t tmp1, tmp2;
  // tmp1 = scaled_vec_conversion_int8<bf16_4_t, uint32_t>(a.x, scale);
  // tmp2 = scaled_vec_conversion_int8<bf16_4_t, uint32_t>(a.y, scale);
  bf16_8_t res;
  // res.x = tmp1.x;
  // res.y = tmp1.y;
  // res.z = tmp2.x;
  // res.w = tmp2.y;
  return res;
}

// int8 -> float
template <>
__inline__ __device__ float scaled_vec_conversion_int8<float, uint8_t>(
    const uint8_t& a, const float scale) {
  float res = int8_to_float(a, scale);
  return res;
}

// int8x2 -> float2
template <>
__inline__ __device__ float2 scaled_vec_conversion_int8<float2, uint16_t>(
    const uint16_t& a, const float scale) {
  // int8x2 -> half2
  uint32_t tmp = scaled_vec_conversion_int8<uint32_t, uint16_t>(a, scale);
  // half2 -> float2
  return half2_to_float2(tmp);
}

// int8x4 -> float4
template <>
__inline__ __device__ Float4_ scaled_vec_conversion_int8<Float4_, uint32_t>(
    const uint32_t& a, const float scale) {
  Float4_ res;
  res.x = scaled_vec_conversion_int8<float2, uint16_t>((uint16_t)a, scale);
  res.y =
      scaled_vec_conversion_int8<float2, uint16_t>((uint16_t)(a >> 16U), scale);
  return res;
}

// int8x8 -> float8
template <>
__inline__ __device__ Float8_
scaled_vec_conversion_int8<Float8_, uint64_t>(const uint64_t& a, const float scale) {
// scaled_vec_conversion_int8<Float8_, uint2>(const uint2& a, const float scale) {
//   Float4_ tmp1, tmp2;
//   tmp1 = scaled_vec_conversion_int8<Float4_, uint32_t>(a.x, scale);
//   tmp2 = scaled_vec_conversion_int8<Float4_, uint32_t>(a.y, scale);
  Float8_ res;
//   res.x = tmp1.x;
//   res.y = tmp1.y;
//   res.z = tmp2.x;
//   res.w = tmp2.y;
  return res;
}

// half -> int8
template <>
__inline__ __device__ uint8_t scaled_vec_conversion_int8<uint8_t, uint16_t>(
    const uint16_t& a, const float scale) {
  uint8_t res = float_to_int8(half_to_float(a), scale);

  return (uint8_t)res;
  // return (uint8_t)(a);
}

// bf16 -> int8
template <>
__inline__ __device__ uint8_t
scaled_vec_conversion_int8<uint8_t, __nv_bfloat16>(const __nv_bfloat16& a,
                                                   const float scale) {
  uint8_t res = float_to_int8(__bfloat162float(a), scale);
  return (uint8_t)res;
}

// float -> int8
template <>
__inline__ __device__ uint8_t
scaled_vec_conversion_int8<uint8_t, float>(const float& a, const float scale) {
  uint8_t res = float_to_int8(a, scale);
  return (uint8_t)res;
  // return (uint8_t)(a);
}

// int8x4 -> float4
template <>
__inline__ __device__ float4 scaled_vec_conversion_int8<float4, uint32_t>(
    const uint32_t& a, const float scale) {
  Float4_ tmp = scaled_vec_conversion_int8<Float4_, uint32_t>(a, scale);
  float4 res = make_float4(tmp.x.x, tmp.x.y, tmp.y.x, tmp.y.y);
  return res;
}

} // namespace int8
} // namespace vllm
