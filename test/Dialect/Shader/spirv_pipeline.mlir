// RUN: %shader-opt %s --shader-lower-to-proxy-backend --shader-to-spirv | FileCheck %s
//
// Tests the full lowering pipeline within shader-opt:
// 1. Proxy backend: saturate -> arith, channel -> vector.extract, derivative_hint stripped
// 2. SPIR-V: texture types -> spirv.image/sampler, texture_sample -> spirv ops

module {
  func.func @full_pipeline(%tex: !shader.texture, %samp: !shader.sampler, %uv: vector<2xf32>, %sample: vector<4xf32>) -> f32 {
    %ordinary = "shader.texture_sample"(%tex, %samp, %uv) : (!shader.texture, !shader.sampler, vector<2xf32>) -> vector<4xf32>
    %reusable = "shader.texture_sample_reusable"(%tex, %samp, %uv) : (!shader.texture, !shader.sampler, vector<2xf32>) -> vector<4xf32>
    %hint = "shader.derivative_hint"(%uv) : (vector<2xf32>) -> vector<2xf32>
    %red = "shader.channel"(%sample) {channel = 0 : i32} : (vector<4xf32>) -> f32
    %x = "shader.saturate"(%red) : (f32) -> f32
    return %x : f32
  }
}

// CHECK-LABEL: func.func @full_pipeline
// CHECK-SAME: !spirv.image<
// CHECK-SAME: !spirv.sampler
// CHECK:      spirv.SampledImage
// CHECK:      spirv.ImageSampleImplicitLod
// CHECK:      spirv.SampledImage
// CHECK:      spirv.ImageSampleImplicitLod
// CHECK-NOT:  shader.derivative_hint
// CHECK:      vector.extract
// CHECK-NOT:  shader.channel
// CHECK:      arith.constant 0.000000e+00
// CHECK:      arith.constant 1.000000e+00
// CHECK:      arith.maximumf
// CHECK:      arith.minimumf
// CHECK-NOT:  shader.saturate
// CHECK-NOT:  !shader.
// CHECK-NOT:  "shader.
