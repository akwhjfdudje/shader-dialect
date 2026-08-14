// RUN: %shader-opt %s --shader-to-spirv | FileCheck %s
//
// Tests that --shader-to-spirv converts shader texture types and sample ops
// to SPIR-V dialect types and ops.

module {
  // CHECK-LABEL: func.func @sample_ordinary
  // CHECK-SAME: !spirv.image<
  // CHECK-SAME: !spirv.sampler
  func.func @sample_ordinary(%tex: !shader.texture, %samp: !shader.sampler, %uv: vector<2xf32>) -> vector<4xf32> {
    // CHECK: spirv.SampledImage
    // CHECK: spirv.ImageSampleImplicitLod
    // CHECK-NOT: shader.texture_sample
    %0 = "shader.texture_sample"(%tex, %samp, %uv) : (!shader.texture, !shader.sampler, vector<2xf32>) -> vector<4xf32>
    return %0 : vector<4xf32>
  }

  // CHECK-LABEL: func.func @sample_reusable
  // CHECK-SAME: !spirv.image<
  // CHECK-SAME: !spirv.sampler
  func.func @sample_reusable(%tex: !shader.texture, %samp: !shader.sampler, %uv: vector<2xf32>) -> vector<4xf32> {
    // CHECK: spirv.SampledImage
    // CHECK: spirv.ImageSampleImplicitLod
    // CHECK-NOT: shader.texture_sample_reusable
    %0 = "shader.texture_sample_reusable"(%tex, %samp, %uv) : (!shader.texture, !shader.sampler, vector<2xf32>) -> vector<4xf32>
    return %0 : vector<4xf32>
  }

  // Both texture_sample and texture_sample_reusable lower to the same SPIR-V ops
}
