// RUN: %shader-opt %s --cse | FileCheck %s

module {
  func.func @reusable_samples(%tex: !shader.texture, %sampler: !shader.sampler, %uv: vector<2xf32>) -> (f32, f32) {
    %sample0 = "shader.texture_sample_reusable"(%tex, %sampler, %uv) : (!shader.texture, !shader.sampler, vector<2xf32>) -> vector<4xf32>
    %sample1 = "shader.texture_sample_reusable"(%tex, %sampler, %uv) : (!shader.texture, !shader.sampler, vector<2xf32>) -> vector<4xf32>
    %r = "shader.channel"(%sample0) {channel = 0 : i32} : (vector<4xf32>) -> f32
    %g = "shader.channel"(%sample1) {channel = 1 : i32} : (vector<4xf32>) -> f32
    return %r, %g : f32, f32
  }

  func.func @ordinary_samples(%tex: !shader.texture, %sampler: !shader.sampler, %uv: vector<2xf32>) -> (f32, f32) {
    %sample0 = "shader.texture_sample"(%tex, %sampler, %uv) : (!shader.texture, !shader.sampler, vector<2xf32>) -> vector<4xf32>
    %sample1 = "shader.texture_sample"(%tex, %sampler, %uv) : (!shader.texture, !shader.sampler, vector<2xf32>) -> vector<4xf32>
    %r = "shader.channel"(%sample0) {channel = 0 : i32} : (vector<4xf32>) -> f32
    %g = "shader.channel"(%sample1) {channel = 1 : i32} : (vector<4xf32>) -> f32
    return %r, %g : f32, f32
  }
}

// CHECK-LABEL: func.func @reusable_samples
// CHECK: "shader.texture_sample_reusable"
// CHECK-NOT: "shader.texture_sample_reusable"
// CHECK: return

// CHECK-LABEL: func.func @ordinary_samples
// CHECK: "shader.texture_sample"
// CHECK: "shader.texture_sample"
