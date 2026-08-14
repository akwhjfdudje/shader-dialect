// RUN: %shader-opt %s --shader-resource-summary | FileCheck %s

module {
  func.func @packed(%tex: !shader.texture, %orm: !shader.texture, %sampler: !shader.sampler, %uv: vector<2xf32>) -> (f32, f32) {
    %base = "shader.texture_sample"(%tex, %sampler, %uv) : (!shader.texture, !shader.sampler, vector<2xf32>) -> vector<4xf32>
    %orm0 = "shader.texture_sample_reusable"(%orm, %sampler, %uv) : (!shader.texture, !shader.sampler, vector<2xf32>) -> vector<4xf32>
    %orm1 = "shader.texture_sample_reusable"(%orm, %sampler, %uv) : (!shader.texture, !shader.sampler, vector<2xf32>) -> vector<4xf32>
    %r = "shader.channel"(%base) {channel = 0 : i32} : (vector<4xf32>) -> f32
    %r_again = "shader.channel"(%base) {channel = 0 : i32} : (vector<4xf32>) -> f32
    %g = "shader.channel"(%orm0) {channel = 1 : i32} : (vector<4xf32>) -> f32
    %b = "shader.channel"(%orm1) {channel = 2 : i32} : (vector<4xf32>) -> f32
    return %g, %b : f32, f32
  }
}

// CHECK: shader.resource_summary
// CHECK: func @packed
// CHECK: resource(texture=@packed.arg0, sampler=@packed.arg2) ordinary_samples=1 reusable_samples=0 channels=[r]
// CHECK: resource(texture=@packed.arg1, sampler=@packed.arg2) ordinary_samples=0 reusable_samples=2 channels=[g, b]
