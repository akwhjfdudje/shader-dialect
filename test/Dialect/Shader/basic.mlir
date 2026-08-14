// RUN: %shader-opt %s --verify-diagnostics | FileCheck %s

module {
  func.func @shader_entry(%tex: !shader.texture, %sampler: !shader.sampler, %uv: vector<2xf32>, %x: f32) -> f32 {
    %sample = "shader.texture_sample"(%tex, %sampler, %uv) : (!shader.texture, !shader.sampler, vector<2xf32>) -> vector<4xf32>
    %r = "shader.channel"(%sample) {channel = 0 : i32} : (vector<4xf32>) -> f32
    %sat = "shader.saturate"(%x) : (f32) -> f32
    return %sat : f32
  }
}

// CHECK-LABEL: func.func @shader_entry
// CHECK: "shader.texture_sample"
// CHECK: "shader.channel"
// CHECK: "shader.saturate"
