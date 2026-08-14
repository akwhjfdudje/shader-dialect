// RUN: %shader-opt %s --shader-lower-to-proxy-backend | FileCheck %s

module {
  func.func @proxy_backend(%tex: !shader.texture, %sampler: !shader.sampler, %uv: vector<2xf32>, %x: f32, %sample: vector<4xf32>) -> f32 {
    %ordinary = "shader.texture_sample"(%tex, %sampler, %uv) : (!shader.texture, !shader.sampler, vector<2xf32>) -> vector<4xf32>
    %reusable = "shader.texture_sample_reusable"(%tex, %sampler, %uv) : (!shader.texture, !shader.sampler, vector<2xf32>) -> vector<4xf32>
    %hint = "shader.derivative_hint"(%uv) : (vector<2xf32>) -> vector<2xf32>
    %red = "shader.channel"(%sample) {channel = 0 : i32} : (vector<4xf32>) -> f32
    %sat = "shader.saturate"(%x) : (f32) -> f32
    return %sat : f32
  }
}

// CHECK-LABEL: func.func @proxy_backend
// CHECK: "shader.texture_sample"
// CHECK: "shader.texture_sample_reusable"
// CHECK-NOT: "shader.derivative_hint"
// CHECK-NOT: "shader.channel"
// CHECK-NOT: "shader.saturate"
// CHECK: vector.extract {{.*}}[0] : f32 from vector<4xf32>
// CHECK: arith.constant 0.000000e+00 : f32
// CHECK: arith.constant 1.000000e+00 : f32
// CHECK: arith.maximumf
// CHECK: arith.minimumf
