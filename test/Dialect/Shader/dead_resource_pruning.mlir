// RUN: %shader-opt %s --shader-prune-dead-resources | FileCheck %s

module {
  func.func @material(%baseTex: !shader.texture,
                      %normalTex: !shader.texture,
                      %clearcoatTex: !shader.texture,
                      %sampler: !shader.sampler,
                      %uv: vector<2xf32>,
                      %roughness: f32) -> f32 {
    %base = "shader.texture_sample"(%baseTex, %sampler, %uv)
      : (!shader.texture, !shader.sampler, vector<2xf32>) -> vector<4xf32>
    %r = "shader.channel"(%base) {channel = 0 : i32} : (vector<4xf32>) -> f32
    %out = arith.mulf %r, %roughness : f32
    return %out : f32
  }

  func.func @caller(%baseTex: !shader.texture,
                    %normalTex: !shader.texture,
                    %clearcoatTex: !shader.texture,
                    %sampler: !shader.sampler,
                    %uv: vector<2xf32>,
                    %roughness: f32) -> f32 {
    %out = func.call @material(%baseTex, %normalTex, %clearcoatTex, %sampler, %uv, %roughness)
      : (!shader.texture, !shader.texture, !shader.texture, !shader.sampler, vector<2xf32>, f32) -> f32
    return %out : f32
  }
}

// CHECK-LABEL: func.func @material(
// CHECK-SAME: %arg0: !shader.texture
// CHECK-SAME: %arg1: !shader.sampler
// CHECK-SAME: %arg2: vector<2xf32>
// CHECK-SAME: %arg3: f32
// CHECK-SAME: ) -> f32
// CHECK-NOT: clearcoat
// CHECK: "shader.texture_sample"(%arg0, %arg1, %arg2)

// CHECK-LABEL: func.func @caller(
// CHECK-SAME: %arg0: !shader.texture
// CHECK-SAME: %arg1: !shader.sampler
// CHECK-SAME: %arg2: vector<2xf32>
// CHECK-SAME: %arg3: f32
// CHECK-SAME: ) -> f32
// CHECK: call @material(%arg0, %arg1, %arg2, %arg3) : (!shader.texture, !shader.sampler, vector<2xf32>, f32) -> f32
