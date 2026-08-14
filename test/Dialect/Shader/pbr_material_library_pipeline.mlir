// RUN: %shader-opt %s --cse --shader-prune-dead-resources --shader-lower-to-proxy-backend | FileCheck %s

// ============================================================================
// PBR Material Library — full optimization pipeline demo
// ============================================================================

module attributes {shader.stage = "fragment", shader.source = "pbr_material_library.shader"} {

  func.func @full_pbr(
      %baseColorTex: !shader.texture,
      %normalTex: !shader.texture,
      %ormTex: !shader.texture,
      %sampler: !shader.sampler,
      %uv: vector<2xf32>,
      %ndotl: f32
  ) -> f32 {
    %base = "shader.texture_sample"(%baseColorTex, %sampler, %uv)
      : (!shader.texture, !shader.sampler, vector<2xf32>) -> vector<4xf32>
    %baseR = "shader.channel"(%base) {channel = 0 : i32}
      : (vector<4xf32>) -> f32

    %normal = "shader.texture_sample"(%normalTex, %sampler, %uv)
      : (!shader.texture, !shader.sampler, vector<2xf32>) -> vector<4xf32>
    %normalY = "shader.channel"(%normal) {channel = 1 : i32}
      : (vector<4xf32>) -> f32

    %ormA = "shader.texture_sample_reusable"(%ormTex, %sampler, %uv)
      : (!shader.texture, !shader.sampler, vector<2xf32>) -> vector<4xf32>
    %ormB = "shader.texture_sample_reusable"(%ormTex, %sampler, %uv)
      : (!shader.texture, !shader.sampler, vector<2xf32>) -> vector<4xf32>
    %occlusion = "shader.channel"(%ormA) {channel = 0 : i32}
      : (vector<4xf32>) -> f32
    %roughness = "shader.channel"(%ormB) {channel = 1 : i32}
      : (vector<4xf32>) -> f32
    %metallic = "shader.channel"(%ormB) {channel = 2 : i32}
      : (vector<4xf32>) -> f32

    %diffuse = arith.mulf %baseR, %ndotl : f32
    %specBoost = arith.addf %diffuse, %normalY : f32
    %occluded = arith.mulf %specBoost, %occlusion : f32
    %clamped = "shader.saturate"(%occluded) : (f32) -> f32
    return %clamped : f32
  }

  func.func @simple_diffuse(
      %baseColorTex: !shader.texture,
      %normalTex: !shader.texture,
      %ormTex: !shader.texture,
      %sampler: !shader.sampler,
      %uv: vector<2xf32>,
      %ndotl: f32
  ) -> f32 {
    %base = "shader.texture_sample"(%baseColorTex, %sampler, %uv)
      : (!shader.texture, !shader.sampler, vector<2xf32>) -> vector<4xf32>
    %baseR = "shader.channel"(%base) {channel = 0 : i32}
      : (vector<4xf32>) -> f32
    %diffuse = arith.mulf %baseR, %ndotl : f32
    %clamped = "shader.saturate"(%diffuse) : (f32) -> f32
    return %clamped : f32
  }

  func.func @emissive(
      %baseColorTex: !shader.texture,
      %sampler: !shader.sampler,
      %uv: vector<2xf32>,
      %emissiveStrength: f32
  ) -> f32 {
    %emissive = "shader.saturate"(%emissiveStrength) : (f32) -> f32
    return %emissive : f32
  }
}

// ── @full_pbr ──
// CHECK-LABEL: func.func @full_pbr(
// CHECK-SAME: !shader.texture
// CHECK-SAME: !shader.texture
// CHECK-SAME: !shader.texture
// CHECK-SAME: !shader.sampler
// CSE: one reusable sample
// CHECK: "shader.texture_sample_reusable"
// Lowering: vector.extract, arith.maximumf, arith.minimumf
// CHECK: vector.extract
// CHECK: arith.maximumf
// CHECK: arith.minimumf

// ── @simple_diffuse ──
// CHECK-LABEL: func.func @simple_diffuse(
// Only one texture arg remains (normalTex and ormTex pruned)
// CHECK-SAME: !shader.texture
// CHECK-NOT: !shader.texture
// CHECK-SAME: !shader.sampler

// ── @emissive ──
// CHECK-LABEL: func.func @emissive(
// No texture args remain (baseColorTex and sampler pruned)
// CHECK-SAME: vector<2xf32>
// CHECK-SAME: f32
// CHECK-NOT: !shader.texture
// CHECK-NOT: !shader.sampler
