// RUN: %shader-opt %s --shader-prune-dead-resources --shader-lower-to-proxy-backend | FileCheck %s

// ============================================================================
// Neural PBR Material — optimizer strips unused neural paths
// ============================================================================

module attributes {shader.stage = "fragment", shader.source = "neural_pbr_material.shader"} {

  // ── @neural_pbr: uses feature texture + MLP ──
  func.func @neural_pbr(
      %tex: !shader.texture, %samp: !shader.sampler, %uv: vector<2xf32>,
      %w0: f32, %w1: f32, %w2: f32, %w3: f32, %b: f32
  ) -> f32 {
    %feat = "shader.texture_sample_reusable"(%tex, %samp, %uv)
      : (!shader.texture, !shader.sampler, vector<2xf32>) -> vector<4xf32>
    %f0 = "shader.channel"(%feat) {channel = 0 : i32}
      : (vector<4xf32>) -> f32
    %f1 = "shader.channel"(%feat) {channel = 1 : i32}
      : (vector<4xf32>) -> f32
    %d0 = arith.mulf %f0, %w0 : f32
    %d1 = arith.mulf %f1, %w1 : f32
    %sum = arith.addf %d0, %d1 : f32
    %biased = arith.addf %sum, %b : f32
    %relu = arith.maximumf %biased, %w0 : f32
    %out = "shader.saturate"(%relu) : (f32) -> f32
    return %out : f32
  }

  // ── @no_neural: skips the MLP entirely, texture is dead ──
  func.func @no_neural(
      %tex: !shader.texture, %samp: !shader.sampler, %uv: vector<2xf32>,
      %w0: f32, %w1: f32, %w2: f32, %w3: f32, %b: f32
  ) -> f32 {
    %c = arith.constant 5.000000e-01 : f32
    return %c : f32
  }
}

// ── @neural_pbr ─────────────────────────────────────────────────────────────
// CHECK-LABEL: func.func @neural_pbr(
// Texture + sampler survive (used in texture_sample_reusable)
// CHECK-SAME: !shader.texture
// CHECK-SAME: !shader.sampler

// channel ops lowered -> vector.extract
// CHECK: vector.extract
// CHECK: vector.extract

// saturate lowered -> arith.minimumf/arith.maximumf
// CHECK: arith.minimumf

// No shader ops remain
// CHECK-NOT: "shader.

// ── @no_neural ──────────────────────────────────────────────────────────────
// CHECK-LABEL: func.func @no_neural(
// featureTex and sampler should be PRUNED from the signature
// CHECK-SAME: vector<2xf32>
// CHECK-NOT: !shader.texture
// CHECK-NOT: !shader.sampler

// Body is just the constant — no texture ops, no MLP
// CHECK: arith.constant 5.000000e-01
// CHECK: return
