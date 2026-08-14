// Neural PBR Material — 4 texture features -> 6 hidden -> 3 outputs (roughness, metallic, AO)
//
// What this demonstrates for the shader-MLIR research:
//   1. Texture features + MLP math coexist in one IR
//   2. texture_sample_reusable -> CSE merges duplicate feature fetches
//   3. Variant specialization can prune unused neural paths
//   4. Proxy lowering converts shader ops to generic MLIR (no shader dialect remaining)
//   5. Stock MLIR passes (CSE, SCCP) operate across the shader/ML boundary
//
// In a real system, the weights would come from offline training, baked as constants
// or specialization parameters. Here they're function arguments for clarity.

module attributes {shader.stage = "fragment", shader.source = "neural_pbr_material.shader"} {

  // ── Full neural PBR: feature texture + MLP decode ──
  func.func @neural_pbr(
      %featureTex: !shader.texture,
      %sampler: !shader.sampler,
      %uv: vector<2xf32>,
      // ── Layer 0->1 weights (4 features -> 6 hidden) ──
      %w00: f32, %w01: f32, %w02: f32, %w03: f32,
      %w10: f32, %w11: f32, %w12: f32, %w13: f32,
      %w20: f32, %w21: f32, %w22: f32, %w23: f32,
      %w30: f32, %w31: f32, %w32: f32, %w33: f32,
      %w40: f32, %w41: f32, %w42: f32, %w43: f32,
      %w50: f32, %w51: f32, %w52: f32, %w53: f32,
      // ── Layer 0 biases (6 hidden) ──
      %b00: f32, %b01: f32, %b02: f32, %b03: f32, %b04: f32, %b05: f32,
      // ── Layer 1->2 weights (6 hidden -> 3 output) ──
      %r0: f32, %r1: f32, %r2: f32, %r3: f32, %r4: f32, %r5: f32,
      %m0: f32, %m1: f32, %m2: f32, %m3: f32, %m4: f32, %m5: f32,
      %a0: f32, %a1: f32, %a2: f32, %a3: f32, %a4: f32, %a5: f32,
      // ── Layer 1 biases (3 output) ──
      %b_out0: f32, %b_out1: f32, %b_out2: f32
  ) -> (f32, f32, f32) {

    // ──────────────────────────────────────────────────────────────────
    // Stage 1: Fetch texture features (reusable — same UV, proven safe)
    // ──────────────────────────────────────────────────────────────────
    %features = "shader.texture_sample_reusable"(%featureTex, %sampler, %uv)
      : (!shader.texture, !shader.sampler, vector<2xf32>) -> vector<4xf32>
    %f0 = "shader.channel"(%features) {channel = 0 : i32}
      : (vector<4xf32>) -> f32
    %f1 = "shader.channel"(%features) {channel = 1 : i32}
      : (vector<4xf32>) -> f32
    %f2 = "shader.channel"(%features) {channel = 2 : i32}
      : (vector<4xf32>) -> f32
    %f3 = "shader.channel"(%features) {channel = 3 : i32}
      : (vector<4xf32>) -> f32

    // ──────────────────────────────────────────────────────────────────
    // Stage 2: Layer 0 -> 1  (4 features -> 6 hidden, ReLU activation)
    //   h_i = ReLU( f0·w_i0 + f1·w_i1 + f2·w_i2 + f3·w_i3 + b_i )
    // ──────────────────────────────────────────────────────────────────
    %c0 = arith.constant 0.0 : f32

    // hidden neuron 0
    %h0_dot0 = arith.mulf %f0, %w00 : f32
    %h0_dot1 = arith.mulf %f1, %w01 : f32
    %h0_dot2 = arith.mulf %f2, %w02 : f32
    %h0_dot3 = arith.mulf %f3, %w03 : f32
    %h0_sum01 = arith.addf %h0_dot0, %h0_dot1 : f32
    %h0_sum23 = arith.addf %h0_dot2, %h0_dot3 : f32
    %h0_sum = arith.addf %h0_sum01, %h0_sum23 : f32
    %h0_biased = arith.addf %h0_sum, %b00 : f32
    %h0 = arith.maximumf %h0_biased, %c0 : f32

    // hidden neuron 1
    %h1_dot0 = arith.mulf %f0, %w10 : f32
    %h1_dot1 = arith.mulf %f1, %w11 : f32
    %h1_dot2 = arith.mulf %f2, %w12 : f32
    %h1_dot3 = arith.mulf %f3, %w13 : f32
    %h1_sum01 = arith.addf %h1_dot0, %h1_dot1 : f32
    %h1_sum23 = arith.addf %h1_dot2, %h1_dot3 : f32
    %h1_sum = arith.addf %h1_sum01, %h1_sum23 : f32
    %h1_biased = arith.addf %h1_sum, %b01 : f32
    %h1 = arith.maximumf %h1_biased, %c0 : f32

    // hidden neuron 2
    %h2_dot0 = arith.mulf %f0, %w20 : f32
    %h2_dot1 = arith.mulf %f1, %w21 : f32
    %h2_dot2 = arith.mulf %f2, %w22 : f32
    %h2_dot3 = arith.mulf %f3, %w23 : f32
    %h2_sum01 = arith.addf %h2_dot0, %h2_dot1 : f32
    %h2_sum23 = arith.addf %h2_dot2, %h2_dot3 : f32
    %h2_sum = arith.addf %h2_sum01, %h2_sum23 : f32
    %h2_biased = arith.addf %h2_sum, %b02 : f32
    %h2 = arith.maximumf %h2_biased, %c0 : f32

    // hidden neuron 3
    %h3_dot0 = arith.mulf %f0, %w30 : f32
    %h3_dot1 = arith.mulf %f1, %w31 : f32
    %h3_dot2 = arith.mulf %f2, %w32 : f32
    %h3_dot3 = arith.mulf %f3, %w33 : f32
    %h3_sum01 = arith.addf %h3_dot0, %h3_dot1 : f32
    %h3_sum23 = arith.addf %h3_dot2, %h3_dot3 : f32
    %h3_sum = arith.addf %h3_sum01, %h3_sum23 : f32
    %h3_biased = arith.addf %h3_sum, %b03 : f32
    %h3 = arith.maximumf %h3_biased, %c0 : f32

    // hidden neuron 4
    %h4_dot0 = arith.mulf %f0, %w40 : f32
    %h4_dot1 = arith.mulf %f1, %w41 : f32
    %h4_dot2 = arith.mulf %f2, %w42 : f32
    %h4_dot3 = arith.mulf %f3, %w43 : f32
    %h4_sum01 = arith.addf %h4_dot0, %h4_dot1 : f32
    %h4_sum23 = arith.addf %h4_dot2, %h4_dot3 : f32
    %h4_sum = arith.addf %h4_sum01, %h4_sum23 : f32
    %h4_biased = arith.addf %h4_sum, %b04 : f32
    %h4 = arith.maximumf %h4_biased, %c0 : f32

    // hidden neuron 5
    %h5_dot0 = arith.mulf %f0, %w50 : f32
    %h5_dot1 = arith.mulf %f1, %w51 : f32
    %h5_dot2 = arith.mulf %f2, %w52 : f32
    %h5_dot3 = arith.mulf %f3, %w53 : f32
    %h5_sum01 = arith.addf %h5_dot0, %h5_dot1 : f32
    %h5_sum23 = arith.addf %h5_dot2, %h5_dot3 : f32
    %h5_sum = arith.addf %h5_sum01, %h5_sum23 : f32
    %h5_biased = arith.addf %h5_sum, %b05 : f32
    %h5 = arith.maximumf %h5_biased, %c0 : f32

    // ──────────────────────────────────────────────────────────────────
    // Stage 3: Layer 1 -> 2  (6 hidden -> 3 output, saturate clamp)
    //   roughness = saturate( Σ h_i · r_i + b_out0 )
    //   metallic  = saturate( Σ h_i · m_i + b_out1 )
    //   ao        = saturate( Σ h_i · a_i + b_out2 )
    // ──────────────────────────────────────────────────────────────────

    // ── roughness ──
    %r_dot0 = arith.mulf %h0, %r0 : f32
    %r_dot1 = arith.mulf %h1, %r1 : f32
    %r_dot2 = arith.mulf %h2, %r2 : f32
    %r_dot3 = arith.mulf %h3, %r3 : f32
    %r_dot4 = arith.mulf %h4, %r4 : f32
    %r_dot5 = arith.mulf %h5, %r5 : f32
    %r_sum01 = arith.addf %r_dot0, %r_dot1 : f32
    %r_sum23 = arith.addf %r_dot2, %r_dot3 : f32
    %r_sum45 = arith.addf %r_dot4, %r_dot5 : f32
    %r_sum0 = arith.addf %r_sum01, %r_sum23 : f32
    %r_sum = arith.addf %r_sum0, %r_sum45 : f32
    %r_pre = arith.addf %r_sum, %b_out0 : f32
    %roughness = "shader.saturate"(%r_pre) : (f32) -> f32

    // ── metallic ──
    %m_dot0 = arith.mulf %h0, %m0 : f32
    %m_dot1 = arith.mulf %h1, %m1 : f32
    %m_dot2 = arith.mulf %h2, %m2 : f32
    %m_dot3 = arith.mulf %h3, %m3 : f32
    %m_dot4 = arith.mulf %h4, %m4 : f32
    %m_dot5 = arith.mulf %h5, %m5 : f32
    %m_sum01 = arith.addf %m_dot0, %m_dot1 : f32
    %m_sum23 = arith.addf %m_dot2, %m_dot3 : f32
    %m_sum45 = arith.addf %m_dot4, %m_dot5 : f32
    %m_sum0 = arith.addf %m_sum01, %m_sum23 : f32
    %m_sum = arith.addf %m_sum0, %m_sum45 : f32
    %m_pre = arith.addf %m_sum, %b_out1 : f32
    %metallic = "shader.saturate"(%m_pre) : (f32) -> f32

    // ── ambient occlusion ──
    %a_dot0 = arith.mulf %h0, %a0 : f32
    %a_dot1 = arith.mulf %h1, %a1 : f32
    %a_dot2 = arith.mulf %h2, %a2 : f32
    %a_dot3 = arith.mulf %h3, %a3 : f32
    %a_dot4 = arith.mulf %h4, %a4 : f32
    %a_dot5 = arith.mulf %h5, %a5 : f32
    %a_sum01 = arith.addf %a_dot0, %a_dot1 : f32
    %a_sum23 = arith.addf %a_dot2, %a_dot3 : f32
    %a_sum45 = arith.addf %a_dot4, %a_dot5 : f32
    %a_sum0 = arith.addf %a_sum01, %a_sum23 : f32
    %a_sum = arith.addf %a_sum0, %a_sum45 : f32
    %a_pre = arith.addf %a_sum, %b_out2 : f32
    %ao = "shader.saturate"(%a_pre) : (f32) -> f32

    return %roughness, %metallic, %ao : f32, f32, f32
  }

  // ── Degraded variant: skips the neural decode entirely ──
  //
  // This represents a material variant where an offline optimization decided
  // the neural path is unnecessary (e.g. constant output, or not visible).
  // The feature texture and all 54 weight parameters become dead resources
  // that the pruning pass eliminates.

  func.func @degraded_variant(
      %featureTex: !shader.texture,
      %sampler: !shader.sampler,
      %uv: vector<2xf32>,
      // ── same weight args as neural_pbr (54 scalars) ──
      %w00: f32, %w01: f32, %w02: f32, %w03: f32,
      %w10: f32, %w11: f32, %w12: f32, %w13: f32,
      %w20: f32, %w21: f32, %w22: f32, %w23: f32,
      %w30: f32, %w31: f32, %w32: f32, %w33: f32,
      %w40: f32, %w41: f32, %w42: f32, %w43: f32,
      %w50: f32, %w51: f32, %w52: f32, %w53: f32,
      %b00: f32, %b01: f32, %b02: f32, %b03: f32, %b04: f32, %b05: f32,
      %r0: f32, %r1: f32, %r2: f32, %r3: f32, %r4: f32, %r5: f32,
      %m0: f32, %m1: f32, %m2: f32, %m3: f32, %m4: f32, %m5: f32,
      %a0: f32, %a1: f32, %a2: f32, %a3: f32, %a4: f32, %a5: f32,
      %b_out0: f32, %b_out1: f32, %b_out2: f32
  ) -> (f32, f32, f32) {
    // This variant doesn't sample the texture or use any weights.
    // It returns constant values — pruning eliminates everything above.
    %c_roughness = arith.constant 0.5 : f32
    %c_metallic = arith.constant 0.0 : f32
    %c_ao = arith.constant 1.0 : f32
    return %c_roughness, %c_metallic, %c_ao : f32, f32, f32
  }
}
