// Demonstrates why ordinary samples and derivative hints are not marked pure.
//
// The repeated ordinary normal samples look identical in generic SSA terms, but
// the dialect does not allow stock CSE to merge them. This models a conservative
// Shader import where implicit derivatives, LOD, or helper-lane behavior has not
// been proven reusable.

module attributes {shader.stage = "fragment", shader.source = "normal_mapping.shader"} {
  func.func @shade_with_derivative_sensitive_normal(
      %normalTex: !shader.texture,
      %sampler: !shader.sampler,
      %uv: vector<2xf32>,
      %viewTerm: f32) -> f32 {
    %duv = "shader.derivative_hint"(%uv)
      : (vector<2xf32>) -> vector<2xf32>

    %n0 = "shader.texture_sample"(%normalTex, %sampler, %duv)
      : (!shader.texture, !shader.sampler, vector<2xf32>) -> vector<4xf32>
    %n1 = "shader.texture_sample"(%normalTex, %sampler, %duv)
      : (!shader.texture, !shader.sampler, vector<2xf32>) -> vector<4xf32>

    %nx = "shader.channel"(%n0) {channel = 0 : i32}
      : (vector<4xf32>) -> f32
    %ny = "shader.channel"(%n1) {channel = 1 : i32}
      : (vector<4xf32>) -> f32

    %normalEnergy = arith.mulf %nx, %ny : f32
    %facing = arith.mulf %normalEnergy, %viewTerm : f32
    %clamped = "shader.saturate"(%facing) : (f32) -> f32
    return %clamped : f32
  }
}
