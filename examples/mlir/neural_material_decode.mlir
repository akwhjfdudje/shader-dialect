// A small neural-material-flavored decode path.
//
// This is not modeling Shader autodiff yet; it is a concrete example of why a
// Shader dialect might be useful beside MLIR math/tensor code later. Texture
// features are kept as shader semantics while the decode math is ordinary MLIR.

module attributes {
  shader.stage = "fragment",
  shader.source = "neural_material_decode.shader",
  shader.differentiable = true
} {
  func.func @decode_neural_material(
      %featureTex: !shader.texture,
      %sampler: !shader.sampler,
      %uv: vector<2xf32>,
      %w0: f32,
      %w1: f32,
      %bias: f32) -> (f32, f32) {
    %features0 = "shader.texture_sample_reusable"(%featureTex, %sampler, %uv)
      : (!shader.texture, !shader.sampler, vector<2xf32>) -> vector<4xf32>
    %features1 = "shader.texture_sample_reusable"(%featureTex, %sampler, %uv)
      : (!shader.texture, !shader.sampler, vector<2xf32>) -> vector<4xf32>

    %f0 = "shader.channel"(%features0) {channel = 0 : i32}
      : (vector<4xf32>) -> f32
    %f1 = "shader.channel"(%features1) {channel = 1 : i32}
      : (vector<4xf32>) -> f32

    %h0 = arith.mulf %f0, %w0 : f32
    %h1 = arith.mulf %f1, %w1 : f32
    %sum0 = arith.addf %h0, %h1 : f32
    %sum1 = arith.addf %sum0, %bias : f32
    %roughness = "shader.saturate"(%sum1) : (f32) -> f32

    %one = arith.constant 1.0 : f32
    %inverted = arith.subf %one, %roughness : f32
    %metallic = "shader.saturate"(%inverted) : (f32) -> f32

    return %roughness, %metallic : f32, f32
  }
}
