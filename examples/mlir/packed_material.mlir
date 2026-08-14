// A material-style fragment shader that samples two packed textures.
//
// The ORM texture is marked reusable because the Shader-side analysis has proven
// the same resource/sampler/coordinate/derivative context. Running `--cse`
// should merge the duplicated reusable sample while preserving the ordinary
// base-color sample semantics.

module attributes {shader.stage = "fragment", shader.source = "packed_material.shader"} {
  func.func @shade_packed_material(
      %baseColorTex: !shader.texture,
      %ormTex: !shader.texture,
      %sampler: !shader.sampler,
      %uv: vector<2xf32>,
      %ndotv: f32) -> (f32, f32, f32) {
    %base = "shader.texture_sample"(%baseColorTex, %sampler, %uv)
      : (!shader.texture, !shader.sampler, vector<2xf32>) -> vector<4xf32>
    %baseR = "shader.channel"(%base) {channel = 0 : i32}
      : (vector<4xf32>) -> f32

    %orm0 = "shader.texture_sample_reusable"(%ormTex, %sampler, %uv)
      : (!shader.texture, !shader.sampler, vector<2xf32>) -> vector<4xf32>
    %orm1 = "shader.texture_sample_reusable"(%ormTex, %sampler, %uv)
      : (!shader.texture, !shader.sampler, vector<2xf32>) -> vector<4xf32>
    %occlusion = "shader.channel"(%orm0) {channel = 0 : i32}
      : (vector<4xf32>) -> f32
    %roughness = "shader.channel"(%orm1) {channel = 1 : i32}
      : (vector<4xf32>) -> f32
    %metallic = "shader.channel"(%orm1) {channel = 2 : i32}
      : (vector<4xf32>) -> f32

    %lit = arith.mulf %baseR, %ndotv : f32
    %occluded = arith.mulf %lit, %occlusion : f32
    %energy = "shader.saturate"(%occluded) : (f32) -> f32

    return %energy, %roughness, %metallic : f32, f32, f32
  }
}
