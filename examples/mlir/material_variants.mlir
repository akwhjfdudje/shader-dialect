// Two already-specialized material entry points in one module.
//
// A future Shader import path could create these from Shader specialization data.
// For now, this shows how different variants keep different resource uses
// visible to later resource-pruning or binding-layout passes.

module attributes {shader.stage = "fragment", shader.source = "material_variants.shader"} {
  func.func @full_material_variant(
      %baseColorTex: !shader.texture,
      %normalTex: !shader.texture,
      %clearcoatTex: !shader.texture,
      %sampler: !shader.sampler,
      %uv: vector<2xf32>,
      %ndotl: f32) -> f32 {
    %base = "shader.texture_sample"(%baseColorTex, %sampler, %uv)
      : (!shader.texture, !shader.sampler, vector<2xf32>) -> vector<4xf32>
    %normal = "shader.texture_sample"(%normalTex, %sampler, %uv)
      : (!shader.texture, !shader.sampler, vector<2xf32>) -> vector<4xf32>
    %coat = "shader.texture_sample_reusable"(%clearcoatTex, %sampler, %uv)
      : (!shader.texture, !shader.sampler, vector<2xf32>) -> vector<4xf32>

    %baseR = "shader.channel"(%base) {channel = 0 : i32} : (vector<4xf32>) -> f32
    %normalY = "shader.channel"(%normal) {channel = 1 : i32} : (vector<4xf32>) -> f32
    %coatA = "shader.channel"(%coat) {channel = 3 : i32} : (vector<4xf32>) -> f32

    %diffuse = arith.mulf %baseR, %ndotl : f32
    %normalBoost = arith.mulf %diffuse, %normalY : f32
    %coated = arith.addf %normalBoost, %coatA : f32
    %out = "shader.saturate"(%coated) : (f32) -> f32
    return %out : f32
  }

  func.func @simple_material_variant(
      %baseColorTex: !shader.texture,
      %sampler: !shader.sampler,
      %uv: vector<2xf32>,
      %ndotl: f32) -> f32 {
    %base = "shader.texture_sample"(%baseColorTex, %sampler, %uv)
      : (!shader.texture, !shader.sampler, vector<2xf32>) -> vector<4xf32>
    %baseR = "shader.channel"(%base) {channel = 0 : i32} : (vector<4xf32>) -> f32
    %diffuse = arith.mulf %baseR, %ndotl : f32
    %out = "shader.saturate"(%diffuse) : (f32) -> f32
    return %out : f32
  }
}
