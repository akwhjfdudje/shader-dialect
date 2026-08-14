// A mini PBR material library with 3 specialized entry points.
//
// @full_pbr:       uses baseColor, normal (ordinary), ORM (reusable, merged by CSE)
// @simple_diffuse: uses baseColor only — normal + ORM are dead and pruned
// @emissive:       pure math, no textures — baseColor is dead and pruned

module attributes {shader.stage = "fragment", shader.source = "pbr_material_library.shader"} {

  // ── Full PBR: 3 textures, reusable ORM samples merge ──
  func.func @full_pbr(
      %baseColorTex: !shader.texture,
      %normalTex: !shader.texture,
      %ormTex: !shader.texture,
      %sampler: !shader.sampler,
      %uv: vector<2xf32>,
      %ndotl: f32
  ) -> f32 {
    // ordinary sample (not marked reusable — stays separate)
    %base = "shader.texture_sample"(%baseColorTex, %sampler, %uv)
      : (!shader.texture, !shader.sampler, vector<2xf32>) -> vector<4xf32>
    %baseR = "shader.channel"(%base) {channel = 0 : i32}
      : (vector<4xf32>) -> f32

    // ordinary normal sample (derivative-sensitive, stays separate)
    %normal = "shader.texture_sample"(%normalTex, %sampler, %uv)
      : (!shader.texture, !shader.sampler, vector<2xf32>) -> vector<4xf32>
    %normalY = "shader.channel"(%normal) {channel = 1 : i32}
      : (vector<4xf32>) -> f32

    // two identical reusable ORM samples — CSE will merge them
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

  // ── Simple diffuse: only baseColor used, normal+ORM are dead ──
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

  // ── Emissive-only: no textures at all, pure math ──
  func.func @emissive(
      %baseColorTex: !shader.texture,
      %sampler: !shader.sampler,
      %uv: vector<2xf32>,
      %emissiveStrength: f32
  ) -> f32 {
    // baseColorTex is completely unused — pruning will remove it
    %emissive = "shader.saturate"(%emissiveStrength) : (f32) -> f32
    return %emissive : f32
  }
}
