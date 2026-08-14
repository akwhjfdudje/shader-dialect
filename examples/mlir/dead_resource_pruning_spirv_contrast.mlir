// ============================================================================
// Dead Resource Pruning:  shader dialect  vs  SPIR-V
// ============================================================================
//
// This file shows the same shader at two levels of IR.  The high-level
// shader dialect makes dead-resource pruning trivial (check use_empty()
// on SSA values).  After lowering to SPIR-V, the same information is
// scattered across global variables, decorations, access chains, pointer
// types, and entry-point metadata.
//
// The concrete task:  %dead_tex  is never sampled.  Remove it.
//
// ============================================================================
// LEVEL 1 — shader dialect (trivial to prune)
// ============================================================================
//
//   func.func @shade(
//       %live_tex:  !shader.texture,   // used → keep
//       %dead_tex:  !shader.texture,   // unused → prune
//       %sampler:   !shader.sampler,
//       %uv:        vector<2xf32>
//   ) -> vector<4xf32> {
//     %color = "shader.texture_sample"(%live_tex, %sampler, %uv)
//         : (!shader.texture, !shader.sampler, vector<2xf32>) -> vector<4xf32>
//     return %color : vector<4xf32>
//   }
//
// Pruning algorithm:
//   for each arg i:
//     if isa<TextureType>(arg[i].getType()) && arg[i].use_empty():
//       eraseArgument(i)
//       updateCallSites(i)
//
// That's it.  One loop.  SSA tells you everything.
//
// ============================================================================
// LEVEL 2 — SPIR-V (much harder)
// ============================================================================
//
// After lowering through the full pipeline (proxy backend → shader-to-spirv
// → func-to-spirv → wrap-spirv-module), the same shader becomes this:

  spirv.module Logical GLSL450 attributes {vce = #spirv.vce<v1.3, [Shader, Sampled1D], []>} {

    // ── types ────────────────────────────────────────────────────────
    %float     = spirv.Float  : f32
    %v4float   = spirv.Vector : vector<4xf32>
    %v2float   = spirv.Vector : vector<2xf32>
    %image_t   = spirv.Image  : !spirv.image<f32, 2D, 0, 0, 0, 0, Unknown>
    %samp_t    = spirv.Sampler : !spirv.sampler
    %sampled_t = spirv.SampledImage : !spirv.sampled-image<f32, 2D, 0, 0, 0, 0, Unknown>

    // ── global variables (resources + interface) ─────────────────────
    // Each argument became a global.  Two textures → two globals.
    // The binding numbers (0, 1) were assigned by the wrap pass.
    // To prune %dead_tex, you must find which global corresponds to it
    // AND renumber all subsequent bindings to close the gap.

    %ptr_live_img = spirv.Pointer : !spirv.ptr<!spirv.image<f32, 2D, 0, 0, 0, 0, Unknown>, UniformConstant>
    %resource_0   = spirv.GlobalVariable %ptr_live_img bind(0, 0) : !spirv.ptr<!spirv.image<f32, 2D, 0, 0, 0, 0, Unknown>, UniformConstant>

    %ptr_dead_img = spirv.Pointer : !spirv.ptr<!spirv.image<f32, 2D, 0, 0, 0, 0, Unknown>, UniformConstant>
    %resource_1   = spirv.GlobalVariable %ptr_dead_img bind(0, 1) : !spirv.ptr<!spirv.image<f32, 2D, 0, 0, 0, 0, Unknown>, UniformConstant>

    %ptr_sampler  = spirv.Pointer : !spirv.ptr<!spirv.sampler, UniformConstant>
    %resource_2   = spirv.GlobalVariable %ptr_sampler bind(0, 2) : !spirv.ptr<!spirv.sampler, UniformConstant>

    %ptr_input    = spirv.Pointer : !spirv.ptr<vector<2xf32>, Input>
    %input_0      = spirv.GlobalVariable %ptr_input : !spirv.ptr<vector<2xf32>, Input> loc(0)

    %ptr_output   = spirv.Pointer : !spirv.ptr<vector<4xf32>, Output>
    %output_0     = spirv.GlobalVariable %ptr_output : !spirv.ptr<vector<4xf32>, Output> loc(0)

    // ── function ─────────────────────────────────────────────────────
    spirv.func @shade() -> () "None" {
      %0 = spirv.mlir.addressof @resource_0
          : !spirv.ptr<!spirv.image<f32, 2D, 0, 0, 0, 0, Unknown>, UniformConstant>
      %1 = spirv.Load "UniformConstant" %0
          : !spirv.image<f32, 2D, 0, 0, 0, 0, Unknown>

      // %dead_tex was lowered to %resource_1, loaded into a value.
      // The optimizer (mlir-opt-23) may eliminate this dead load,
      // but the GlobalVariable itself persists in the module.
      %2 = spirv.mlir.addressof @resource_1
          : !spirv.ptr<!spirv.image<f32, 2D, 0, 0, 0, 0, Unknown>, UniformConstant>
      %3 = spirv.Load "UniformConstant" %2
          : !spirv.image<f32, 2D, 0, 0, 0, 0, Unknown>

      %4 = spirv.mlir.addressof @resource_2
          : !spirv.ptr<!spirv.sampler, UniformConstant>
      %5 = spirv.Load "UniformConstant" %4
          : !spirv.sampler

      %6 = spirv.mlir.addressof @input_0
          : !spirv.ptr<vector<2xf32>, Input>
      %7 = spirv.Load "Input" %6
          : vector<2xf32>

      %8 = spirv.SampledImage %sampled_t %1, %5
          : !spirv.sampled-image<f32, 2D, 0, 0, 0, 0, Unknown>
      %9 = spirv.ImageSampleImplicitLod %v4float %8, %7
          : vector<4xf32>

      spirv.Store "Output" %output_0, %9
          : vector<4xf32>
      spirv.Return
    }

    // ── entry point ──────────────────────────────────────────────────
    spirv.EntryPoint "Fragment" @shade, @input_0, @output_0
    spirv.ExecutionMode  @shade "OriginUpperLeft"
  }
