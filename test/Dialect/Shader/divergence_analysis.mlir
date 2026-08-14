// RUN: %shader-opt %s --shader-divergence-analysis 2>&1 | FileCheck %s

// Warnings go to stderr and appear before the module on stdout.
// Exactly two warnings are expected (divergent_if + divergent_for).
// CHECK: warning: shader.texture_sample inside divergent control flow
// CHECK: warning: shader.texture_sample inside divergent control flow
// CHECK-NOT: warning
// CHECK: module {

func.func @uniform_flow(%tex: !shader.texture, %sampler: !shader.sampler,
                        %uv: vector<2xf32>, %weight: f32) -> f32 {
  %sample = "shader.texture_sample"(%tex, %sampler, %uv)
      : (!shader.texture, !shader.sampler, vector<2xf32>) -> vector<4xf32>
  %channel = "shader.channel"(%sample) <{channel = 0 : i32}>
      : (vector<4xf32>) -> f32
  return %channel : f32
}

func.func @divergent_if(%tex: !shader.texture, %sampler: !shader.sampler,
                        %uv: vector<2xf32>, %threshold: f32) -> f32 {
  %x = vector.extract %uv[0] : f32 from vector<2xf32>
  %cond = arith.cmpf ogt, %x, %threshold : f32
  %result = scf.if %cond -> (f32) {
    %sample = "shader.texture_sample"(%tex, %sampler, %uv)
        : (!shader.texture, !shader.sampler, vector<2xf32>) -> vector<4xf32>
    %ch = "shader.channel"(%sample) <{channel = 0 : i32}>
        : (vector<4xf32>) -> f32
    scf.yield %ch : f32
  } else {
    %c0 = arith.constant 0.0 : f32
    scf.yield %c0 : f32
  }
  return %result : f32
}

func.func @uniform_if(%tex: !shader.texture, %sampler: !shader.sampler,
                      %uv: vector<2xf32>) -> f32 {
  %true = arith.constant true
  %result = scf.if %true -> (f32) {
    %sample = "shader.texture_sample"(%tex, %sampler, %uv)
        : (!shader.texture, !shader.sampler, vector<2xf32>) -> vector<4xf32>
    %ch = "shader.channel"(%sample) <{channel = 0 : i32}>
        : (vector<4xf32>) -> f32
    scf.yield %ch : f32
  } else {
    %c0 = arith.constant 0.0 : f32
    scf.yield %c0 : f32
  }
  return %result : f32
}

func.func @divergent_for(%tex: !shader.texture, %sampler: !shader.sampler,
                         %uv: vector<2xf32>) -> f32 {
  %lb = arith.constant 0 : index
  %step = arith.constant 1 : index
  %x = vector.extract %uv[0] : f32 from vector<2xf32>
  %ub_f = arith.fptoui %x : f32 to i32
  %ub = arith.index_castui %ub_f : i32 to index
  %init = arith.constant 0.0 : f32
  %result = scf.for %i = %lb to %ub step %step iter_args(%acc = %init) -> f32 {
    %sample = "shader.texture_sample"(%tex, %sampler, %uv)
        : (!shader.texture, !shader.sampler, vector<2xf32>) -> vector<4xf32>
    %ch = "shader.channel"(%sample) <{channel = 0 : i32}>
        : (vector<4xf32>) -> f32
    %sum = arith.addf %acc, %ch : f32
    scf.yield %sum : f32
  }
  return %result : f32
}
