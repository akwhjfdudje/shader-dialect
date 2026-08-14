// RUN: %shader-opt %s --shader-hoist-divergent-samples | FileCheck %s

// CHECK-LABEL: func.func @hoist_divergent_if
// CHECK:       arith.cmpf
// CHECK:       %[[SAMPLE:.*]] = "shader.texture_sample"
// CHECK:       scf.if
// CHECK-NOT:     "shader.texture_sample" inside scf.if

func.func @hoist_divergent_if(%tex: !shader.texture, %samp: !shader.sampler,
                              %uv: vector<2xf32>, %t: f32) -> f32 {
  %x = vector.extract %uv[0] : f32 from vector<2xf32>
  %cond = arith.cmpf ogt, %x, %t : f32
  %result = scf.if %cond -> (f32) {
    %s = "shader.texture_sample"(%tex, %samp, %uv)
        : (!shader.texture, !shader.sampler, vector<2xf32>) -> vector<4xf32>
    %ch = "shader.channel"(%s) <{channel = 0 : i32}>
        : (vector<4xf32>) -> f32
    scf.yield %ch : f32
  } else {
    %c0 = arith.constant 0.0 : f32
    scf.yield %c0 : f32
  }
  return %result : f32
}

// CHECK-LABEL: func.func @uniform_if_unchanged
// CHECK:       scf.if
// CHECK:         "shader.texture_sample"

func.func @uniform_if_unchanged(%tex: !shader.texture, %samp: !shader.sampler,
                                %uv: vector<2xf32>) -> f32 {
  %true = arith.constant true
  %result = scf.if %true -> (f32) {
    %s = "shader.texture_sample"(%tex, %samp, %uv)
        : (!shader.texture, !shader.sampler, vector<2xf32>) -> vector<4xf32>
    %ch = "shader.channel"(%s) <{channel = 0 : i32}>
        : (vector<4xf32>) -> f32
    scf.yield %ch : f32
  } else {
    %c0 = arith.constant 0.0 : f32
    scf.yield %c0 : f32
  }
  return %result : f32
}
