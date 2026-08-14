// RUN: %shader-opt %s --verify-diagnostics

// ── Channel value out of enum range (0-6) ──
module {
  func.func @channel_value_out_of_range(%v: vector<4xf32>) -> f32 {
    // expected-error@+1 {{'shader.channel' op channel value out of range, must be 0-6}}
    %bad = "shader.channel"(%v) {channel = 7 : i32} : (vector<4xf32>) -> f32
    return %bad : f32
  }
}

// ── Single channel (R) with wrong result type ──
module {
  func.func @channel_r_wrong_result(%v: vector<4xf32>) -> vector<2xf32> {
    // expected-error@+1 {{result type must be 'f32' for channel 'r', got 'vector<2xf32>'}}
    %bad = "shader.channel"(%v) {channel = 0 : i32} : (vector<4xf32>) -> vector<2xf32>
    return %bad : vector<2xf32>
  }
}

// ── Multi channel (RG) with wrong result type ──
module {
  func.func @channel_rg_wrong_result(%v: vector<4xf32>) -> vector<4xf32> {
    // expected-error@+1 {{result type must be 'vector<2xf32>' for channel 'rg', got 'vector<4xf32>'}}
    %bad = "shader.channel"(%v) {channel = 4 : i32} : (vector<4xf32>) -> vector<4xf32>
    return %bad : vector<4xf32>
  }
}
