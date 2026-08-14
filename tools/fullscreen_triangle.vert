#version 450
// Fullscreen triangle vertex shader for Shader-MLIR fragment shader testbed.
// Generates a fullscreen triangle from gl_VertexIndex (no vertex buffer needed).
// Outputs interpolants at locations matching the fragment shader's Input variables.

layout(location = 0) out vec2  outUV0;
layout(location = 1) out float outScalar1;
layout(location = 2) out vec2  outUV2;
layout(location = 3) out float outScalar3;
layout(location = 4) out vec2  outUV4;
layout(location = 5) out float outScalar5;

// Push constants to drive scalar values from host.
// Layout must match push constant struct in C++ runner (float ndotl, float emissive).
layout(push_constant) uniform PushConstants {
    float ndotl;
    float emissive;
} pc;

void main() {
    // Fullscreen triangle: vertices at (-1,-1), (3,-1), (-1,3)
    vec2 pos;
    if (gl_VertexIndex == 0) {
        pos = vec2(-1.0, -1.0);
    } else if (gl_VertexIndex == 1) {
        pos = vec2(3.0, -1.0);
    } else {
        pos = vec2(-1.0, 3.0);
    }

    vec2 uv = (pos + 1.0) * 0.5;  // map [-1,1] -> [0,1]

    gl_Position = vec4(pos, 0.0, 1.0);

    outUV0     = uv;
    outScalar1 = pc.ndotl;

    outUV2     = uv;
    outScalar3 = pc.ndotl;

    outUV4     = uv;
    outScalar5 = pc.emissive;
}
