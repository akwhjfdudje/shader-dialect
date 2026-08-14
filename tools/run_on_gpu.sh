#!/bin/bash
# run_on_gpu.sh — Compile a Shader-MLIR .mlir file to SPIR-V and prepare for
#                GPU execution on Windows via Vulkan.
#
# Usage:
#   ./tools/run_on_gpu.sh examples/mlir/pbr_material_library.mlir [entry_point]
#
# This script:
#   1. Generates fragment.spv from the MLIR source via serialize_spirv.py
#   2. Compiles the vertex shader (fullscreen_triangle.vert) to SPIR-V
#      using glslangValidator from the Windows Vulkan SDK
#   3. Validates both SPIR-V binaries with spirv-val
#   4. Prints the command to run on Windows

set -euo pipefail

REPO="$(cd "$(dirname "$0")/.." && pwd)"
BUILD_DIR="$REPO/build"
TOOLS_DIR="$REPO/tools"
FRAG_SPV="$BUILD_DIR/fragment.spv"
VERT_SPV="$BUILD_DIR/vertex.spv"
GLSLANG="/mnt/c/VulkanSDK/1.4.350.0/Bin/glslangValidator.exe"
ENTRY_POINT="${2:-}"

if [ $# -lt 1 ]; then
    echo "Usage: $0 <input.mlir> [entry_point_name]"
    echo "Example: $0 examples/mlir/pbr_material_library.mlir full_pbr"
    exit 1
fi

INPUT="$1"
if [ ! -f "$INPUT" ]; then
    # Try relative to REPO
    INPUT="$REPO/$1"
fi
if [ ! -f "$INPUT" ]; then
    echo "ERROR: input file not found: $1"
    exit 1
fi

mkdir -p "$BUILD_DIR"

echo "=== Step 1: Generate fragment SPIR-V ==="
python3 "$TOOLS_DIR/serialize_spirv.py" "$INPUT" -o "$FRAG_SPV"

echo ""
echo "=== Step 2: Validate fragment SPIR-V ==="
spirv-val "$FRAG_SPV"
echo "  fragment SPIR-V valid."

echo ""
echo "=== Step 3: Compile vertex shader to SPIR-V ==="
if [ -f "$GLSLANG" ]; then
    powershell.exe -Command "& '$GLSLANG' -V '$TOOLS_DIR/fullscreen_triangle.vert' -o '$VERT_SPV'" 2>&1
else
    # Fall back to WSL's glslangValidator if installed
    if command -v glslangValidator &>/dev/null; then
        glslangValidator -V "$TOOLS_DIR/fullscreen_triangle.vert" -o "$VERT_SPV"
    else
        echo "ERROR: glslangValidator not found."
        echo "  Install with: sudo apt install glslang-tools"
        echo "  Or ensure Vulkan SDK is at: $GLSLANG"
        exit 1
    fi
fi

if [ ! -f "$VERT_SPV" ]; then
    echo "ERROR: vertex SPIR-V not generated."
    exit 1
fi

echo ""
echo "=== Step 4: Validate vertex SPIR-V ==="
spirv-val "$VERT_SPV"
echo "  vertex SPIR-V valid."

echo ""
echo "=== Step 5: Ready for GPU execution ==="
echo ""
echo "  SPIR-V files written to:"
echo "    $FRAG_SPV"
echo "    $VERT_SPV"
echo ""
echo "  Next steps:"
echo ""
echo "    # 1. Build the Vulkan runner (only needed once):"
echo "    powershell -Command \"cmd.exe /c 'C:\\Users\\arsha\\Documents\\work\\work_things\\research\\shader\\tools\\build_runner.bat'\""
echo ""
if [ -n "$ENTRY_POINT" ]; then
    echo "    # 2. Execute on GPU:"
    echo "    powershell -Command \"& build\\vulkan_runner.exe build\\vertex.spv build\\fragment.spv $ENTRY_POINT\""
else
    echo "    # 2. Execute on GPU:"
    echo "    powershell -Command \"& build\\vulkan_runner.exe build\\vertex.spv build\\fragment.spv\""
fi
