# Shader Dialect

Out-of-tree experimental MLIR dialect (`shader`) for preserving and analyzing
shader-language semantics alongside generic MLIR optimizations, with SPIR-V binary output.

## Build

Requires LLVM/MLIR 23 or later with SPIR-V target support.

```sh
# Install LLVM toolchain
wget -qO- https://apt.llvm.org/llvm-snapshot.gpg.key | sudo tee /etc/apt/trusted.gpg.d/llvm.asc
echo "deb http://apt.llvm.org/bookworm/ llvm-toolchain-bookworm main" | sudo tee /etc/apt/sources.list.d/llvm.list
sudo apt update
sudo apt install llvm-23-dev libmlir-23-dev mlir-23-tools spirv-tools

# Build
cmake -G Ninja -S . -B build \
  -DMLIR_DIR=/usr/lib/llvm-23/lib/cmake/mlir \
  -DLLVM_DIR=/usr/lib/llvm-23/lib/cmake/llvm
cmake --build build
```

## Run tests

```sh
ctest --test-dir build --output-on-failure
```

## Compile an example shader to SPIR-V binary

```sh
python3 tools/serialize_spirv.py examples/mlir/pbr_material_library.mlir -o output.spv
spirv-val output.spv
```

The pipeline runs: CSE -> dead resource pruning -> proxy backend lowering ->
shader-to-SPIRV type conversion -> SPIR-V op conversion -> spirv.module wrapping ->
binary serialization.

## Run shaders on the GPU (Vulkan)

The SPIR-V output includes descriptor bindings, I/O locations, and fragment entry
points — ready for Vulkan execution. A companion Vulkan host runner
(`tools/vulkan_runner.cpp`) loads the SPIR-V and dispatches it on the GPU.

**Prerequisites:**
- Windows with Vulkan SDK and Visual Studio Build Tools
- An NVIDIA or other Vulkan-capable GPU

**Quick start (CMake targets, recommended):**

```sh
# Configure with GPU targets (WSL):
cmake -G Ninja -S . -B build \
  -DMLIR_DIR=/usr/lib/llvm-23/lib/cmake/mlir \
  -DLLVM_DIR=/usr/lib/llvm-23/lib/cmake/llvm

# Build the vertex shader and Vulkan host, then generate fragment SPIR-V:
cmake --build build --target vertex-spv vulkan-runner shader-gpu-run

# Execute on GPU (run from Windows PowerShell in the repo root):
build\vulkan_runner.exe build\fullscreen_triangle.vert.spv build\fragment.spv full_pbr
```

**Quick start (manual scripts):**

```sh
# 1. Generate SPIR-V and compile the vertex shader (run from WSL):
./tools/run_on_gpu.sh examples/mlir/pbr_material_library.mlir full_pbr

# 2. Build the Vulkan host runner (run from WSL or Developer PowerShell):
powershell -File tools/build_runner.ps1

# 3. Execute on the GPU (run from Windows PowerShell in the repo root):
build\vulkan_runner.exe build\vertex.spv build\fragment.spv full_pbr
```

**Available CMake GPU targets:**

| Target | What it does |
|---|---|
| `vertex-spv` | Compiles `fullscreen_triangle.vert` → `build/fullscreen_triangle.vert.spv` |
| `vulkan-runner` | Builds `vulkan_runner.exe` with MSVC |
| `shader-gpu-run` | Generates `build/fragment.spv` from `SHADER_GPU_INPUT` (default: `pbr_material_library.mlir`) |

The runner creates a minimal graphics pipeline with:
- A fullscreen triangle vertex shader (no vertex buffer)
- Dummy 1x1 white textures (so `texture_sample` always returns `vec4(1,1,1,1)`)
- Push constants for scalar inputs (ndotl=0.5, emissive=1.0)
- 8 R32_SFLOAT color attachments (one per possible output location)

It prints the GPU name and output values per location. See
`tools/run_on_gpu.sh` for the complete pipeline orchestration.

## Available passes

| Pass | Flag | What it does |
|---|---|---|
| CSE | `--cse` | Merges duplicate `texture_sample_reusable` ops (stock MLIR) |
| Dead resource pruning | `--shader-prune-dead-resources` | Removes unused texture/sampler args |
| Proxy backend lowering | `--shader-lower-to-proxy-backend` | Lowers `saturate` -> `arith.max/min`, `channel` -> `vector.extract` |
| Resource summary | `--shader-resource-summary` | Prints per-function texture/sampler usage |
| Divergence analysis | `--shader-divergence-analysis` | Classifies values as uniform/varying; warns on `texture_sample` in divergent control flow |
| Divergent sample hoisting | `--shader-hoist-divergent-samples` | Hoists `texture_sample` out of divergent `scf.if` regions so implicit derivatives are well-defined |
| Shader-to-SPIRV | `--shader-to-spirv` | Converts `!shader.texture`/`shader.texture_sample` -> SPIR-V types/ops |
| Wrap SPIR-V module | `--shader-wrap-spirv-module` | Wraps converted ops in `spirv.module` with entry points |

## Directory layout

```md
examples/mlir/              Handwritten shader IR (PBR materials, neural MLP, CSE demos)
test/Dialect/               FileCheck and lit tests
benchmarks/                 Benchmark manifests and synthetic IR generators
lib/                        Dialect implementation and passes
tools/serialize_spirv.py    Full shader->SPIRV binary pipeline script
tools/benchmark_shader.py   Optimization timing and metrics
tools/vulkan_runner.cpp     C++ Vulkan host — executes SPIR-V on GPU (Windows)
tools/fullscreen_triangle.vert  GLSL vertex shader for the Vulkan pipeline
tools/run_on_gpu.sh         WSL-side script: MLIR→SPIRV + vertex shader compile
tools/build_runner.ps1      PowerShell script: MSVC compile of vulkan_runner
docs/                       Dialect specifications
```
