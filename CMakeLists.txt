cmake_minimum_required(VERSION 3.20)

project(ShaderMLIR LANGUAGES CXX C)

set(CMAKE_CXX_STANDARD 17 CACHE STRING "C++ standard to conform to")
set(CMAKE_CXX_STANDARD_REQUIRED YES)
set(CMAKE_CXX_EXTENSIONS NO)

find_package(MLIR REQUIRED CONFIG)
find_package(LLVM REQUIRED CONFIG)

message(STATUS "Using MLIRConfig.cmake in: ${MLIR_DIR}")
message(STATUS "Using LLVMConfig.cmake in: ${LLVM_DIR}")

list(APPEND CMAKE_MODULE_PATH "${MLIR_CMAKE_DIR}")
list(APPEND CMAKE_MODULE_PATH "${LLVM_CMAKE_DIR}")

include(TableGen)
include(AddLLVM)
include(AddMLIR)

include_directories(${LLVM_INCLUDE_DIRS})
include_directories(${MLIR_INCLUDE_DIRS})
include_directories(${PROJECT_SOURCE_DIR}/include)
include_directories(${PROJECT_BINARY_DIR}/include)

add_definitions(${LLVM_DEFINITIONS})

add_subdirectory(include)
add_subdirectory(lib)
add_subdirectory(tools)

enable_testing()
add_subdirectory(test)

find_package(Python3 COMPONENTS Interpreter)
if(Python3_Interpreter_FOUND)
  add_custom_target(shader-bench
    COMMAND ${Python3_EXECUTABLE}
            ${PROJECT_SOURCE_DIR}/tools/benchmark_shader.py
            --build-dir ${PROJECT_BINARY_DIR}
    WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}
    DEPENDS shader-opt
    USES_TERMINAL
  )
endif()

# -- GPU execution targets (Vulkan, Windows host) --

# Locate glslangValidator for GLSL -> SPIR-V compilation.
# On WSL, the Windows Vulkan SDK binaries are accessible via /mnt/c/.
# CMake on Linux doesn't auto-append .exe, so we list both names.
find_program(GLSLLANG_VALIDATOR
  NAMES glslangValidator.exe glslangValidator
  PATHS /mnt/c/VulkanSDK/1.4.350.0/Bin
        /mnt/c/VulkanSDK/1.3.296.0/Bin
  NO_DEFAULT_PATH
)
if(NOT GLSLLANG_VALIDATOR)
  find_program(GLSLLANG_VALIDATOR glslangValidator)
endif()

if(GLSLLANG_VALIDATOR)
  set(VERT_SPV "${PROJECT_BINARY_DIR}/fullscreen_triangle.vert.spv")
  # glslangValidator is a Windows binary; invoke via cmd.exe so it sees
  # Windows-style paths.  cmake -E echo can't pipe here, so we use a
  # shell one-liner that converts the Linux path via wslpath.
  add_custom_command(
    OUTPUT ${VERT_SPV}
    COMMAND bash -c
            "'${GLSLLANG_VALIDATOR}' -V tools/fullscreen_triangle.vert -o build/fullscreen_triangle.vert.spv"
    DEPENDS "${PROJECT_SOURCE_DIR}/tools/fullscreen_triangle.vert"
    WORKING_DIRECTORY "${PROJECT_SOURCE_DIR}"
    COMMENT "Compiling vertex shader -> ${VERT_SPV}"
  )
  add_custom_target(vertex-spv DEPENDS ${VERT_SPV})
  message(STATUS "GPU: glslangValidator found — vertex-spv target available")
else()
  message(STATUS "GPU: glslangValidator not found — vertex-spv target disabled")
endif()

# Vulkan host runner — compiled with MSVC on Windows.
# The build_runner.bat script invokes MSVC; run via cmd.exe.
# Convert Linux /mnt/c/ paths to Windows C:\ for cmd.exe.
set(VULKAN_RUNNER_EXE "${PROJECT_BINARY_DIR}/vulkan_runner.exe")
find_program(CMD_EXE cmd.exe
  PATHS /mnt/c/Windows/System32
  NO_DEFAULT_PATH
)
if(CMD_EXE)
  # Convert /mnt/c/... to C:/... (cmd.exe accepts forward slashes)
  string(REPLACE "/mnt/c/" "C:/" WIN_PROJECT_DIR "${PROJECT_SOURCE_DIR}")
  add_custom_target(vulkan-runner
    COMMAND ${CMD_EXE} /c "${WIN_PROJECT_DIR}/tools/build_runner.bat"
    BYPRODUCTS ${VULKAN_RUNNER_EXE}
    WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}
    COMMENT "Building vulkan_runner.exe with MSVC"
  )
  message(STATUS "GPU: cmd.exe found — vulkan-runner target available")
else()
  message(STATUS "GPU: cmd.exe not found — vulkan-runner target disabled")
endif()

# Convenience target: compile a specific .mlir to SPIR-V and prepare for GPU run.
# Configure with -DSHADER_GPU_INPUT=... -DSHADER_GPU_ENTRY=" full_pbr"
set(SHADER_GPU_INPUT "examples/mlir/pbr_material_library.mlir"
    CACHE STRING "Input .mlir file for shader-gpu-run target")
set(SHADER_GPU_ENTRY " full_pbr"
    CACHE STRING "Entry point name for shader-gpu-run (include leading space)")

add_custom_target(shader-gpu-run
  COMMAND ${Python3_EXECUTABLE}
          ${PROJECT_SOURCE_DIR}/tools/serialize_spirv.py
          ${SHADER_GPU_INPUT}
          -o ${PROJECT_BINARY_DIR}/fragment.spv
  COMMAND ${CMAKE_COMMAND} -E echo ""
  COMMAND ${CMAKE_COMMAND} -E echo "SPIR-V ready. Run on Windows:"
  COMMAND ${CMAKE_COMMAND} -E echo "  cd ${PROJECT_SOURCE_DIR}"
  COMMAND ${CMAKE_COMMAND} -E echo "  build\\vulkan_runner.exe build\\fullscreen_triangle.vert.spv build\\fragment.spv${SHADER_GPU_ENTRY}"
  WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}
  DEPENDS shader-opt ${VERT_SPV}
  USES_TERMINAL
)
