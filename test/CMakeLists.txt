find_program(LLVM_LIT lit lit-23 llvm-lit HINTS ${LLVM_TOOLS_BINARY_DIR} /usr/lib/llvm-23/bin /usr/bin)
find_program(FILECHECK FileCheck HINTS ${LLVM_TOOLS_BINARY_DIR} /usr/lib/llvm-23/bin)
find_package(Python3 COMPONENTS Interpreter)
find_program(SPIRV_DIS spirv-dis)
find_program(SPIRV_VAL spirv-val)

if(FILECHECK)
  add_test(
    NAME shader-basic-filecheck
    COMMAND /bin/sh -c "$<TARGET_FILE:shader-opt> ${CMAKE_CURRENT_SOURCE_DIR}/Dialect/Shader/basic.mlir --verify-diagnostics | ${FILECHECK} ${CMAKE_CURRENT_SOURCE_DIR}/Dialect/Shader/basic.mlir"
  )

  add_test(
    NAME shader-cse-filecheck
    COMMAND /bin/sh -c "$<TARGET_FILE:shader-opt> ${CMAKE_CURRENT_SOURCE_DIR}/Dialect/Shader/cse.mlir --cse | ${FILECHECK} ${CMAKE_CURRENT_SOURCE_DIR}/Dialect/Shader/cse.mlir"
  )

  add_test(
    NAME shader-resource-summary-filecheck
    COMMAND /bin/sh -c "$<TARGET_FILE:shader-opt> ${CMAKE_CURRENT_SOURCE_DIR}/Dialect/Shader/resource_summary.mlir --shader-resource-summary | ${FILECHECK} ${CMAKE_CURRENT_SOURCE_DIR}/Dialect/Shader/resource_summary.mlir"
  )

  add_test(
    NAME shader-dead-resource-pruning-filecheck
    COMMAND /bin/sh -c "$<TARGET_FILE:shader-opt> ${CMAKE_CURRENT_SOURCE_DIR}/Dialect/Shader/dead_resource_pruning.mlir --shader-prune-dead-resources | ${FILECHECK} ${CMAKE_CURRENT_SOURCE_DIR}/Dialect/Shader/dead_resource_pruning.mlir"
  )

  add_test(
    NAME shader-proxy-backend-lowering-filecheck
    COMMAND /bin/sh -c "$<TARGET_FILE:shader-opt> ${CMAKE_CURRENT_SOURCE_DIR}/Dialect/Shader/proxy_backend_lowering.mlir --shader-lower-to-proxy-backend | ${FILECHECK} ${CMAKE_CURRENT_SOURCE_DIR}/Dialect/Shader/proxy_backend_lowering.mlir"
  )

  add_test(
    NAME shader-pbr-pipeline-filecheck
    COMMAND /bin/sh -c "$<TARGET_FILE:shader-opt> ${CMAKE_CURRENT_SOURCE_DIR}/Dialect/Shader/pbr_material_library_pipeline.mlir --cse --shader-prune-dead-resources --shader-lower-to-proxy-backend | ${FILECHECK} ${CMAKE_CURRENT_SOURCE_DIR}/Dialect/Shader/pbr_material_library_pipeline.mlir"
  )

  add_test(
    NAME shader-pbr-verifier-errors
    COMMAND /bin/sh -c "$<TARGET_FILE:shader-opt> ${CMAKE_CURRENT_SOURCE_DIR}/Dialect/Shader/pbr_material_library_verifier_errors.mlir --verify-diagnostics"
  )

  add_test(
    NAME shader-neural-pbr-pipeline-filecheck
    COMMAND /bin/sh -c "$<TARGET_FILE:shader-opt> ${CMAKE_CURRENT_SOURCE_DIR}/Dialect/Shader/neural_pbr_material_pipeline.mlir --shader-prune-dead-resources --shader-lower-to-proxy-backend | ${FILECHECK} ${CMAKE_CURRENT_SOURCE_DIR}/Dialect/Shader/neural_pbr_material_pipeline.mlir"
  )

  add_test(
    NAME shader-divergence-analysis-filecheck
    COMMAND /bin/sh -c "$<TARGET_FILE:shader-opt> ${CMAKE_CURRENT_SOURCE_DIR}/Dialect/Shader/divergence_analysis.mlir --shader-divergence-analysis 2>&1 | ${FILECHECK} ${CMAKE_CURRENT_SOURCE_DIR}/Dialect/Shader/divergence_analysis.mlir"
  )

  add_test(
    NAME shader-hoist-divergent-filecheck
    COMMAND /bin/sh -c "$<TARGET_FILE:shader-opt> ${CMAKE_CURRENT_SOURCE_DIR}/Dialect/Shader/hoist_divergent_samples.mlir --shader-hoist-divergent-samples | ${FILECHECK} ${CMAKE_CURRENT_SOURCE_DIR}/Dialect/Shader/hoist_divergent_samples.mlir"
  )

  # ── SPIR-V lowering tests ──

  add_test(
    NAME shader-spirv-lowering-filecheck
    COMMAND /bin/sh -c "$<TARGET_FILE:shader-opt> ${CMAKE_CURRENT_SOURCE_DIR}/Dialect/Shader/spirv_lowering.mlir --shader-to-spirv | ${FILECHECK} ${CMAKE_CURRENT_SOURCE_DIR}/Dialect/Shader/spirv_lowering.mlir"
  )

  add_test(
    NAME shader-spirv-pipeline-filecheck
    COMMAND /bin/sh -c "$<TARGET_FILE:shader-opt> ${CMAKE_CURRENT_SOURCE_DIR}/Dialect/Shader/spirv_pipeline.mlir --shader-lower-to-proxy-backend --shader-to-spirv | ${FILECHECK} ${CMAKE_CURRENT_SOURCE_DIR}/Dialect/Shader/spirv_pipeline.mlir"
  )
else()
  message(WARNING "FileCheck not found; direct dialect tests will not be registered")
endif()

# ── SPIR-V binary tests (requires spirv-tools) ──

if(SPIRV_DIS AND Python3_Interpreter_FOUND)
  add_test(
    NAME shader-spirv-binary-disasm
    COMMAND /bin/sh -c "${Python3_EXECUTABLE} ${CMAKE_SOURCE_DIR}/tools/serialize_spirv.py ${CMAKE_SOURCE_DIR}/examples/mlir/neural_material_decode.mlir -o /tmp/shader_test_disasm.spv && ${SPIRV_DIS} /tmp/shader_test_disasm.spv | ${FILECHECK} ${CMAKE_CURRENT_SOURCE_DIR}/Dialect/Shader/spirv_disasm_check.txt"
  )
else()
  message(WARNING "spirv-dis or Python3 not found; SPIR-V disassembly check will not be registered")
endif()

if(SPIRV_VAL AND Python3_Interpreter_FOUND)
  set(SPIRV_EXAMPLE_FILES
    packed_material
    derivative_sensitive_sampling
    neural_material_decode
    material_variants
    pbr_material_library
    neural_pbr_material
  )
  foreach(example ${SPIRV_EXAMPLE_FILES})
    add_test(
      NAME shader-spirv-validate-${example}
      COMMAND /bin/sh -c "${Python3_EXECUTABLE} ${CMAKE_SOURCE_DIR}/tools/serialize_spirv.py ${CMAKE_SOURCE_DIR}/examples/mlir/${example}.mlir -o /tmp/shader_spirv_val_${example}.spv && ${SPIRV_VAL} /tmp/shader_spirv_val_${example}.spv"
    )
  endforeach()
else()
  message(WARNING "spirv-val or Python3 not found; SPIR-V validation tests will not be registered")
endif()

set(SHADER_EXAMPLE_FILES
  packed_material
  derivative_sensitive_sampling
  neural_material_decode
  material_variants
  pbr_material_library
  neural_pbr_material
)

foreach(example ${SHADER_EXAMPLE_FILES})
  add_test(
    NAME shader-example-${example}
    COMMAND $<TARGET_FILE:shader-opt>
            ${CMAKE_SOURCE_DIR}/examples/mlir/${example}.mlir
            --cse
  )
endforeach()

if(LLVM_LIT)
  configure_file(lit.site.cfg.py.in lit.site.cfg.py @ONLY)
  add_test(
    NAME shader-mlir-lit
    COMMAND ${LLVM_LIT} -sv ${CMAKE_CURRENT_BINARY_DIR}
  )
else()
  message(WARNING "llvm-lit not found; lit tests will not be registered")
endif()
