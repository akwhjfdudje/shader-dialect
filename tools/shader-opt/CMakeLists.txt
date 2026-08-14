add_llvm_executable(shader-opt
  shader-opt.cpp
)

llvm_update_compile_flags(shader-opt)

target_link_libraries(shader-opt
  PRIVATE
  MLIRShaderDialect
  MLIRShaderTransforms
  MLIRArithDialect
  MLIRArithToSPIRV
  MLIRFuncDialect
  MLIRFuncToSPIRV
  MLIRIR
  MLIRMlirOptMain
  MLIRParser
  MLIRPass
  MLIRSPIRVConversion
  MLIRSPIRVDialect
  MLIRSPIRVTransforms
  MLIRSupport
  MLIRTransforms
  MLIRVectorDialect
  MLIRVectorToSPIRV
)
