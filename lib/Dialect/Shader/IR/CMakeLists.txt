add_mlir_dialect_library(MLIRShaderDialect
  ShaderDialect.cpp
  ShaderOps.cpp

  DEPENDS
  MLIRShaderOpsIncGen

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRSideEffectInterfaces
)
