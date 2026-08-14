add_mlir_dialect_library(MLIRShaderTransforms
  DeadResourcePruning.cpp
  HoistDivergentSamples.cpp
  LowerToProxyBackend.cpp
  LowerToSPIRV.cpp
  ResourceUseSummary.cpp
  ShaderDivergenceAnalysis.cpp
  WrapSPIRVModule.cpp

  LINK_LIBS PUBLIC
  MLIRArithDialect
  MLIRFuncDialect
  MLIRIR
  MLIRPass
  MLIRSCFDialect
  MLIRSPIRVDialect
  MLIRShaderDialect
  MLIRSupport
  MLIRVectorDialect
)
