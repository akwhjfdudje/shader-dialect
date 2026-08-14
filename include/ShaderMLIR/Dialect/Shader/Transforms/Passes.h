//===- Passes.h - Shader dialect passes -----------------------*- C++ -*-===//

#ifndef SHADER_MLIR_DIALECT_SHADER_TRANSFORMS_PASSES_H
#define SHADER_MLIR_DIALECT_SHADER_TRANSFORMS_PASSES_H

/**
 * @file Passes.h
 * @brief Declares registration and utility functions for Shader MLIR passes.
 */

namespace mlir {
namespace shader {

/**
 * @brief Registers the dead resource pruning transformation pass.
 *
 * This pass detects and removes unused shader resources from function arguments.
 */
void registerShaderDeadResourcePruningPass();

/**
 * @brief Registers the proxy backend lowering transformation pass.
 *
 * This pass lowers saturate and channel extraction ops to standard MLIR.
 */
void registerShaderLowerToProxyBackendPass();

/**
 * @brief Registers the resource use summary analysis pass.
 *
 * This pass analyzes and prints a summary of how texture and sampler resources are used.
 */
void registerShaderResourceSummaryPass();

/**
 * @brief Registers the shader-to-SPIRV lowering pass.
 *
 * Converts shader texture ops and resource types into the SPIR-V dialect
 * and wraps the result in a spirv.module for serialization.
 */
void registerShaderToSPIRVPass();

/**
 * @brief Registers the spirv.module wrapping pass.
 *
 * Wraps spirv.func ops in a spirv.module for SPIR-V serialization.
 */
void registerShaderWrapSPIRVModulePass();

/**
 * @brief Registers the divergence analysis pass.
 *
 * Classifies SSA values as uniform/varying and detects texture samples
 * inside divergent control flow (scf.if / scf.for with varying conditions).
 */
void registerShaderDivergenceAnalysisPass();

/**
 * @brief Registers the divergent sample hoisting pass.
 *
 * Hoists shader.texture_sample ops out of scf.if regions whose
 * conditions are varying, so implicit derivatives are well-defined.
 */
void registerShaderHoistDivergentSamplesPass();

/**
 * @brief Registers all passes associated with the Shader dialect in the global registry.
 */
void registerShaderPasses();

} // namespace shader
} // namespace mlir

#endif // SHADER_MLIR_DIALECT_SHADER_TRANSFORMS_PASSES_H
