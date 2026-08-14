//===- ShaderDialect.h - Generic shader semantic dialect -------*- C++ -*-===//

#ifndef SHADER_MLIR_DIALECT_SHADER_IR_SHADERDIALECT_H
#define SHADER_MLIR_DIALECT_SHADER_IR_SHADERDIALECT_H

#include "mlir/IR/Dialect.h"

/**
 * @file ShaderDialect.h
 * @brief Declares the main Dialect class for Shader MLIR.
 */

namespace mlir {
namespace shader {

/**
 * @class ShaderDialect
 * @brief Main dialect class for preserving and optimizing shader-specific semantics.
 */
class ShaderDialect;

} // namespace shader
} // namespace mlir

#include "ShaderMLIR/Dialect/Shader/IR/ShaderOpsDialect.h.inc"

#endif // SHADER_MLIR_DIALECT_SHADER_IR_SHADERDIALECT_H
