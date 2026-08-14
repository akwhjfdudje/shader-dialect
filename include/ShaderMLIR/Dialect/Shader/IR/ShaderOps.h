//===- ShaderOps.h - Generic shader semantic operations --------*- C++ -*-===//

#ifndef SHADER_MLIR_DIALECT_SHADER_IR_SHADEROPS_H
#define SHADER_MLIR_DIALECT_SHADER_IR_SHADEROPS_H

#include "ShaderMLIR/Dialect/Shader/IR/ShaderDialect.h"
#include "mlir/Bytecode/BytecodeOpInterface.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/Interfaces/InferTypeOpInterface.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

/**
 * @file ShaderOps.h
 * @brief Declares the custom operations and types for the Shader MLIR dialect.
 *
 * This file acts as the primary include for interacting with shader-specific
 * operations and types, pulling in the ODS generated classes.
 */

namespace mlir {
namespace shader {

/// Shader channel component selection.
enum class Channel : int32_t {
  R = 0,
  G = 1,
  B = 2,
  A = 3,
  RG = 4,
  RGB = 5,
  RGBA = 6,
};

/// Returns the string representation of a Channel enum value.
llvm::StringRef stringifyChannel(Channel ch);

/// Parses a string into a Channel enum value, returning std::nullopt
/// if the string is not recognized.
std::optional<Channel> symbolizeChannel(llvm::StringRef str);

/// Returns true if @p type is a 1-D float vector of length @p size
/// (e.g. vector<2xf32>, vector<4xf32>).
bool isFloatVectorOfSize(mlir::Type type, unsigned size);

/// Returns true if @p type is a 1-D float vector of any length.
bool isFloatVector(mlir::Type type);

} // namespace shader
} // namespace mlir

#define GET_TYPEDEF_CLASSES
#include "ShaderMLIR/Dialect/Shader/IR/ShaderOpsTypes.h.inc"

#define GET_OP_CLASSES
#include "ShaderMLIR/Dialect/Shader/IR/ShaderOps.h.inc"

#endif // SHADER_MLIR_DIALECT_SHADER_IR_SHADEROPS_H
