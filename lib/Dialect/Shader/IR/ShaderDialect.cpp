/**
 * @file ShaderDialect.cpp
 * @brief Implementation of the Shader dialect's initialization and registration.
 *
 * This file registers the custom types and operations defined in the shader dialect.
 */

#include "ShaderMLIR/Dialect/Shader/IR/ShaderDialect.h"
#include "ShaderMLIR/Dialect/Shader/IR/ShaderOps.h"

using namespace mlir;
using namespace mlir::shader;

#include "ShaderMLIR/Dialect/Shader/IR/ShaderOpsDialect.cpp.inc"

/**
 * @brief Initializes the Shader dialect.
 * 
 * Adds all ODS-generated types and operations to the dialect.
 */
void ShaderDialect::initialize() {
  addTypes<
#define GET_TYPEDEF_LIST
#include "ShaderMLIR/Dialect/Shader/IR/ShaderOpsTypes.cpp.inc"
      >();

  addOperations<
#define GET_OP_LIST
#include "ShaderMLIR/Dialect/Shader/IR/ShaderOps.cpp.inc"
      >();
}
