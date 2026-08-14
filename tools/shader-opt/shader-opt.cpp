//===- shader-opt.cpp - Shader semantic MLIR optimizer driver ------------===//

/**
 * @file shader-opt.cpp
 * @brief Command-line tool driver for the Shader MLIR optimizer.
 *
 * Instantiates the main entrypoint for the optimizer driver, registering the
 * custom Shader dialect, standard dialects (Func, Arith, Vector), and target
 * transforms passes.
 */

#include "ShaderMLIR/Dialect/Shader/IR/ShaderDialect.h"
#include "ShaderMLIR/Dialect/Shader/Transforms/Passes.h"
#include "mlir/Conversion/ArithToSPIRV/ArithToSPIRV.h"
#include "mlir/Conversion/FuncToSPIRV/FuncToSPIRV.h"
#include "mlir/Conversion/VectorToSPIRV/VectorToSPIRV.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/SPIRV/IR/SPIRVDialect.h"
#include "mlir/Dialect/SPIRV/Transforms/SPIRVConversion.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/Tools/mlir-opt/MlirOptMain.h"
#include "mlir/Transforms/Passes.h"

/**
 * @brief Main compiler tool driver entrypoint.
 *
 * Configures the dialect registry, inserts required dialects, registers
 * transformation passes, and handles the CLI optimizer loop.
 *
 * @param argc Number of command line arguments.
 * @param argv Command line argument values.
 * @return Exit code representing optimizer run success/failure.
 */
int main(int argc, char **argv) {
  mlir::DialectRegistry registry;
  registry.insert<mlir::shader::ShaderDialect, mlir::arith::ArithDialect,
                  mlir::func::FuncDialect, mlir::scf::SCFDialect,
                  mlir::spirv::SPIRVDialect, mlir::vector::VectorDialect>();
  mlir::registerTransformsPasses();
  mlir::shader::registerShaderPasses();
  return mlir::asMainReturnCode(
      mlir::MlirOptMain(argc, argv, "Shader semantic MLIR optimizer driver\n",
                        registry));
}
