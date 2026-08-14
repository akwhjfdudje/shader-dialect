/**
 * @file ShaderDivergenceAnalysis.cpp
 * @brief Diagnostic pass: emits warnings for divergent texture samples.
 *
 * Delegates classification to the shared DivergenceInfo analysis.
 */

#include "ShaderMLIR/Dialect/Shader/Transforms/Passes.h"

#include "ShaderMLIR/Dialect/Shader/Analysis/DivergenceAnalysis.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"

using namespace mlir;
using namespace mlir::shader;

namespace {

struct ShaderDivergenceAnalysisPass
    : public PassWrapper<ShaderDivergenceAnalysisPass,
                         OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(ShaderDivergenceAnalysisPass)

  StringRef getArgument() const final {
    return "shader-divergence-analysis";
  }
  StringRef getDescription() const final {
    return "Classify values as uniform/varying and detect texture "
           "samples in divergent control flow";
  }
  void getDependentDialects(DialectRegistry &r) const final {
    r.insert<scf::SCFDialect>();
  }

  void runOnOperation() final {
    ModuleOp module = getOperation();

    module.walk([&](func::FuncOp func) {
      DivergenceInfo info = DivergenceInfo::compute(func);
      for (Operation *op : info.flaggedSamples) {
        op->emitWarning()
            << "shader.texture_sample inside divergent control flow: "
            << "implicit derivatives are undefined";
      }
    });
  }
};

} // namespace

void mlir::shader::registerShaderDivergenceAnalysisPass() {
  PassRegistration<ShaderDivergenceAnalysisPass>();
}
