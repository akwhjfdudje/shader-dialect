/**
 * @file ResourceUseSummary.cpp
 * @brief Prints shader resource usage from the shared ResourceUseInfo analysis.
 */

#include "ShaderMLIR/Dialect/Shader/Transforms/Passes.h"

#include "ShaderMLIR/Dialect/Shader/Analysis/ResourceUseAnalysis.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/AsmState.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
#include "llvm/Support/raw_ostream.h"

using namespace mlir;
using namespace mlir::shader;

namespace {

struct ResourceUseSummaryPass
    : public PassWrapper<ResourceUseSummaryPass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(ResourceUseSummaryPass)

  StringRef getArgument() const final { return "shader-resource-summary"; }
  StringRef getDescription() const final {
    return "Print a summary of texture/sampler resources used by shader ops";
  }

  void runOnOperation() final {
    ModuleOp module = getOperation();
    ResourceUseInfo info = ResourceUseInfo::compute(module);
    AsmState asmState(module);

    llvm::outs() << "shader.resource_summary {\n";

    for (auto &[funcName, resources] : info.perFuncResources) {
      llvm::outs() << "  func @" << funcName << " {\n";
      if (resources.empty()) {
        llvm::outs() << "    no texture resources\n";
      } else {
        for (auto &[texture, samplerMap] : resources) {
          for (auto &[sampler, ri] : samplerMap) {
            llvm::outs() << "    resource(texture=" << argName(texture, asmState)
                         << ", sampler=" << argName(sampler, asmState) << ") "
                         << "ordinary_samples=" << ri.ordinarySamples
                         << " reusable_samples=" << ri.reusableSamples
                         << " channels=[";
            llvm::interleaveComma(ri.channels, llvm::outs(),
                                  [](StringRef c) { llvm::outs() << c; });
            llvm::outs() << "]\n";
          }
        }
      }
      llvm::outs() << "  }\n";
    }
    llvm::outs() << "}\n";
  }

  static std::string argName(Value value, AsmState &asmState) {
    if (auto arg = dyn_cast<BlockArgument>(value)) {
      if (auto func = dyn_cast<func::FuncOp>(arg.getOwner()->getParentOp())) {
        std::string name;
        llvm::raw_string_ostream os(name);
        os << "@" << func.getSymName() << ".arg" << arg.getArgNumber();
        return name;
      }
    }
    std::string name;
    llvm::raw_string_ostream os(name);
    value.printAsOperand(os, asmState);
    return name;
  }
};

} // namespace

void mlir::shader::registerShaderResourceSummaryPass() {
  PassRegistration<ResourceUseSummaryPass>();
}

void mlir::shader::registerShaderPasses() {
  registerShaderDeadResourcePruningPass();
  registerShaderDivergenceAnalysisPass();
  registerShaderHoistDivergentSamplesPass();
  registerShaderLowerToProxyBackendPass();
  registerShaderResourceSummaryPass();
  registerShaderToSPIRVPass();
  registerShaderWrapSPIRVModulePass();
}
