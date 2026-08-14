/**
 * @file DeadResourcePruning.cpp
 * @brief Bottom-up dead resource pruning via fixed-point iteration.
 *
 * Uses ResourceUseInfo (shared with ResourceUseSummary) to identify
 * unused resource arguments, then prunes and propagates through call
 * chains.
 */

#include "ShaderMLIR/Dialect/Shader/Transforms/Passes.h"

#include "ShaderMLIR/Dialect/Shader/Analysis/ResourceUseAnalysis.h"
#include "ShaderMLIR/Dialect/Shader/IR/ShaderOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/BitVector.h"

using namespace mlir;
using namespace mlir::shader;

namespace {

struct DeadResourcePruningPass
    : public PassWrapper<DeadResourcePruningPass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(DeadResourcePruningPass)

  StringRef getArgument() const final { return "shader-prune-dead-resources"; }

  StringRef getDescription() const final {
    return "Bottom-up dead resource pruning via fixed-point iteration";
  }

  void runOnOperation() final {
    ModuleOp module = getOperation();
    bool changed = true;

    while (changed) {
      changed = false;
      ResourceUseInfo info = ResourceUseInfo::compute(module);

      module.walk([&](func::FuncOp func) {
        if (func.isExternal())
          return;

        auto it = info.deadArgs.find(func.getSymName());
        if (it == info.deadArgs.end() || it->second.none())
          return;

        const BitVector &dead = it->second;

        // Erase corresponding operands at call sites.
        module.walk([&](func::CallOp call) {
          if (call.getCallee() == func.getSymName())
            call->eraseOperands(dead);
        });

        func.eraseArguments(dead);
        changed = true;
      });
    }
  }
};

} // namespace

void mlir::shader::registerShaderDeadResourcePruningPass() {
  PassRegistration<DeadResourcePruningPass>();
}
