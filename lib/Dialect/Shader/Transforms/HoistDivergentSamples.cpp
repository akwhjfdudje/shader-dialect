/**
 * @file HoistDivergentSamples.cpp
 * @brief Hoist texture_sample ops out of divergent scf.if regions.
 *
 * Uses the shared DivergenceInfo analysis for classification.
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

/// Hoist divergent texture_sample ops from a region.
static bool hoistInRegion(Region &region,
                           const llvm::DenseSet<Value> &varying) {
  bool changed = false;

  for (Block &block : region) {
    for (auto it = block.begin(), e = block.end(); it != e;) {
      auto ifOp = dyn_cast<scf::IfOp>(&*it);
      if (!ifOp) {
        for (auto &sub : it->getRegions())
          changed |= hoistInRegion(sub, varying);
        ++it;
        continue;
      }

      // Recurse into nested regions inside the if first.
      for (auto &sub : it->getRegions())
        changed |= hoistInRegion(sub, varying);

      // Skip uniform branches.
      if (!varying.contains(ifOp.getCondition())) {
        ++it;
        continue;
      }

      // Collect ordinary texture_sample ops inside the branches.
      SmallVector<TextureSampleOp> toHoist;
      for (auto &b : ifOp.getThenRegion())
        for (auto &op : b)
          if (auto s = dyn_cast<TextureSampleOp>(&op))
            toHoist.push_back(s);
      if (!ifOp.getElseRegion().empty())
        for (auto &b : ifOp.getElseRegion())
          for (auto &op : b)
            if (auto s = dyn_cast<TextureSampleOp>(&op))
              toHoist.push_back(s);

      if (toHoist.empty()) {
        ++it;
        continue;
      }

      // Hoist: move each sample before the scf.if.
      for (auto sampleOp : toHoist) {
        sampleOp->moveBefore(ifOp);
        changed = true;
      }
      ++it;
    }
  }
  return changed;
}

// ── Pass ─────────────────────────────────────────────────────────────

struct HoistDivergentSamplesPass
    : public PassWrapper<HoistDivergentSamplesPass,
                         OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(HoistDivergentSamplesPass)

  StringRef getArgument() const final {
    return "shader-hoist-divergent-samples";
  }
  StringRef getDescription() const final {
    return "Hoist texture_sample ops out of divergent scf.if regions";
  }
  void getDependentDialects(DialectRegistry &r) const final {
    r.insert<scf::SCFDialect>();
  }

  void runOnOperation() final {
    ModuleOp module = getOperation();

    module.walk([&](func::FuncOp func) {
      DivergenceInfo info = DivergenceInfo::compute(func);
      hoistInRegion(func.getBody(), info.varying);
    });
  }
};

} // namespace

void mlir::shader::registerShaderHoistDivergentSamplesPass() {
  PassRegistration<HoistDivergentSamplesPass>();
}
