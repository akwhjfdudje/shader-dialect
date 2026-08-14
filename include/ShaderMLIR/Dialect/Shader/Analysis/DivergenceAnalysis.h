//===- DivergenceAnalysis.h - Shared uniform/varying classification -----*- C++ -*-===//
#ifndef SHADER_MLIR_DIALECT_SHADER_ANALYSIS_DIVERGENCE_H
#define SHADER_MLIR_DIALECT_SHADER_ANALYSIS_DIVERGENCE_H

#include "ShaderMLIR/Dialect/Shader/IR/ShaderOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinTypes.h"
#include "llvm/ADT/DenseSet.h"

namespace mlir {
namespace shader {

/// Shared analysis result consumed by ShaderDivergenceAnalysis and
/// HoistDivergentSamples.  Computed per-function.
struct DivergenceInfo {
  /// Values classified as varying (everything else is uniform).
  llvm::DenseSet<Value> varying;

  /// Operations that appear inside a divergent control-flow region
  /// (scf.if with varying condition, scf.for with varying bound/step).
  llvm::DenseSet<Operation *> divergentOps;

  /// List of texture_sample ops flagged inside divergent regions,
  /// for the diagnostic pass to report.
  SmallVector<Operation *> flaggedSamples;

  static bool isCoordinateType(Type t) {
    auto vt = dyn_cast<VectorType>(t);
    return vt && vt.getRank() == 1 && isa<FloatType>(vt.getElementType());
  }

  static bool anyVaryingOperand(Operation *op,
                                 const llvm::DenseSet<Value> &varying) {
    for (Value operand : op->getOperands())
      if (varying.contains(operand))
        return true;
    return false;
  }

  static void markResultsVarying(Operation *op,
                                  llvm::DenseSet<Value> &varying) {
    for (Value result : op->getResults())
      varying.insert(result);
  }

  /// Compute the analysis for a function.
  static DivergenceInfo compute(func::FuncOp func) {
    DivergenceInfo info;

    // Seed: coordinate arguments start varying.
    for (BlockArgument arg : func.getArguments())
      if (isCoordinateType(arg.getType()))
        info.varying.insert(arg);

    classifyRegion(func.getBody(), /*divergent=*/false, info);
    return info;
  }

private:
  static void classifyRegion(Region &region, bool divergent,
                              DivergenceInfo &info) {
    for (Block &block : region) {
      for (Operation &op : block) {

        // ── scf.if ──────────────────────────────────────────────
        if (auto ifOp = dyn_cast<scf::IfOp>(&op)) {
          bool condVarying = info.varying.contains(ifOp.getCondition());
          classifyRegion(ifOp.getThenRegion(), divergent || condVarying, info);
          if (!ifOp.getElseRegion().empty())
            classifyRegion(ifOp.getElseRegion(), divergent || condVarying,
                           info);
          if (anyVaryingOperand(&op, info.varying))
            markResultsVarying(&op, info.varying);
          continue;
        }

        // ── scf.for ─────────────────────────────────────────────
        if (auto forOp = dyn_cast<scf::ForOp>(&op)) {
          bool forDivergent = divergent ||
                              info.varying.contains(forOp.getLowerBound()) ||
                              info.varying.contains(forOp.getUpperBound()) ||
                              info.varying.contains(forOp.getStep());
          classifyRegion(forOp.getRegion(), forDivergent, info);
          if (forDivergent || anyVaryingOperand(&op, info.varying))
            markResultsVarying(&op, info.varying);
          continue;
        }

        // ── shader.texture_sample (ordinary) ────────────────────
        if (isa<TextureSampleOp>(op)) {
          if (divergent) {
            info.divergentOps.insert(&op);
            info.flaggedSamples.push_back(&op);
          }
          markResultsVarying(&op, info.varying);
          continue;
        }

        // ── shader.texture_sample_reusable ──────────────────────
        if (isa<TextureSampleReusableOp>(op)) {
          markResultsVarying(&op, info.varying);
          continue;
        }

        // ── propagation ─────────────────────────────────────────
        if (anyVaryingOperand(&op, info.varying))
          markResultsVarying(&op, info.varying);
      }
    }
  }
};

} // namespace shader
} // namespace mlir

#endif // SHADER_MLIR_DIALECT_SHADER_ANALYSIS_DIVERGENCE_H
