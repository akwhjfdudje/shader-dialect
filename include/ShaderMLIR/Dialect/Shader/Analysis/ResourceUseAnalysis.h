//===- ResourceUseAnalysis.h - Shared resource-use analysis -----------*- C++ -*-===//
#ifndef SHADER_MLIR_DIALECT_SHADER_ANALYSIS_RESOURCE_USE_H
#define SHADER_MLIR_DIALECT_SHADER_ANALYSIS_RESOURCE_USE_H

#include "ShaderMLIR/Dialect/Shader/IR/ShaderOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "llvm/ADT/BitVector.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/MapVector.h"
#include "llvm/ADT/SetVector.h"

namespace mlir {
namespace shader {

struct ResourceKey {
  Value texture;
  Value sampler;
  bool operator==(const ResourceKey &o) const {
    return texture == o.texture && sampler == o.sampler;
  }
};

struct PerResourceInfo {
  unsigned ordinarySamples = 0;
  unsigned reusableSamples = 0;
  llvm::SetVector<StringRef> channels;
};

/// Shared analysis result consumed by ResourceUseSummary and
/// DeadResourcePruning.  One instance per ModuleOp, populated once.
struct ResourceUseInfo  {

  /// For each function: which resource-handle arguments are unused.
  llvm::DenseMap<StringRef, llvm::BitVector> deadArgs;

  /// For each function: per-(texture, sampler) pair usage.
  using PerTextureMap = llvm::MapVector<Value, llvm::MapVector<Value, PerResourceInfo>>;
  llvm::DenseMap<StringRef, PerTextureMap> perFuncResources;

  /// Compute the analysis for a module.  Must be called from a pass
  /// that has the module as its top-level operation.
  static ResourceUseInfo compute(ModuleOp module) {
    ResourceUseInfo info;

    module.walk([&](func::FuncOp func) {
      StringRef name = func.getSymName();

      // Dead resource arguments.
      llvm::BitVector dead(func.getNumArguments());
      for (unsigned i = 0; i < func.getNumArguments(); ++i) {
        Type t = func.getArgument(i).getType();
        if ((isa<TextureType, SamplerType, BufferType>(t)) &&
            func.getArgument(i).use_empty())
          dead.set(i);
      }
      info.deadArgs[name] = std::move(dead);

      // Resource usage per (texture, sampler) pair.
      PerTextureMap resources;
      llvm::DenseMap<Operation *, ResourceKey> sampleToKey;

      func.walk([&](Operation *op) {
        if (auto s = dyn_cast<TextureSampleOp>(op)) {
          ResourceKey key{s.getTexture(), s.getSampler()};
          resources[key.texture][key.sampler].ordinarySamples++;
          sampleToKey[op] = key;
        } else if (auto s = dyn_cast<TextureSampleReusableOp>(op)) {
          ResourceKey key{s.getTexture(), s.getSampler()};
          resources[key.texture][key.sampler].reusableSamples++;
          sampleToKey[op] = key;
        } else if (auto ch = dyn_cast<ChannelOp>(op)) {
          auto it = sampleToKey.find(ch.getSample().getDefiningOp());
          if (it != sampleToKey.end()) {
            resources[it->second.texture][it->second.sampler]
                .channels.insert(
                    stringifyChannel(static_cast<Channel>(ch.getChannel())));
          }
        }
      });

      info.perFuncResources[name] = std::move(resources);
    });

    return info;
  }
};

} // namespace shader
} // namespace mlir

#endif // SHADER_MLIR_DIALECT_SHADER_ANALYSIS_RESOURCE_USE_H
