/**
 * @file LowerToSPIRV.cpp
 * @brief Lower shader dialect texture types/ops to SPIR-V dialect.
 *
 * Converts !shader.texture -> !spirv.image, !shader.sampler -> !spirv.sampler,
 * shader.texture_sample -> spirv.SampledImage + spirv.ImageSampleImplicitLod,
 * and shader.texture_sample_reusable -> same.
 *
 * Output stays in func.func / builtin.module — no spirv.module wrapping.
 * This is the first stage of SPIR-V lowering. The second stage (func.func ->
 * spirv.func + arith/vector -> SPIR-V) is a separate pass.
 */

#include "ShaderMLIR/Dialect/Shader/Transforms/Passes.h"

#include "ShaderMLIR/Dialect/Shader/IR/ShaderOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SPIRV/IR/SPIRVDialect.h"
#include "mlir/Dialect/SPIRV/IR/SPIRVOps.h"
#include "mlir/Dialect/SPIRV/IR/SPIRVTypes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Pass/Pass.h"

using namespace mlir;
using namespace mlir::shader;

namespace {

static spirv::ImageType imageType(MLIRContext *ctx) {
  return spirv::ImageType::get(
      Float32Type::get(ctx), spirv::Dim::Dim2D,
      spirv::ImageDepthInfo::NoDepth,
      spirv::ImageArrayedInfo::NonArrayed,
      spirv::ImageSamplingInfo::SingleSampled,
      spirv::ImageSamplerUseInfo::SamplerUnknown,
      spirv::ImageFormat::Unknown);
}

struct ShaderToSPIRVPass
    : public PassWrapper<ShaderToSPIRVPass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(ShaderToSPIRVPass)

  StringRef getArgument() const final { return "shader-to-spirv"; }

  StringRef getDescription() const final {
    return "Lower shader texture types and ops to SPIR-V dialect";
  }

  void getDependentDialects(DialectRegistry &registry) const final {
    registry.insert<spirv::SPIRVDialect>();
  }

  void runOnOperation() final {
    ModuleOp module = getOperation();
    MLIRContext *ctx = &getContext();
    auto imgTy = imageType(ctx);
    auto sampTy = spirv::SamplerType::get(ctx);

    SmallVector<Operation *> textureOps;
    module.walk([&](Operation *op) {
      if (isa<TextureSampleOp, TextureSampleReusableOp>(op))
        textureOps.push_back(op);
    });

    for (Operation *op : textureOps) {
      OpBuilder builder(op);
      Location loc = op->getLoc();
      auto sampledImageTy = spirv::SampledImageType::get(imgTy);
      auto sampledImage = builder.create<spirv::SampledImageOp>(
          loc, sampledImageTy, op->getOperand(0), op->getOperand(1));
      Value coords = op->getOperand(2);
      auto resultTy = op->getResult(0).getType();
      auto sample = builder.create<spirv::ImageSampleImplicitLodOp>(
          loc, resultTy, sampledImage, coords,
          /*image_operands=*/nullptr,
          /*operand_arguments=*/ValueRange{});
      op->getResult(0).replaceAllUsesWith(sample);
      op->erase();
    }

    // Convert function argument types: !shader.texture -> !spirv.image, etc.
    module.walk([&](Operation *op) {
      for (auto &region : op->getRegions())
        for (auto &block : region)
          for (auto arg : block.getArguments()) {
            if (isa<TextureType>(arg.getType()))
              arg.setType(imgTy);
            else if (isa<SamplerType>(arg.getType()))
              arg.setType(sampTy);
          }
    });

    // Update FuncOp function type signatures
    SmallVector<func::FuncOp> funcs;
    module.walk([&](func::FuncOp f) { funcs.push_back(f); });
    for (auto func : funcs) {
      auto ft = func.getFunctionType();
      SmallVector<Type> inputs;
      bool changed = false;
      for (Type t : ft.getInputs()) {
        if (isa<TextureType>(t)) {
          inputs.push_back(imgTy);
          changed = true;
        } else if (isa<SamplerType>(t)) {
          inputs.push_back(sampTy);
          changed = true;
        } else {
          inputs.push_back(t);
        }
      }
      if (changed) {
        SmallVector<Type> results(ft.getResults().begin(),
                                  ft.getResults().end());
        func.setType(FunctionType::get(ctx, inputs, results));
      }
    }
  }
};

} // namespace

void mlir::shader::registerShaderToSPIRVPass() {
  PassRegistration<ShaderToSPIRVPass>();
}
