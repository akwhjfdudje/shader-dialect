/**
 * @file LowerToProxyBackend.cpp
 * @brief Lower shader semantics to proxy IR.
 *
 * This pass lowers high-level shader ops into standard arith and vector ops.
 */

#include "ShaderMLIR/Dialect/Shader/Transforms/Passes.h"

#include "ShaderMLIR/Dialect/Shader/IR/ShaderOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"

using namespace mlir;
using namespace mlir::shader;

namespace {

/**
 * @class LowerToProxyBackendPass
 * @brief Lowers shader dialect ops to standard MLIR dialects.
 *
 * Replaces high-level ops like saturate and channel extraction with arith and vector equivalents.
 */
struct LowerToProxyBackendPass
    : public PassWrapper<LowerToProxyBackendPass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(LowerToProxyBackendPass)

  StringRef getArgument() const final { return "shader-lower-to-proxy-backend"; }

  StringRef getDescription() const final {
    return "Lower CPU-measurable shader semantics while preserving samples as "
           "external backend costs";
  }

  void getDependentDialects(DialectRegistry &registry) const final {
    registry.insert<arith::ArithDialect, vector::VectorDialect>();
  }

  void runOnOperation() final {
    IRRewriter rewriter(&getContext());
    SmallVector<Operation *> ops;
    getOperation().walk([&](Operation *op) {
      if (isa<SaturateOp, ChannelOp, DerivativeHintOp>(op))
        ops.push_back(op);
    });

    for (Operation *op : ops) {
      if (auto saturate = dyn_cast<SaturateOp>(op)) {
        lowerSaturate(rewriter, saturate);
        continue;
      }
      if (auto channel = dyn_cast<ChannelOp>(op)) {
        lowerChannel(rewriter, channel);
        continue;
      }
      if (auto hint = dyn_cast<DerivativeHintOp>(op)) {
        rewriter.replaceOp(hint, hint.getInput());
        continue;
      }
    }
  }

  /**
   * @brief Returns true if the type is a float or a vector of floats.
   * @param type The type to inspect.
   * @return True if lowering to arith float ops is supported for this type.
   */
  static bool supportsFloatMinMax(Type type) {
    if (isa<FloatType>(type))
      return true;
    auto vectorType = dyn_cast<VectorType>(type);
    return vectorType && isa<FloatType>(vectorType.getElementType());
  }

  /**
   * @brief Creates a float attribute (scalar or splat vector) for a given value.
   * @param builder The IR builder to use.
   * @param type The target type for the attribute.
   * @param value The constant float value.
   * @return A TypedAttr representing the value in the target type.
   */
  static TypedAttr floatLikeAttr(OpBuilder &builder, Type type, double value) {
    if (auto floatType = dyn_cast<FloatType>(type))
      return builder.getFloatAttr(floatType, value);

    auto vectorType = dyn_cast<VectorType>(type);
    if (!vectorType || !isa<FloatType>(vectorType.getElementType()))
      return nullptr;

    Attribute element = builder.getFloatAttr(vectorType.getElementType(), value);
    return DenseElementsAttr::get(vectorType, element);
  }

  /**
   * @brief Lowers a saturate operation to arith.maximumf and arith.minimumf.
   * @param rewriter The rewriter to perform the transformation.
   * @param op The SaturateOp to lower.
   */
  static void lowerSaturate(IRRewriter &rewriter, SaturateOp op) {
    Type type = op.getType();
    if (!supportsFloatMinMax(type))
      return;

    TypedAttr zeroAttr = floatLikeAttr(rewriter, type, 0.0);
    TypedAttr oneAttr = floatLikeAttr(rewriter, type, 1.0);
    if (!zeroAttr || !oneAttr)
      return;

    Location loc = op.getLoc();
    rewriter.setInsertionPoint(op);
    Value zero = rewriter.create<arith::ConstantOp>(loc, type, zeroAttr);
    Value one = rewriter.create<arith::ConstantOp>(loc, type, oneAttr);
    Value lowerBound =
        rewriter.create<arith::MaximumFOp>(loc, op.getInput(), zero);
    Value clamped = rewriter.create<arith::MinimumFOp>(loc, lowerBound, one);
    rewriter.replaceOp(op, clamped);
  }

  /**
   * @brief Maps single-channel enum values (R, G, B, A) to integer indices.
   * @param channel The channel enum value.
   * @return The corresponding index (0-3) or nullopt for multi-channel values.
   */
  static std::optional<int64_t> channelIndex(Channel channel) {
    switch (channel) {
    case Channel::R: return 0;
    case Channel::G: return 1;
    case Channel::B: return 2;
    case Channel::A: return 3;
    default: return std::nullopt;
    }
  }

  /**
   * @brief Lowers a channel extraction op to vector.extract.
   * @param rewriter The rewriter to perform the transformation.
   * @param op The ChannelOp to lower.
   */
  static void lowerChannel(IRRewriter &rewriter, ChannelOp op) {
    auto index = channelIndex(static_cast<Channel>(op.getChannel()));
    auto vectorType = dyn_cast<VectorType>(op.getSample().getType());

    // nothing here, or type mismatch. either way, break.
    if (!index || !vectorType || vectorType.getRank() != 1 ||
        vectorType.getDimSize(0) <= *index ||
        vectorType.getElementType() != op.getType())
      return;

    // something here
    rewriter.setInsertionPoint(op);
    Value extracted =
        rewriter.create<vector::ExtractOp>(op.getLoc(), op.getSample(), *index);
    rewriter.replaceOp(op, extracted);
  }
};

} // namespace

void mlir::shader::registerShaderLowerToProxyBackendPass() {
  PassRegistration<LowerToProxyBackendPass>();
}
