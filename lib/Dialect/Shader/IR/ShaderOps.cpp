//===- ShaderOps.cpp - Generic shader semantic operations -----------------===//

/**
 * @file ShaderOps.cpp
 * @brief Implements the custom operations and types for the Shader MLIR dialect.
 *
 * This file registers and instantiates the code generated from the ODS and
 * Declarative Type Definition specifications.
 */

#include "ShaderMLIR/Dialect/Shader/IR/ShaderOps.h"

#include "llvm/ADT/StringSwitch.h"
#include "llvm/ADT/TypeSwitch.h"

using namespace mlir;
using namespace mlir::shader;

// ─── Helpers ───

bool mlir::shader::isFloatVectorOfSize(mlir::Type type, unsigned size) {
  auto vecType = llvm::dyn_cast<mlir::VectorType>(type);
  if (!vecType || vecType.getRank() != 1)
    return false;
  return vecType.getDimSize(0) == size &&
         llvm::isa<mlir::FloatType>(vecType.getElementType());
}

bool mlir::shader::isFloatVector(mlir::Type type) {
  auto vecType = llvm::dyn_cast<mlir::VectorType>(type);
  if (!vecType || vecType.getRank() != 1)
    return false;
  return llvm::isa<mlir::FloatType>(vecType.getElementType());
}

llvm::StringRef mlir::shader::stringifyChannel(Channel ch) {
  switch (ch) {
  case Channel::R:    return "r";
  case Channel::G:    return "g";
  case Channel::B:    return "b";
  case Channel::A:    return "a";
  case Channel::RG:   return "rg";
  case Channel::RGB:  return "rgb";
  case Channel::RGBA: return "rgba";
  }
  return "";
}

std::optional<Channel> mlir::shader::symbolizeChannel(llvm::StringRef str) {
  return llvm::StringSwitch<std::optional<Channel>>(str)
      .Case("r",    Channel::R)
      .Case("g",    Channel::G)
      .Case("b",    Channel::B)
      .Case("a",    Channel::A)
      .Case("rg",   Channel::RG)
      .Case("rgb",  Channel::RGB)
      .Case("rgba", Channel::RGBA)
      .Default(std::nullopt);
}

// ─── Internal helpers for ChannelOp ───

static std::optional<unsigned> channelIndex(Channel channel) {
  switch (channel) {
  case Channel::R:  return 0;
  case Channel::G:  return 1;
  case Channel::B:  return 2;
  case Channel::A:  return 3;
  default:          return std::nullopt; // RG, RGB, RGBA
  }
}

static unsigned channelCount(Channel channel) {
  switch (channel) {
  case Channel::RG:   return 2;
  case Channel::RGB:  return 3;
  case Channel::RGBA: return 4;
  default:            return 1; // R, G, B, A
  }
}

// ─── ChannelOp verifier ───

LogicalResult ChannelOp::verify() {
  int32_t raw = static_cast<int32_t>(getChannel());
  if (raw < 0 || raw > 6)
    return emitOpError("channel value out of range, must be 0-6");

  auto ch = static_cast<Channel>(raw);
  auto vecType = llvm::dyn_cast<VectorType>(getSample().getType());
  if (!vecType || vecType.getRank() != 1)
    return emitOpError("input must be a 1-D vector");

  unsigned dimSize = vecType.getDimSize(0);

  if (auto idx = channelIndex(ch)) {
    // Single channel: index must be strictly less than the vector dim.
    if (*idx >= dimSize)
      return emitOpError("channel index ")
             << *idx << " exceeds input vector dimension " << dimSize;

    // Single channel: result must be the input vector's element type (f32).
    auto expectedResult = vecType.getElementType();
    if (getResult().getType() != expectedResult)
      return emitOpError("result type must be ")
             << expectedResult << " for channel '" << stringifyChannel(ch)
             << "', got " << getResult().getType();
  } else {
    // Multi channel: the vector must have at least that many elements.
    unsigned needed = channelCount(ch);
    if (dimSize < needed)
      return emitOpError("input vector has ")
             << dimSize << " elements, but channel " << stringifyChannel(ch)
             << " needs at least " << needed;

    // Multi channel: result must be a 1-D float vector of the right size.
    auto expectedResult = VectorType::get({needed}, vecType.getElementType());
    if (getResult().getType() != expectedResult)
      return emitOpError("result type must be ")
             << expectedResult << " for channel '" << stringifyChannel(ch)
             << "', got " << getResult().getType();
  }

  return success();
}

// ─── Generated boilerplate ───

#define GET_TYPEDEF_CLASSES
#include "ShaderMLIR/Dialect/Shader/IR/ShaderOpsTypes.cpp.inc"

#define GET_OP_CLASSES
#include "ShaderMLIR/Dialect/Shader/IR/ShaderOps.cpp.inc"
