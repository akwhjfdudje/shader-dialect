//===- ShaderOps.td - Generic shader semantic operations ----*- tablegen -*-===//

#ifndef SHADER_MLIR_DIALECT_SHADER_IR_SHADEROPS_TD
#define SHADER_MLIR_DIALECT_SHADER_IR_SHADEROPS_TD

include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/OpBase.td"
include "mlir/Interfaces/InferTypeOpInterface.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

def Shader_Dialect : Dialect {
  let name = "shader";
  let cppNamespace = "::mlir::shader";
  let useDefaultTypePrinterParser = 1;
  let summary = "Generic shader semantic dialect";
  let description = [{
    The Shader dialect models shader-language semantics that should survive
    before lowering into generic math, GPU, or SPIR-V dialects. It is intended
    to explore whether semantics common to shader languages can be preserved and
    analyzed usefully in MLIR, independent of any single source language.
  }];
}

class Shader_Type<string name, string typeMnemonic>
    : TypeDef<Shader_Dialect, name> {
  let mnemonic = typeMnemonic;
}

def Shader_TextureType : Shader_Type<"Texture", "texture"> {
  let summary = "opaque shader texture resource handle";
}

def Shader_SamplerType : Shader_Type<"Sampler", "sampler"> {
  let summary = "opaque shader sampler resource handle";
}

def Shader_BufferType : Shader_Type<"Buffer", "buffer"> {
  let summary = "opaque shader buffer resource handle";
}

def Shader_TextureLike : Type<CPred<"::llvm::isa<::mlir::shader::TextureType>($_self)">,
                              "shader texture">;
def Shader_SamplerLike : Type<CPred<"::llvm::isa<::mlir::shader::SamplerType>($_self)">,
                              "shader sampler">;

// ─── Type constraints ───

def Shader_CoordType : Type<
    CPred<"::mlir::shader::isFloatVectorOfSize($_self, 2) || "
          "::mlir::shader::isFloatVectorOfSize($_self, 3)">,
    "2D or 3D float vector (vector<2xf32> or vector<3xf32>)">;

def Shader_TextureSampleResult : Type<
    CPred<"::mlir::shader::isFloatVectorOfSize($_self, 4)">,
    "4-element float vector (vector<4xf32>)">;

def Shader_SampleVector : Type<
    CPred<"::mlir::shader::isFloatVector($_self)">,
    "1-D float vector">;

// ─── Ops ───

class Shader_Op<string mnemonic, list<Trait> traits = []> :
    Op<Shader_Dialect, mnemonic, traits>;

def TextureSampleOp : Shader_Op<"texture_sample"> {
  let summary = "ordinary shader texture sample";
  let description = [{
    Represents a texture sample whose reuse is not known to be legal. The op is
    intentionally not marked Pure, so generic CSE and motion passes should not
    merge it without a shader-specific legality proof.
  }];

  let arguments = (ins Shader_TextureLike:$texture,
                       Shader_SamplerLike:$sampler,
                       Shader_CoordType:$coords);
  let results = (outs Shader_TextureSampleResult:$result);
}

def TextureSampleReusableOp : Shader_Op<"texture_sample_reusable", [Pure]> {
  let summary = "texture sample with explicit reuse legality";
  let description = [{
    Represents a texture sample that source-language or shader-side analysis has
    proven reusable:
    same resource, sampler, coordinates, LOD/derivative context, and no semantic
    barrier. Marking this op Pure lets standard MLIR CSE merge identical samples.
  }];

  let arguments = (ins Shader_TextureLike:$texture,
                       Shader_SamplerLike:$sampler,
                       Shader_CoordType:$coords);
  let results = (outs Shader_TextureSampleResult:$result);
}

def ChannelOp : Shader_Op<"channel", [Pure]> {
  let summary = "extract a named channel from a shader value";
  let description = [{
    Extracts a named component (r, g, b, a) or multi-channel combination
    (rg, rgb, rgba) from a shader vector value. The channel is selected via a
    dialect integer attribute that encodes the Channel enum.
    The verifier ensures the channel index does not exceed the input vector
    dimensions.
  }];

  let arguments = (ins Shader_SampleVector:$sample, I32Attr:$channel);
  let results = (outs AnyType:$result);

  let hasVerifier = 1;
}

def SaturateOp : Shader_Op<"saturate", [Pure, SameOperandsAndResultType]> {
  let summary = "shader saturate intrinsic";
  let description = [{
    Models saturate(x), preserving the shader intrinsic until a target-specific
    lowering decides whether to emit clamp, min/max, or a native instruction.
  }];

  let arguments = (ins AnyFloat:$input);
  let results = (outs AnyFloat:$result);
}

def DerivativeHintOp : Shader_Op<"derivative_hint", [SameOperandsAndResultType]> {
  let summary = "derivative-sensitive semantic marker";
  let description = [{
    Marks a value as derivative-sensitive. This op is intentionally not Pure in
    the initial dialect so generic passes do not freely remove, merge, or move it
    across operations that may affect implicit derivative behavior.
  }];

  let arguments = (ins Shader_CoordType:$input);
  let results = (outs Shader_CoordType:$result);
}

#endif // SHADER_MLIR_DIALECT_SHADER_IR_SHADEROPS_TD
