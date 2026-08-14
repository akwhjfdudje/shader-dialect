/**
 * @file WrapSPIRVModule.cpp
 * @brief Wrap func.func and spirv.func in spirv.module with full interface
 *        variable lowering.
 *
 * Converts function arguments and return values into SPIR-V interface
 * variables:
 *   !spirv.image / !spirv.sampler  -> descriptor-bound GlobalVariable
 *   vector<Nxf32>                   -> Input GlobalVariable (location)
 *   f32 / i32 (scalars)             -> PushConstant struct GlobalVariable
 *   return values                   -> Output struct GlobalVariable (location)
 */

#include "ShaderMLIR/Dialect/Shader/Transforms/Passes.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SPIRV/IR/SPIRVDialect.h"
#include "mlir/Dialect/SPIRV/IR/SPIRVOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Pass/Pass.h"

using namespace mlir;

namespace {

enum ArgKind { Descriptor, InputVar, PushConstant };
struct ArgInfo { ArgKind kind; int binding; };

static bool isResourceType(Type type) {
  return isa<spirv::ImageType>(type) || isa<spirv::SamplerType>(type);
}
static bool isInputType(Type type) {
  auto vt = dyn_cast<VectorType>(type);
  return vt && vt.getRank() == 1 && isa<FloatType>(vt.getElementType());
}
static bool isPushConstantType(Type type) {
  return isa<FloatType>(type) || isa<IntegerType>(type);
}

struct WrapSPIRVModulePass
    : public PassWrapper<WrapSPIRVModulePass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(WrapSPIRVModulePass)
  StringRef getArgument() const final { return "shader-wrap-spirv-module"; }
  StringRef getDescription() const final {
    return "Wrap functions in spirv.module with full interface lowering";
  }
  void getDependentDialects(DialectRegistry &r) const final {
    r.insert<spirv::SPIRVDialect>();
  }

  void runOnOperation() final {
    ModuleOp module = getOperation();
    auto *ctx = &getContext();

    // 1. Collect functions
    SmallVector<func::FuncOp> funcFuncs;
    SmallVector<spirv::FuncOp> spvFuncs;
    module.walk([&](Operation *op) {
      if (auto ff = dyn_cast<func::FuncOp>(op))  funcFuncs.push_back(ff);
      if (auto sf = dyn_cast<spirv::FuncOp>(op)) spvFuncs.push_back(sf);
    });
    if (funcFuncs.empty() && spvFuncs.empty()) return;

    // 2. Collect argument types across all functions, assign bindings/locations
    DenseMap<Type, int> descBinding;     // descriptor set binding
    DenseMap<Type, int> inputLocation;   // input location
    SmallVector<Type> descTypes, inputTypes, pushTypes;

    auto collectArgs = [&](auto func) {
      for (auto arg : func.getArguments()) {
        Type t = arg.getType();
        if (isResourceType(t) && !descBinding.count(t)) {
          descBinding[t] = static_cast<int>(descTypes.size());
          descTypes.push_back(t);
        } else if (isInputType(t) && !inputLocation.count(t)) {
          inputLocation[t] = static_cast<int>(inputTypes.size());
          inputTypes.push_back(t);
        } else if (isPushConstantType(t)) {
          if (std::find(pushTypes.begin(), pushTypes.end(), t) == pushTypes.end())
            pushTypes.push_back(t);
        }
      }
    };
    for (auto ff : funcFuncs) collectArgs(ff);
    for (auto sf : spvFuncs) collectArgs(sf);

    // Build push constant struct type (one struct containing all scalar types)
    Type pcStructType;
    if (!pushTypes.empty()) {
      if (pushTypes.size() == 1)
        pcStructType = pushTypes[0];
      else
        pcStructType = spirv::StructType::get(
            SmallVector<Type>(pushTypes.begin(), pushTypes.end()));
    }

    // 3. Create spirv.module
    auto vce = spirv::VerCapExtAttr::get(
        spirv::Version::V_1_3,
        SmallVector<spirv::Capability>{spirv::Capability::Shader,
                                       spirv::Capability::Sampled1D},
        SmallVector<spirv::Extension>{}, ctx);
    OpBuilder builder = OpBuilder::atBlockBegin(module.getBody());
    auto spvModule = builder.create<spirv::ModuleOp>(
        module.getLoc(), spirv::AddressingModel::Logical,
        spirv::MemoryModel::GLSL450, vce);
    auto *spvBody = spvModule.getBody(0);

    // 4. Create all GlobalVariables
    //   4a. Descriptor-bound resources (UniformConstant)
    SmallVector<spirv::GlobalVariableOp> descGlobals;
    descGlobals.resize(descTypes.size());
    {
      OpBuilder gb(spvBody, spvBody->begin());
      for (auto &[type, binding] : descBinding) {
        auto ptrTy = spirv::PointerType::get(
            type, spirv::StorageClass::UniformConstant);
        auto sym = gb.getStringAttr("resource_" + std::to_string(binding));
        auto gv = spirv::GlobalVariableOp::create(
            gb, module.getLoc(), TypeAttr::get(ptrTy), sym, FlatSymbolRefAttr());
        gv->setAttr("descriptor_set", gb.getI32IntegerAttr(0));
        gv->setAttr("binding", gb.getI32IntegerAttr(binding));
        descGlobals[binding] = gv;
      }
    }

    //   4b. Input variables (fragment inputs, one per unique vector type)
    SmallVector<spirv::GlobalVariableOp> inputGlobals;
    inputGlobals.resize(inputTypes.size());
    {
      OpBuilder gb(spvBody, spvBody->begin());
      for (auto &[type, loc] : inputLocation) {
        auto ptrTy = spirv::PointerType::get(type, spirv::StorageClass::Input);
        auto sym = gb.getStringAttr("input_" + std::to_string(loc));
        auto gv = spirv::GlobalVariableOp::create(
            gb, module.getLoc(), TypeAttr::get(ptrTy), sym, FlatSymbolRefAttr());
        gv->setAttr("location", gb.getI32IntegerAttr(loc));
        inputGlobals[loc] = gv;
      }
    }

    //   4c. Push constant variable (one struct for all scalar args)
    spirv::GlobalVariableOp pcGlobal;
    if (pcStructType) {
      OpBuilder gb(spvBody, spvBody->begin());
      auto ptrTy = spirv::PointerType::get(
          pcStructType, spirv::StorageClass::PushConstant);
      auto gv = spirv::GlobalVariableOp::create(
          gb, module.getLoc(), TypeAttr::get(ptrTy),
          gb.getStringAttr("push_constants"), FlatSymbolRefAttr());
      pcGlobal = gv;
    }

    //   4d. Output variable (one struct for all return values, per function)
    //       We compute this per-function below and store them.

    // 5. Helper: classify each argument of a function
    auto classifyArgs = [&](auto func) -> SmallVector<ArgInfo> {
      SmallVector<ArgInfo> result;
      for (auto arg : func.getArguments()) {
        Type t = arg.getType();
        if (isResourceType(t))
          result.push_back({Descriptor, descBinding.lookup(t)});
        else if (isInputType(t))
          result.push_back({InputVar, inputLocation.lookup(t)});
        else
          result.push_back({PushConstant, 0});
      }
      return result;
    };

    // 6. Build output type (struct) for a result type range
    auto buildOutputType = [&](TypeRange results) -> Type {
      if (results.empty()) return builder.getNoneType();
      if (results.size() == 1) return results[0];
      return spirv::StructType::get(
          SmallVector<Type>(results.begin(), results.end()));
    };

    // 7. Process spirv.func ops
    SmallVector<spirv::GlobalVariableOp> outputGlobals;
    SmallVector<Type> outputTypes;
    DenseMap<Type, spirv::GlobalVariableOp> outputGvMap;

    for (auto sf : spvFuncs) {
      auto argInfo = classifyArgs(sf);
      auto resultType = sf.getResultTypes();
      Type outTy = buildOutputType(resultType);

      // Create output global if needed
      spirv::GlobalVariableOp outGv;
      if (outTy && !isa<NoneType>(outTy)) {
        if (outputGvMap.count(outTy)) {
          outGv = outputGvMap[outTy];
        } else {
          OpBuilder gb(spvBody, spvBody->begin());
          auto ptrTy = spirv::PointerType::get(
              outTy, spirv::StorageClass::Output);
          int loc = static_cast<int>(outputGvMap.size());
          auto sym = gb.getStringAttr("output_" + std::to_string(loc));
          auto gv = spirv::GlobalVariableOp::create(
              gb, module.getLoc(), TypeAttr::get(ptrTy), sym,
              FlatSymbolRefAttr());
          gv->setAttr("location", gb.getI32IntegerAttr(loc));
          outputGvMap[outTy] = gv;
          outputGlobals.push_back(gv);
          outGv = gv;
        }
      }

      // Replace args
      unsigned n = sf.getNumArguments();
      if (n > 0) {
        OpBuilder b(&sf.getBody().front(), sf.getBody().front().begin());
        for (unsigned i = 0; i < n; ++i) {
          auto arg = sf.getBody().front().getArgument(0);
          Type t = arg.getType();
          Value replacement;
          auto &info = argInfo[i];
          if (info.kind == Descriptor) {
            auto &gv = descGlobals[info.binding];
            auto addr = spirv::AddressOfOp::create(b, sf.getLoc(), gv);
            replacement = spirv::LoadOp::create(
                b, sf.getLoc(), t, addr.getResult(), spirv::MemoryAccessAttr(),
                IntegerAttr());
          } else if (info.kind == InputVar) {
            auto &gv = inputGlobals[info.binding];
            auto addr = spirv::AddressOfOp::create(b, sf.getLoc(), gv);
            replacement = spirv::LoadOp::create(
                b, sf.getLoc(), t, addr.getResult(), spirv::MemoryAccessAttr(),
                IntegerAttr());
          } else { // PushConstant
            if (!pcGlobal) {
              replacement = b.create<spirv::UndefOp>(sf.getLoc(), t);
            } else {
              auto addr = spirv::AddressOfOp::create(b, sf.getLoc(), pcGlobal);
              auto load = spirv::LoadOp::create(
                  b, sf.getLoc(), pcStructType, addr.getResult(),
                  spirv::MemoryAccessAttr(), IntegerAttr());
              // Find index of this type in pushTypes
              auto it = std::find(pushTypes.begin(), pushTypes.end(), t);
              int idx = static_cast<int>(it - pushTypes.begin());
              if (pushTypes.size() == 1)
                replacement = load;
              else
                replacement = b.create<spirv::CompositeExtractOp>(
                    sf.getLoc(), t, load, b.getI32ArrayAttr({idx}));
            }
          }
          arg.replaceAllUsesWith(replacement);
          sf.getBody().front().eraseArgument(0);
        }
      }
      sf.setType(FunctionType::get(ctx, {}, sf.getResultTypes()));

      // Replace return with store to output variable
      for (auto &block : sf.getBody()) {
        if (auto ret = block.getTerminator()) {
          OpBuilder rb(ret);
          if (outGv && ret->getNumOperands() > 0) {
            auto addr = spirv::AddressOfOp::create(rb, sf.getLoc(), outGv);
            Value storeVal;
            if (ret->getNumOperands() == 1) {
              storeVal = ret->getOperand(0);
            } else {
              storeVal = rb.create<spirv::CompositeConstructOp>(
                  ret->getLoc(), outTy, ret->getOperands());
            }
            rb.create<spirv::StoreOp>(sf.getLoc(), addr.getResult(), storeVal,
                                      spirv::MemoryAccessAttr(),
                                      IntegerAttr());
          }
          if (isa<spirv::ReturnValueOp>(ret)) {
            rb.create<spirv::ReturnOp>(ret->getLoc());
            ret->erase();
          }
          break;
        }
      }
      sf.setType(FunctionType::get(ctx, {}, {}));
      sf->moveBefore(spvBody, spvBody->end());
    }

    // 8. Process func.func ops (same logic but with func.return)
    for (auto ff : funcFuncs) {
      auto fnType = ff.getFunctionType();
      auto argInfo = classifyArgs(ff);
      Type outTy = buildOutputType(fnType.getResults());

      spirv::GlobalVariableOp outGv;
      if (outTy && !isa<NoneType>(outTy)) {
        if (outputGvMap.count(outTy)) {
          outGv = outputGvMap[outTy];
        } else {
          OpBuilder gb(spvBody, spvBody->begin());
          auto ptrTy = spirv::PointerType::get(outTy, spirv::StorageClass::Output);
          int loc = static_cast<int>(outputGvMap.size());
          auto sym = gb.getStringAttr("output_" + std::to_string(loc));
          auto gv = spirv::GlobalVariableOp::create(
              gb, module.getLoc(), TypeAttr::get(ptrTy), sym,
              FlatSymbolRefAttr());
          gv->setAttr("location", gb.getI32IntegerAttr(loc));
          outputGvMap[outTy] = gv;
          outputGlobals.push_back(gv);
          outGv = gv;
        }
      }

      builder.setInsertionPointToEnd(spvBody);
      auto spvFunc = builder.create<spirv::FuncOp>(
          ff.getLoc(), ff.getSymName(),
          FunctionType::get(ctx, fnType.getInputs(), outTy),
          spirv::FunctionControl::None);
      spvFunc.getBody().takeBody(ff.getBody());

      // Replace args by type
      unsigned n = spvFunc.getNumArguments();
      if (n > 0) {
        OpBuilder b(&spvFunc.getBody().front(),
                    spvFunc.getBody().front().begin());
        for (unsigned i = 0; i < n; ++i) {
          auto arg = spvFunc.getBody().front().getArgument(0);
          Type t = arg.getType();
          Value replacement;
          auto &info = argInfo[i];
          if (info.kind == Descriptor) {
            auto &gv = descGlobals[info.binding];
            auto addr = spirv::AddressOfOp::create(b, ff.getLoc(), gv);
            replacement = spirv::LoadOp::create(
                b, ff.getLoc(), t, addr.getResult(), spirv::MemoryAccessAttr(),
                IntegerAttr());
          } else if (info.kind == InputVar) {
            auto &gv = inputGlobals[info.binding];
            auto addr = spirv::AddressOfOp::create(b, ff.getLoc(), gv);
            replacement = spirv::LoadOp::create(
                b, ff.getLoc(), t, addr.getResult(), spirv::MemoryAccessAttr(),
                IntegerAttr());
          } else {
            if (!pcGlobal) {
              replacement = b.create<spirv::UndefOp>(ff.getLoc(), t);
            } else {
              auto addr = spirv::AddressOfOp::create(b, ff.getLoc(), pcGlobal);
              auto load = spirv::LoadOp::create(
                  b, ff.getLoc(), pcStructType, addr.getResult(),
                  spirv::MemoryAccessAttr(), IntegerAttr());
              auto it = std::find(pushTypes.begin(), pushTypes.end(), t);
              int idx = static_cast<int>(it - pushTypes.begin());
              if (pushTypes.size() == 1)
                replacement = load;
              else
                replacement = b.create<spirv::CompositeExtractOp>(
                    ff.getLoc(), t, load, b.getI32ArrayAttr({idx}));
            }
          }
          arg.replaceAllUsesWith(replacement);
          spvFunc.getBody().front().eraseArgument(0);
        }
      }
      spvFunc.setType(FunctionType::get(ctx, {}, outTy));

      // func.return -> spirv return + store
      for (auto &block : spvFunc.getBody()) {
        if (auto ret = block.getTerminator()) {
          OpBuilder rb(ret);
          if (outGv && ret->getNumOperands() > 0) {
            auto addr = spirv::AddressOfOp::create(rb, ff.getLoc(), outGv);
            Value storeVal;
            if (ret->getNumOperands() == 1) {
              storeVal = ret->getOperand(0);
            } else {
              storeVal = rb.create<spirv::CompositeConstructOp>(
                  ret->getLoc(), outTy, ret->getOperands());
            }
            rb.create<spirv::StoreOp>(ff.getLoc(), addr.getResult(), storeVal,
                                      spirv::MemoryAccessAttr(), IntegerAttr());
          }
          rb.create<spirv::ReturnOp>(ret->getLoc());
          ret->erase();
          break;
        }
      }
      spvFunc.setType(FunctionType::get(ctx, {}, {}));
      ff.erase();
    }

    // 9. Create entry points with interface variables
    SmallVector<spirv::FuncOp> wrapped;
    for (auto &op : *spvBody)
      if (auto sf = dyn_cast<spirv::FuncOp>(op)) wrapped.push_back(sf);

    SmallVector<Attribute> interface;
    for (auto &gv : inputGlobals)
      interface.push_back(SymbolRefAttr::get(gv.getSymNameAttr()));
    for (auto &gv : outputGlobals)
      interface.push_back(SymbolRefAttr::get(gv.getSymNameAttr()));

    for (auto sf : wrapped) {
      builder.setInsertionPointToEnd(spvBody);
      builder.create<spirv::EntryPointOp>(
          sf.getLoc(), spirv::ExecutionModel::Fragment, sf, interface);
      builder.create<spirv::ExecutionModeOp>(
          sf.getLoc(), sf, spirv::ExecutionMode::OriginUpperLeft,
          ArrayRef<int32_t>{});
    }
  }
};

} // namespace

void mlir::shader::registerShaderWrapSPIRVModulePass() {
  PassRegistration<WrapSPIRVModulePass>();
}
