//===- InsertRuntime.cpp --------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "circt/Dialect/Arc/ArcOps.h"
#include "circt/Dialect/Arc/ArcPasses.h"
#include "circt/Dialect/Arc/ModelInfo.h"
#include "circt/Dialect/Arc/Runtime/Common.h"
#include "circt/Dialect/Arc/Runtime/JITBind.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMTypes.h"

#include "mlir/Pass/Pass.h"

#define DEBUG_TYPE "arc-insert-runtime"

namespace circt {
namespace arc {
#define GEN_PASS_DEF_INSERTRUNTIME
#include "circt/Dialect/Arc/ArcPasses.h.inc"
} // namespace arc
} // namespace circt

using namespace mlir;
using namespace circt;
using namespace arc;

namespace {

// API Helpers
struct RuntimeFunction {
  LLVM::LLVMFuncOp llvmFuncOp = {};
  bool used = false;
};

struct AllocInstanceFunction : public RuntimeFunction {
  explicit AllocInstanceFunction(ImplicitLocOpBuilder &builder) {
    /*
     uint8_t *
     arcRuntimeIR_allocInstance(const ArcRuntimeModelInfo *model, const char
     *args);
    */
    auto ptrTy = LLVM::LLVMPointerType::get(builder.getContext());
    llvmFuncOp = LLVM::LLVMFuncOp::create(
        builder, runtime::APICallbacks::symNameAllocInstance,
        LLVM::LLVMFunctionType::get(ptrTy, {ptrTy, ptrTy}));
  }
};

struct DeleteInstanceFunction : public RuntimeFunction {
  explicit DeleteInstanceFunction(ImplicitLocOpBuilder &builder) {
    /*
     void arcRuntimeIR_deleteInstance(uint8_t *modelState);
    */
    auto ptrTy = LLVM::LLVMPointerType::get(builder.getContext());
    auto voidTy = LLVM::LLVMVoidType::get(builder.getContext());
    llvmFuncOp = LLVM::LLVMFuncOp::create(
        builder, runtime::APICallbacks::symNameDeleteInstance,
        LLVM::LLVMFunctionType::get(voidTy, {ptrTy}));
  }
};

struct OnEvalFunction : public RuntimeFunction {
  explicit OnEvalFunction(ImplicitLocOpBuilder &builder) {
    /*
     void arcRuntimeIR_onEval(uint8_t *modelState);
    */
    auto ptrTy = LLVM::LLVMPointerType::get(builder.getContext());
    auto voidTy = LLVM::LLVMVoidType::get(builder.getContext());
    llvmFuncOp =
        LLVM::LLVMFuncOp::create(builder, runtime::APICallbacks::symNameOnEval,
                                 LLVM::LLVMFunctionType::get(voidTy, {ptrTy}));
  }
};

struct OnInitializedFunction : public RuntimeFunction {
  explicit OnInitializedFunction(ImplicitLocOpBuilder &builder) {
    /*
     void arcRuntimeIR_onInitialized(uint8_t *modelState);
    */
    auto ptrTy = LLVM::LLVMPointerType::get(builder.getContext());
    auto voidTy = LLVM::LLVMVoidType::get(builder.getContext());
    llvmFuncOp = LLVM::LLVMFuncOp::create(
        builder, runtime::APICallbacks::symNameOnInitialized,
        LLVM::LLVMFunctionType::get(voidTy, {ptrTy}));
  }
};

// Lowering Helpers

struct RuntimeModelContext; // Forward declaration

struct GlobalRuntimeContext {
  GlobalRuntimeContext() = delete;

  /// Constructs a global context and adds the available runtime API function
  /// declarations to the MLIR module
  explicit GlobalRuntimeContext(ModuleOp moduleOp)
      : mlirModuleOp(moduleOp), globalBuilder(createBuilder(moduleOp)),
        allocInstanceFn(globalBuilder), deleteInstanceFn(globalBuilder),
        onEvalFn(globalBuilder), onInitializedFn(globalBuilder) {}

  /// Delete all API functions that are never called
  void deleteUnusedFunctions() {
    for (auto *fn : apiFunctions)
      if (!fn->used)
        fn->llvmFuncOp->erase();
  }

  /// Add an Arc model to the global runtime context
  void addModel(ModelOp &modelOp, const ModelInfo &modelInfo);
  /// Build a RuntimeModelOp for each registered model
  LogicalResult buildRuntimeModelOps();
  /// Find and assign instances of the registered models within the root module
  LogicalResult collectInstances();

  /// The root module
  ModuleOp mlirModuleOp;
  /// Builder for global operations
  ImplicitLocOpBuilder globalBuilder;

  // API Functions
  AllocInstanceFunction allocInstanceFn;
  DeleteInstanceFunction deleteInstanceFn;
  OnEvalFunction onEvalFn;
  OnInitializedFunction onInitializedFn;
  const std::array<RuntimeFunction *, 4> apiFunctions = {
      &allocInstanceFn, &deleteInstanceFn, &onEvalFn, &onInitializedFn};

  // Maps model symbol name to model context
  SmallDenseMap<StringAttr, std::unique_ptr<RuntimeModelContext>> models;

private:
  static ImplicitLocOpBuilder createBuilder(ModuleOp &moduleOp) {
    auto builder = ImplicitLocOpBuilder(moduleOp.getLoc(), moduleOp);
    builder.setInsertionPointToStart(moduleOp.getBody());
    return builder;
  }
};

struct RuntimeModelContext {
  RuntimeModelContext() = delete;
  /// Construct the local context for an Arc model within the global context
  RuntimeModelContext(GlobalRuntimeContext &globalContext, ModelOp &modelOp,
                      const ModelInfo &modelInfo)
      : globalContext(globalContext), modelOp(modelOp), modelInfo(modelInfo) {}

  /// Register an MLIR defined instance of our model
  void addInstance(SimInstantiateOp &instantiateOp) {
    assert(!instantiateOp.getRuntimeModelAttr());
    assert(!!runtimeModelOp);
    instantiateOp.setRuntimeModelAttr(
        FlatSymbolRefAttr::get(runtimeModelOp.getSymNameAttr()));
    instances.push_back(instantiateOp);
  }

  /// Insert runtime calls to the model and its instances
  LogicalResult lower();

  /// The global runtime context
  GlobalRuntimeContext &globalContext;
  /// This context's model
  ModelOp modelOp;
  /// Model metadata
  const ModelInfo &modelInfo;
  /// List of registered instances
  SmallVector<SimInstantiateOp> instances;
  /// The model's corresponding RuntimeModelOp
  RuntimeModelOp runtimeModelOp;

private:
  LogicalResult lowerInstance(SimInstantiateOp &instance);
};
struct InsertRuntimePass
    : public arc::impl::InsertRuntimeBase<InsertRuntimePass> {
  using InsertRuntimeBase::InsertRuntimeBase;

  void runOnOperation() override;
};

} // namespace

void GlobalRuntimeContext::addModel(ModelOp &modelOp,
                                    const ModelInfo &modelInfo) {
  auto newModel =
      std::make_unique<RuntimeModelContext>(*this, modelOp, modelInfo);
  models[modelOp.getNameAttr()] = std::move(newModel);
}

// Find all instances in the MLIR Module and assign them to their
// respective Arc Model
LogicalResult GlobalRuntimeContext::collectInstances() {
  bool hasFailed = false;
  mlirModuleOp.getBody()->walk([&](Operation *op) -> WalkResult {
    if (auto instOp = dyn_cast<SimInstantiateOp>(op)) {
      // Don't touch instances which somehow already carry a runtime model
      if (instOp.getRuntimeModel())
        return WalkResult::skip();
      auto instanceModelSym = llvm::cast<SimModelInstanceType>(
                                  instOp.getBody().getArgument(0).getType())
                                  .getModel()
                                  .getAttr();
      auto modelContext = models.find(instanceModelSym);
      if (modelContext == models.end()) {
        hasFailed = true;
        instOp->emitOpError(" does not refer to a known Arc model.");
      } else {
        modelContext->second->addInstance(instOp);
      }
      return WalkResult::skip();
    }
    if (auto instOp = dyn_cast<ModelOp>(op))
      return WalkResult::skip();
    return WalkResult::advance();
  });
  return success(!hasFailed);
}

// Build the global RuntimeModelOp for each model
LogicalResult GlobalRuntimeContext::buildRuntimeModelOps() {
  auto savedLoc = globalBuilder.getLoc();
  for (auto &[_, model] : models) {
    globalBuilder.setLoc(model->modelOp.getLoc());
    auto symName = globalBuilder.getStringAttr(Twine("arcRuntimeModel_") +
                                               model->modelInfo.name);
    model->runtimeModelOp = RuntimeModelOp::create(
        globalBuilder, symName,
        globalBuilder.getStringAttr(model->modelInfo.name),
        static_cast<uint64_t>(model->modelInfo.numStateBytes));
  }
  globalBuilder.setLoc(savedLoc);
  return success();
}

// Lower the model and all of its instances
LogicalResult RuntimeModelContext::lower() {
  bool hasFailed = false;
  for (auto &instance : instances)
    if (failed(lowerInstance(instance)))
      hasFailed = true;
  return success(!hasFailed);
}

LogicalResult RuntimeModelContext::lowerInstance(SimInstantiateOp &instance) {
  // For now, these get invoked by the lowering of SimInstantiateOp
  globalContext.allocInstanceFn.used = true;
  globalContext.onInitializedFn.used = true;
  globalContext.deleteInstanceFn.used = true;

  // Insert onEval call for every step call
  OpBuilder instBodyBuilder(instance);
  instBodyBuilder.setInsertionPointToStart(
      &instance.getBody().getBlocks().front());
  auto runtimeInst =
      UnrealizedConversionCastOp::create(
          instBodyBuilder, instance.getLoc(),
          LLVM::LLVMPointerType::get(instBodyBuilder.getContext()),
          instance.getBody().getArgument(0))
          .getResult(0);

  instance.getBody().getBlocks().front().walk([&](SimStepOp stepOp) {
    instBodyBuilder.setInsertionPoint(stepOp);
    globalContext.onEvalFn.used = true;
    LLVM::CallOp::create(instBodyBuilder, stepOp.getLoc(),
                         globalContext.onEvalFn.llvmFuncOp, {runtimeInst});
  });

  return success();
}

void InsertRuntimePass::runOnOperation() {
  // Construct the global context and collect information on all
  // models and instances
  auto &modelInfo = getAnalysis<ModelInfoAnalysis>();
  auto globalContext = std::make_unique<GlobalRuntimeContext>(getOperation());
  for (auto &[mOp, mInfo] : modelInfo.infoMap)
    globalContext->addModel(mOp, mInfo);
  if (failed(globalContext->buildRuntimeModelOps()) ||
      failed(globalContext->collectInstances())) {
    signalPassFailure();
    return;
  }

  // Lower all models
  for (auto &[_, model] : globalContext->models) {
    // If provided, append extra instance arguments
    if (!extraArgs.empty()) {
      for (auto &instance : model->instances) {
        StringAttr newArgs;
        if (!instance.getRuntimeArgsAttr() ||
            instance.getRuntimeArgsAttr().getValue().empty())
          newArgs = StringAttr::get(&getContext(), Twine(extraArgs));
        else
          newArgs = StringAttr::get(&getContext(),
                                    Twine(instance.getRuntimeArgsAttr()) +
                                        Twine(";") + Twine(extraArgs));
        instance.setRuntimeArgsAttr(newArgs);
      }
    }

    if (failed(model->lower()))
      signalPassFailure();
  }

  globalContext->deleteUnusedFunctions();
  markAnalysesPreserved<ModelInfoAnalysis>();
}
