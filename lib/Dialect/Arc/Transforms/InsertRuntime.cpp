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
#include "circt/Dialect/Arc/Runtime/TraceTaps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMTypes.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
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

protected:
  // Add attributes for passing the model state pointer to the runtime library
  void setModelStateArgAttrs(OpBuilder &builder, unsigned argIndex,
                             bool isMutable) {
    llvmFuncOp.setArgAttr(0, LLVM::LLVMDialect::getNoCaptureAttrName(),
                          builder.getUnitAttr());
    llvmFuncOp.setArgAttr(0, LLVM::LLVMDialect::getNoFreeAttrName(),
                          builder.getUnitAttr());
    llvmFuncOp.setArgAttr(0, LLVM::LLVMDialect::getNoAliasAttrName(),
                          builder.getUnitAttr());
    if (!isMutable)
      llvmFuncOp.setArgAttr(0, LLVM::LLVMDialect::getReadonlyAttrName(),
                            builder.getUnitAttr());
  }
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
    llvmFuncOp.setResultAttr(0, LLVM::LLVMDialect::getNoAliasAttrName(),
                             builder.getUnitAttr());
    llvmFuncOp.setResultAttr(0, LLVM::LLVMDialect::getNoUndefAttrName(),
                             builder.getUnitAttr());
    llvmFuncOp.setResultAttr(0, LLVM::LLVMDialect::getNonNullAttrName(),
                             builder.getUnitAttr());
    llvmFuncOp.setResultAttr(0, LLVM::LLVMDialect::getAlignAttrName(),
                             builder.getI64IntegerAttr(16));
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
    setModelStateArgAttrs(builder, 0, true);
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
    setModelStateArgAttrs(builder, 0, true);
  }
};

struct SwapTraceBufferFunction : public RuntimeFunction {
  explicit SwapTraceBufferFunction(ImplicitLocOpBuilder &builder) {
    /*
     uint64_t *arcRuntimeIR_swapTraceBuffer(const uint8_t *modelState);
    */
    auto ptrTy = LLVM::LLVMPointerType::get(builder.getContext());
    llvmFuncOp = LLVM::LLVMFuncOp::create(
        builder, runtime::APICallbacks::symNameSwapTraceBuffer,
        LLVM::LLVMFunctionType::get(ptrTy, {ptrTy}));
    llvmFuncOp.setResultAttr(0, LLVM::LLVMDialect::getNoAliasAttrName(),
                             builder.getUnitAttr());
    llvmFuncOp.setResultAttr(0, LLVM::LLVMDialect::getNoUndefAttrName(),
                             builder.getUnitAttr());
    llvmFuncOp.setResultAttr(0, LLVM::LLVMDialect::getNonNullAttrName(),
                             builder.getUnitAttr());
    llvmFuncOp.setResultAttr(0, LLVM::LLVMDialect::getAlignAttrName(),
                             builder.getI64IntegerAttr(8));
    setModelStateArgAttrs(builder, 0, false);
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
        onEvalFn(globalBuilder), onInitializedFn(globalBuilder),
        swapTraceBufferFn(globalBuilder) {}

  /// Delete all API functions that are never called
  void deleteUnusedFunctions() {
    for (auto *fn : apiFunctions)
      if (!fn->used)
        fn->llvmFuncOp->erase();
  }

  static Type getTraceExtendedType(Type stateType) {
    auto numBits = stateType.getIntOrFloatBitWidth();
    auto numQWords = std::max((numBits + 63) / 64, 1U);
    return IntegerType::get(stateType.getContext(), numQWords * 64);
  }

  /// Add an Arc model to the global runtime context
  void addModel(ModelOp &modelOp, const ModelInfo &modelInfo);
  /// Build a RuntimeModelOp for each registered model
  LogicalResult buildRuntimeModelOps();
  /// Find and assign instances of the registered models within the root module
  LogicalResult collectInstances();
  LogicalResult buildTraceInstrumentation();

  LLVM::LLVMFuncOp getTraceInstrumentFn(Type ty) const {
    assert(ty.getIntOrFloatBitWidth() % 64 == 0);
    auto fn = traceInstrumentationFns.find(ty);
    assert(fn != traceInstrumentationFns.end());
    return fn->second;
  }

  /// The root module
  ModuleOp mlirModuleOp;
  /// Builder for global operations
  ImplicitLocOpBuilder globalBuilder;

  // API Functions
  AllocInstanceFunction allocInstanceFn;
  DeleteInstanceFunction deleteInstanceFn;
  OnEvalFunction onEvalFn;
  OnInitializedFunction onInitializedFn;
  SwapTraceBufferFunction swapTraceBufferFn;
  const std::array<RuntimeFunction *, 5> apiFunctions = {
      &allocInstanceFn, &deleteInstanceFn, &onEvalFn, &onInitializedFn,
      &swapTraceBufferFn};

  // Maps model symbol name to model context
  SmallDenseMap<StringAttr, std::unique_ptr<RuntimeModelContext>> models;

private:
  static ImplicitLocOpBuilder createBuilder(ModuleOp &moduleOp) {
    auto builder = ImplicitLocOpBuilder(moduleOp.getLoc(), moduleOp);
    builder.setInsertionPointToStart(moduleOp.getBody());
    return builder;
  }
  void buildTraceInstrumentationFn(Type ty);

  SmallDenseMap<Type, LLVM::LLVMFuncOp> traceInstrumentationFns;
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

  void addTappedStateWrite(StateWriteOp &writeOp) {
    assert(writeOp.getTraceTapModel().has_value() &&
           writeOp.getTraceTapIndex().has_value());
    assert(modelOp.getSymNameAttr() ==
           writeOp.getTraceTapModelAttr().getAttr());
    tappedWrites.push_back(writeOp);
  }

  bool hasTraceTaps() { return runtimeModelOp.getTraceTaps().has_value(); }

  LogicalResult insertTraceInstrumentation();

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
  SmallVector<StateWriteOp> tappedWrites;

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

LogicalResult GlobalRuntimeContext::buildTraceInstrumentation() {
  if (llvm::none_of(
          models, [](auto &modelIt) { return modelIt.second->hasTraceTaps(); }))
    return success();

  swapTraceBufferFn.used = true;
  SetVector<Type> tappedTypes;

  mlirModuleOp.getBody()->walk([&](StateWriteOp writeOp) {
    if (!writeOp.getTraceTapModel().has_value())
      return;
    auto modelCtxt = models.find(writeOp.getTraceTapModelAttr().getAttr());
    assert(modelCtxt != models.end() && "Unknown referenced model");
    modelCtxt->second->addTappedStateWrite(writeOp);
    if (isa<IntegerType>(writeOp.getValue().getType()))
      buildTraceInstrumentationFn(writeOp.getValue().getType());
    else
      writeOp->emitWarning("Tracing of non-integer type is not supported");
  });

  return success();
}

// Build a trace instrumentation function recording the change of a state
// value to the trace buffer. Calls the runtime library if the current buffer
// is running out of space.
// Pseudocode of the constructed function:
//
//
// void _arc_trace_instrument_i{BW}(uint8_t *simState, uint64_t traceTapId,
//                                   uint{BW}_t newValue) {
//   // BB: "capcaityCheckBlock"
//   const uint32_t reqSize = {BW} / 64 + 1;
//   ArcState *runtimeState = (ArcState*)(simState - sizeof(ArcState));
//   uint64_t *oldBuffer = runtimeState->traceBuffer;
//   const uint32_t oldSize = runtimeState->traceBufferSize;
//   uint32_t newSize = oldSize + reqSize;
//   uint64_t *storePtr = &oldBuffer[oldSize];
//   if (newSize >= runtime::defaultTraceBufferCapacity) [[unlikely]] {
//     // BB: "swapBufferBlock"
//     storePtr = arcRuntimeIR_swapBuffer(simState);
//     runtimeState->traceBuffer = storePtr;
//     newSize = reqSize;
//   }
//   // BB: "bufferStoreBlock"
//   storePtr[0] = traceTapId;
//   for (unsigned qword = 0; qword < {BW} / 64; ++qword) // Unrolled
//     storePtr[qword + 1] = (uint64_t)(newValue >> (64 * qword));
//   runtimeState->traceBufferSize = newSize;
// }
//

void GlobalRuntimeContext::buildTraceInstrumentationFn(Type ty) {
  assert(isa<IntegerType>(ty));
  // Check if we've already built the function
  auto traceTy = getTraceExtendedType(ty);
  if (traceInstrumentationFns.contains(traceTy))
    return;

  // Build the function signature
  auto typeQWords = traceTy.getIntOrFloatBitWidth() / 64;
  assert(traceTy.getIntOrFloatBitWidth() % 64 == 0);
  auto *ctx = ty.getContext();
  auto i64Ty = IntegerType::get(ctx, 64);
  auto i32Ty = IntegerType::get(ctx, 32);
  auto llvmPtrTy = LLVM::LLVMPointerType::get(ctx);
  auto llvmFnTy = LLVM::LLVMFunctionType::get(LLVM::LLVMVoidType::get(ctx),
                                              {llvmPtrTy, i64Ty, traceTy});
  auto symName = StringAttr::get(
      ctx, "_arc_trace_instrument_i" + Twine(traceTy.getIntOrFloatBitWidth()));
  auto funcOp = LLVM::LLVMFuncOp::create(globalBuilder, symName, llvmFnTy,
                                         LLVM::Linkage::Private);
  funcOp.setNoInline(true);
  traceInstrumentationFns.insert({traceTy, funcOp});

  // Build the body of the function
  OpBuilder::InsertionGuard g(globalBuilder);
  auto *capcaityCheckBlock = funcOp.addEntryBlock(globalBuilder);
  auto *swapBufferBlock = &funcOp.getRegion().emplaceBlock();
  auto *bufferStoreBlock = &funcOp.getRegion().emplaceBlock();
  // storePtr
  bufferStoreBlock->addArgument(llvmPtrTy, globalBuilder.getLoc());
  // newSize
  bufferStoreBlock->addArgument(i32Ty, globalBuilder.getLoc());

  globalBuilder.setInsertionPointToStart(capcaityCheckBlock);

  auto statePtr = capcaityCheckBlock->getArgument(0);
  auto bufferPtrPtr = LLVM::GEPOp::create(
      globalBuilder, llvmPtrTy, globalBuilder.getI8Type(), statePtr,
      {LLVM::GEPArg(static_cast<int>(offsetof(ArcState, traceBuffer)) -
                    static_cast<int>(sizeof(ArcState)))});
  auto bufferSizePtr = LLVM::GEPOp::create(
      globalBuilder, llvmPtrTy, globalBuilder.getI8Type(), statePtr,
      {LLVM::GEPArg(static_cast<int>(offsetof(ArcState, traceBufferSize)) -
                    static_cast<int>(sizeof(ArcState)))});
  // > const uint32_t reqSize = {BW} / 64 + 1;
  auto requiredSize = typeQWords + 1;
  auto reqSizeCst = LLVM::ConstantOp::create(
      globalBuilder, globalBuilder.getI32IntegerAttr(requiredSize));
  // > uint64_t *oldBuffer = runtimeState->traceBuffer;
  auto bufferPtrVal =
      LLVM::LoadOp::create(globalBuilder, llvmPtrTy, bufferPtrPtr);
  // > const uint32_t oldSize = runtimeState->traceBufferSize;
  auto bufferSizeVal =
      LLVM::LoadOp::create(globalBuilder, i32Ty, bufferSizePtr);
  auto capacityConstant = LLVM::ConstantOp::create(
      globalBuilder,
      globalBuilder.getI32IntegerAttr(runtime::defaultTraceBufferCapacity));
  // > uint32_t newSize = oldSize + reqSize;
  auto newSizeVal =
      LLVM::AddOp::create(globalBuilder, bufferSizeVal, reqSizeCst);
  // > uint64_t *storePtr = &oldBuffer[oldSize];
  auto storePtr =
      LLVM::GEPOp::create(globalBuilder, llvmPtrTy, i64Ty, bufferPtrVal,
                          {LLVM::GEPArg(bufferSizeVal)});
  // > if (newSize >= runtime::defaultTraceBufferCapacity) [[unlikely]]
  auto needsSwap = LLVM::ICmpOp::create(globalBuilder, LLVM::ICmpPredicate::ugt,
                                        newSizeVal, capacityConstant);
  LLVM::CondBrOp::create(
      globalBuilder, needsSwap, swapBufferBlock, {}, bufferStoreBlock,
      {storePtr, newSizeVal},
      /*weights*/
      std::pair<int32_t, int32_t>(0, std::numeric_limits<int32_t>::max()));

  globalBuilder.setInsertionPointToStart(swapBufferBlock);
  // > storePtr = arcRuntimeIR_swapBuffer(simState);
  auto swapCall = LLVM::CallOp::create(
      globalBuilder, swapTraceBufferFn.llvmFuncOp, {statePtr});
  // > runtimeState->traceBuffer = storePtr;
  LLVM::StoreOp::create(globalBuilder, swapCall.getResult(), bufferPtrPtr);
  LLVM::BrOp::create(globalBuilder, {swapCall.getResult(), reqSizeCst},
                     bufferStoreBlock);

  globalBuilder.setInsertionPointToStart(bufferStoreBlock);
  // > storePtr[0] = traceTapId;
  LLVM::StoreOp::create(globalBuilder, capcaityCheckBlock->getArgument(1),
                        bufferStoreBlock->getArgument(0));

  // > for (unsigned qword = 0; qword < {BW} / 64; ++qword) // Unrolled
  for (unsigned qWord = 0; qWord < typeQWords; ++qWord) {
    // > storePtr[qword + 1] = (uint64_t)(newValue >> (64 * qword));
    auto dataStorePtr = LLVM::GEPOp::create(globalBuilder, llvmPtrTy, i64Ty,
                                            bufferStoreBlock->getArgument(0),
                                            {LLVM::GEPArg(qWord + 1)});
    Value storeVal = capcaityCheckBlock->getArgument(2);
    if (qWord > 0) {
      auto shiftCst = LLVM::ConstantOp::create(
          globalBuilder,
          globalBuilder.getIntegerAttr(storeVal.getType(), qWord * 64));
      storeVal = LLVM::LShrOp::create(globalBuilder, storeVal, shiftCst);
    }
    if (storeVal.getType() != i64Ty)
      storeVal = LLVM::TruncOp::create(globalBuilder, i64Ty, storeVal);
    LLVM::StoreOp::create(globalBuilder, storeVal, dataStorePtr);
  }
  // > runtimeState->traceBufferSize = newSize;
  LLVM::StoreOp::create(globalBuilder, bufferStoreBlock->getArgument(1),
                        bufferSizePtr);
  LLVM::ReturnOp::create(globalBuilder, Value{});
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
        static_cast<uint64_t>(model->modelInfo.numStateBytes),
        model->modelOp.getTraceTapsAttr());
    model->modelOp.setTraceTapsAttr({});
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
  if (failed(insertTraceInstrumentation()))
    hasFailed = true;
  return success(!hasFailed);
}

// Insert call to the trace instrumentation function to each tapped write
LogicalResult RuntimeModelContext::insertTraceInstrumentation() {
  if (!hasTraceTaps() || tappedWrites.empty())
    return success();
  bool hasFailed = false;
  ImplicitLocOpBuilder builder(runtimeModelOp.getLoc(),
                               runtimeModelOp.getContext());
  auto ptrTy = LLVM::LLVMPointerType::get(builder.getContext());
  for (auto writeOp : tappedWrites) {
    builder.setInsertionPoint(writeOp);
    builder.setLoc(writeOp.getLoc());
    // Lookup the instrumentation function for the state's type
    auto tapId = *writeOp.getTraceTapIndex();
    assert(tapId < runtimeModelOp.getTraceTapsAttr().size());
    auto tapAttr = cast<TraceTapAttr>(runtimeModelOp.getTraceTapsAttr()[tapId]);
    auto traceTy = GlobalRuntimeContext::getTraceExtendedType(
        writeOp.getValue().getType());
    auto instrumentFn = globalContext.getTraceInstrumentFn(traceTy);
    // Strip the tap annotation
    writeOp.setTraceTapIndex(std::nullopt);
    writeOp.setTraceTapModel(std::nullopt);
    // Test if the new value differs from the old value
    auto oldRead = StateReadOp::create(builder, writeOp.getState());
    auto hasChanged = LLVM::ICmpOp::create(builder, LLVM::ICmpPredicate::ne,
                                           writeOp.getValue(), oldRead);
    scf::IfOp::create(
        builder, hasChanged, [&](OpBuilder scfBuilder, Location loc) {
          // Pull the state write itself under the condition
          scfBuilder.clone(*writeOp.getOperation());
          // Invoke the instrumentation function
          auto statePtrCast = UnrealizedConversionCastOp::create(
              scfBuilder, loc, ptrTy, writeOp.getState());
          auto baseStatePtr = LLVM::GEPOp::create(
              scfBuilder, loc, ptrTy, scfBuilder.getI8Type(),
              statePtrCast.getResult(0),
              {LLVM::GEPArg(-1 *
                            static_cast<int32_t>(tapAttr.getStateOffset()))});
          auto tapIdxCst = LLVM::ConstantOp::create(
              scfBuilder, loc, scfBuilder.getI64IntegerAttr(tapId));
          Value storeVal = writeOp.getValue();
          if (traceTy != storeVal.getType())
            storeVal = LLVM::ZExtOp::create(scfBuilder, loc, traceTy, storeVal)
                           .getResult();
          LLVM::CallOp::create(scfBuilder, loc, instrumentFn,
                               {baseStatePtr, tapIdxCst, storeVal});
          scf::YieldOp::create(builder, loc);
        });
    writeOp.erase();
  }
  tappedWrites.clear();
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
      failed(globalContext->buildTraceInstrumentation()) ||
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
