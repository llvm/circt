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
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMTypes.h"

#include "circt/Tools/arcilator/ArcRuntime/Common.h"
#include "circt/Tools/arcilator/ArcRuntime/JITBind.h"

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
struct InsertRuntimePass
    : public arc::impl::InsertRuntimeBase<InsertRuntimePass> {
  using InsertRuntimeBase::InsertRuntimeBase;

  void runOnOperation() override;
};
} // namespace

struct RuntimeFuncOps {
  LLVM::LLVMFuncOp allocInstance;
  LLVM::LLVMFuncOp deleteInstance;
  LLVM::LLVMFuncOp onEval;
};

static void buildInstanceLifecycleFnDecls(OpBuilder &builder, Location loc,
                                          RuntimeFuncOps &funcOps) {
  auto ptrTy = LLVM::LLVMPointerType::get(builder.getContext());
  auto voidTy = LLVM::LLVMVoidType::get(builder.getContext());
  // AllocInstance
  funcOps.allocInstance = LLVM::LLVMFuncOp::create(
      builder, loc, runtime::APICallbacks::symNameAllocInstance,
      LLVM::LLVMFunctionType::get(ptrTy, {ptrTy, ptrTy}));
  // DeleteInstance
  funcOps.deleteInstance = LLVM::LLVMFuncOp::create(
      builder, loc, runtime::APICallbacks::symNameDeleteInstance,
      LLVM::LLVMFunctionType::get(voidTy, {ptrTy}));
  // OnEval
  funcOps.onEval = LLVM::LLVMFuncOp::create(
      builder, loc, runtime::APICallbacks::symNameOnEval,
      LLVM::LLVMFunctionType::get(voidTy, {ptrTy}));
}

void InsertRuntimePass::runOnOperation() {
  auto &modelInfo = getAnalysis<ModelInfoAnalysis>();

  OpBuilder builder(getOperation());
  builder.setInsertionPointToStart(getOperation().getBody());
  SmallDenseMap<StringAttr, RuntimeModelOp> arcModelSymMap;
  for (auto &[modelOp, modelInfo] : modelInfo.infoMap) {
    auto symName =
        builder.getStringAttr(Twine("arcRuntimeModel_") + modelInfo.name);
    auto rtModelOp =
        RuntimeModelOp::create(builder, modelOp.getLoc(), symName,
                               builder.getStringAttr(modelInfo.name),
                               static_cast<uint64_t>(modelInfo.numStateBytes));
    arcModelSymMap.insert({modelOp.getSymNameAttr(), rtModelOp});
  }

  SmallVector<SimInstantiateOp> instantiateOps;
  getOperation().getBody()->walk([&](Operation *op) -> WalkResult {
    if (auto instOp = dyn_cast<SimInstantiateOp>(op)) {
      // Don't touch instances which somehow already carry a runtime model
      if (!instOp.getRuntimeModel())
        instantiateOps.push_back(instOp);
      return WalkResult::skip();
    }
    if (auto instOp = dyn_cast<ModelOp>(op))
      return WalkResult::skip();
    return WalkResult::advance();
  });

  RuntimeFuncOps funcOps;
  if (!instantiateOps.empty())
    buildInstanceLifecycleFnDecls(builder, getOperation().getLoc(), funcOps);

  for (auto instantiateOp : instantiateOps) {
    // Point the instance to the respective RuntimeModel
    auto instanceModelSym =
        llvm::cast<SimModelInstanceType>(
            instantiateOp.getBody().getArgument(0).getType())
            .getModel()
            .getAttr();
    auto rtModelOp = arcModelSymMap.find(instanceModelSym);
    if (rtModelOp == arcModelSymMap.end()) {
      instantiateOp->emitOpError(" does not refer to a known Arc model.");
      signalPassFailure();
      continue;
    }
    instantiateOp.setRuntimeModelAttr(
        FlatSymbolRefAttr::get(rtModelOp->getSecond().getSymNameAttr()));

    // Add extra arguments
    if (!extraArgs.empty()) {
      StringAttr newArgs;
      if (!instantiateOp.getRuntimeArgsAttr() ||
          instantiateOp.getRuntimeArgsAttr().getValue().empty())
        newArgs = StringAttr::get(&getContext(), Twine(extraArgs));
      else
        newArgs = StringAttr::get(&getContext(),
                                  Twine(instantiateOp.getRuntimeArgsAttr()) +
                                      Twine(";") + Twine(extraArgs));
      instantiateOp.setRuntimeArgsAttr(newArgs);
    }
    // Insert onEval call
    OpBuilder instBodyBuilder(instantiateOp);
    instBodyBuilder.setInsertionPointToStart(
        &instantiateOp.getBody().getBlocks().front());
    auto runtimeInst =
        UnrealizedConversionCastOp::create(
            instBodyBuilder, instantiateOp.getLoc(),
            LLVM::LLVMPointerType::get(instBodyBuilder.getContext()),
            instantiateOp.getBody().getArgument(0))
            .getResult(0);

    instantiateOp.getBody().getBlocks().front().walk([&](SimStepOp stepOp) {
      instBodyBuilder.setInsertionPoint(stepOp);
      LLVM::CallOp::create(instBodyBuilder, stepOp.getLoc(), funcOps.onEval,
                           {runtimeInst});
    });
  }
  markAnalysesPreserved<ModelInfoAnalysis>();
}
