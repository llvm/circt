//===- LowerTaps.cpp - Implement LowerTaps Pass ---------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "PassDetails.h"
#include "circt/Dialect/Arc/ArcDialect.h"
#include "circt/Dialect/Arc/ArcOps.h"
#include "circt/Dialect/SystemC/SystemCDialect.h"
#include "circt/Dialect/SystemC/SystemCOps.h"
#include "circt/Support/LLVM.h"
#include "circt/Support/SymCache.h"
#include "mlir/Dialect/EmitC/IR/EmitC.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/DialectConversion.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/ToolOutputFile.h"
#include <algorithm>

#define DEBUG_TYPE "arc-lower-taps"

using namespace circt;
using namespace arc;

//===----------------------------------------------------------------------===//
// Helpers
//===----------------------------------------------------------------------===//

static std::string getLegalizedAndUniquedCName(StringRef name) {
  std::string newName = name.str();
  // TODO
  std::replace(newName.begin(), newName.end(), '/', '_');
  return newName;
}

//===----------------------------------------------------------------------===//
// Op lowerings
//===----------------------------------------------------------------------===//

static void lowerStateTap(StateTapOp op, OpBuilder &apiBuilder) {
  auto *ctxt = apiBuilder.getContext();
  bool genGetter =
      op.getMode() == TapMode::Read || op.getMode() == TapMode::ReadWrite;
  bool genSetter =
      op.getMode() == TapMode::Write || op.getMode() == TapMode::ReadWrite;

  auto voidStarTy =
      mlir::emitc::PointerType::get(mlir::emitc::OpaqueType::get(ctxt, "void"));
  auto stateTy = op.getState().getType();
  bool isMemoryTap = op.getKind() == TapKind::Memory;
  auto stateBaseTy = !isMemoryTap ? cast<StateType>(stateTy).getType()
                                  : cast<MemoryType>(stateTy).getWordType();
  auto funcName = getLegalizedAndUniquedCName(op.getTapName());

  unsigned offset = 0;
  if (auto storageGet = op.getState().getDefiningOp<StorageGetOp>())
    offset = storageGet.getOffset();

  unsigned stride = 0;
  if (isMemoryTap) {
    stride = (cast<MemoryType>(stateTy).getWordType().getWidth() + 7) / 8;
    stride =
        llvm::alignToPowerOf2(stride, llvm::bit_ceil(std::min(stride, 8U)));
  }

  auto buildStateAccess = [&](ImplicitLocOpBuilder &builder) -> Value {
    std::string expr;
    if (isMemoryTap)
      expr = "((uint8_t*) state + " + std::to_string(offset) + " + " +
             std::to_string(stride) + " * idx)";
    else
      expr = "((uint8_t*) state + " + std::to_string(offset) + ")";
    Value indexExp = builder.create<mlir::emitc::ConstantOp>(
        voidStarTy, mlir::emitc::OpaqueAttr::get(ctxt, expr));
    Value castExp = builder.create<mlir::emitc::CastOp>(
        mlir::emitc::PointerType::get(stateBaseTy), indexExp);
    Value applyExp =
        builder.create<mlir::emitc::ApplyOp>(stateBaseTy, "*", castExp);
    return applyExp;
  };

  auto buildFunc =
      [&](OpBuilder &builder, Location loc, StringRef funcName,
          ArrayRef<StringRef> argNames, ArrayRef<Type> inputTypes,
          ArrayRef<Type> outputTypes,
          const std::function<void(ImplicitLocOpBuilder & bodyBuilder)>
              &buildBody) {
        SmallVector<StringRef> argNames2(argNames);
        SmallVector<Type> inputTypes2(inputTypes);
        if (isMemoryTap) {
          argNames2.push_back("idx");
          inputTypes2.push_back(builder.getI32Type());
        }
        FunctionType type = builder.getFunctionType(inputTypes2, outputTypes);
        auto getterFuncOp =
            builder.create<systemc::FuncOp>(loc, funcName, argNames2, type);
        auto &block = getterFuncOp.getBody().front();
        ImplicitLocOpBuilder funcBodyBuilder(loc, &block, block.begin());
        buildBody(funcBodyBuilder);
      };

  if (genGetter) {
    buildFunc(apiBuilder, op.getLoc(), "get_" + funcName, StringRef("state"),
              voidStarTy, stateBaseTy, [&](ImplicitLocOpBuilder &builder) {
                Value applyExp = buildStateAccess(builder);
                builder.create<systemc::ReturnOp>(applyExp);
              });
  }

  if (genSetter) {
    buildFunc(apiBuilder, op.getLoc(), "set_" + funcName, {"state", "value"},
              {voidStarTy, stateBaseTy}, {},
              [&](ImplicitLocOpBuilder &builder) {
                Value applyExp = buildStateAccess(builder);
                builder.create<systemc::AssignOp>(
                    applyExp, builder.getBlock()->getArgument(1));
                builder.create<systemc::ReturnOp>();
              });
  }

  op.erase();
}

static void lowerArcModel(ModelOp op, OpBuilder &builder) {
  auto *ctxt = builder.getContext();
  auto voidStarTy =
      mlir::emitc::PointerType::get(mlir::emitc::OpaqueType::get(ctxt, "void"));
  auto initFuncOp = builder.create<systemc::FuncOp>(
      op.getLoc(), "alloc_and_init_" + op.getName().str(),
      SmallVector<StringRef>{}, builder.getFunctionType({}, voidStarTy));
  auto &block = initFuncOp.getBody().front();
  ImplicitLocOpBuilder funcBodyBuilder(op.getLoc(), &block, block.begin());
  Value stateExp =
      funcBodyBuilder
          .create<mlir::emitc::CallOp>(
              op.getLoc(), voidStarTy, "calloc",
              builder.getArrayAttr(
                  {mlir::emitc::OpaqueAttr::get(ctxt, "1"),
                   mlir::emitc::OpaqueAttr::get(
                       ctxt, std::to_string(op.getBodyBlock()
                                                .getArgumentTypes()
                                                .front()
                                                .cast<StorageType>()
                                                .getSize()))}),
              nullptr, ValueRange{})
          ->getResult(0);
  funcBodyBuilder.create<systemc::ReturnOp>(op.getLoc(), stateExp);
}

//===----------------------------------------------------------------------===//
// LowerTaps pass
//===----------------------------------------------------------------------===//

namespace {
struct LowerTapsPass : LowerTapsBase<LowerTapsPass> {
  void runOnOperation() override;
  void printToAPIHeader(ModuleOp apiModule);

  using LowerTapsBase::apiFile;
};
} // namespace

void LowerTapsPass::runOnOperation() {
  OpBuilder apiBuilder = OpBuilder::atBlockBegin(getOperation().getBody());
  ModuleOp apiModule = apiBuilder.create<ModuleOp>(getOperation().getLoc());
  apiBuilder.setInsertionPointToStart(apiModule.getBody());

  apiBuilder.create<mlir::emitc::IncludeOp>(apiModule.getLoc(), "stdlib.h",
                                            true);
  apiBuilder.create<mlir::emitc::IncludeOp>(apiModule.getLoc(), "stdint.h",
                                            true);

  getOperation().walk([&](Operation *op) {
    if (auto tapOp = dyn_cast<StateTapOp>(op))
      lowerStateTap(tapOp, apiBuilder);
    if (auto modelOp = dyn_cast<ModelOp>(op))
      lowerArcModel(modelOp, apiBuilder);
  });

  printToAPIHeader(apiModule);
  apiModule->erase();
}

void LowerTapsPass::printToAPIHeader(ModuleOp apiModule) {
  // Print to the output file if one was given, or stdout otherwise.
  if (apiFile.empty()) {
    apiModule->print(llvm::outs());
    llvm::outs() << "\n";
  } else {
    std::error_code ec;
    llvm::ToolOutputFile outputFile(apiFile, ec,
                                    llvm::sys::fs::OpenFlags::OF_None);
    if (ec) {
      mlir::emitError(apiModule.getLoc(), "unable to open API file: ")
          << ec.message();
      return signalPassFailure();
    }
    apiModule->print(outputFile.os());
    outputFile.os() << "\n";
    outputFile.keep();
  }
}

std::unique_ptr<Pass> arc::createLowerTapsPass(llvm::StringRef apiFile) {
  auto pass = std::make_unique<LowerTapsPass>();
  if (!apiFile.empty())
    pass->apiFile.assign(apiFile);
  return pass;
}
