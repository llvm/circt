//===- LowerMemoryInitializers.cpp ----------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "circt/Dialect/Arc/ArcOps.h"
#include "circt/Dialect/Arc/ArcPasses.h"
#include "circt/Dialect/HW/HWOps.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Pass/Pass.h"

#include "llvm/ADT/TypeSwitch.h"

#define DEBUG_TYPE "arc-lower-memory-initializers"

namespace circt {
namespace arc {
#define GEN_PASS_DEF_LOWERMEMORYINITIALIZERS
#include "circt/Dialect/Arc/ArcPasses.h.inc"
} // namespace arc
} // namespace circt

using namespace mlir;
using namespace circt;
using namespace arc;

namespace {
struct LowerMemoryInitializersPass
    : public arc::impl::LowerMemoryInitializersBase<
          LowerMemoryInitializersPass> {
  void runOnOperation() override;
  LogicalResult processInitializerFunction(func::FuncOp funcOp);
  LogicalResult lowerFilledInitialization(InitializeMemoryOp initOp,
                                          InitMemoryFilledOp fillOp);

  SymbolTable *symbolTable;
};
} // namespace

LogicalResult LowerMemoryInitializersPass::lowerFilledInitialization(
    InitializeMemoryOp initOp, InitMemoryFilledOp fillOp) {
  auto loc =
      FusedLoc::get(initOp.getContext(),
                    std::array<Location, 2>{initOp.getLoc(), fillOp.getLoc()});
  ImplicitLocOpBuilder builder(loc, initOp);

  auto wordType = initOp.getMemory().getType().getWordType();
  auto addrType = initOp.getMemory().getType().getAddressType();
  auto indexType = builder.getIndexType();

  auto destBits = wordType.getIntOrFloatBitWidth();
  APInt constVal = fillOp.getValue().zextOrTrunc(destBits);

  if (fillOp.getRepeat()) {
    auto sourceBits = fillOp.getValueAttr().getType().getIntOrFloatBitWidth();
    unsigned shiftWidth = sourceBits;
    while (shiftWidth <= destBits) {
      constVal |= constVal << shiftWidth;
      shiftWidth *= 2;
    }
  }

  auto fillValue = builder.create<arith::ConstantOp>(
      builder.getIntegerAttr(wordType, constVal));
  auto zero =
      builder.create<arith::ConstantOp>(builder.getIntegerAttr(indexType, 0));
  auto one =
      builder.create<arith::ConstantOp>(builder.getIntegerAttr(indexType, 1));
  auto limit = builder.create<arith::ConstantOp>(builder.getIntegerAttr(
      indexType, initOp.getMemory().getType().getNumWords()));
  auto forOp = builder.create<scf::ForOp>(zero, limit, one);
  builder.setInsertionPointToStart(forOp.getBody());
  auto addr = builder.createOrFold<arith::IndexCastUIOp>(
      addrType, forOp.getInductionVar());
  builder.create<MemoryWriteOp>(initOp.getMemory(), addr, /*enable*/ Value{},
                                fillValue);

  return success();
}

LogicalResult
LowerMemoryInitializersPass::processInitializerFunction(func::FuncOp funcOp) {
  SmallVector<InitializeMemoryOp> initOps;
  SmallPtrSet<Operation *, 4> cleanupSet;

  funcOp.walk(
      [&](InitializeMemoryOp initMemOp) { initOps.push_back(initMemOp); });

  bool hasFailed = false;

  for (auto initOp : initOps) {
    auto defOp = initOp.getInitializer().getDefiningOp();
    if (!defOp) {
      initOp.emitError("Cannot lower initializer passed as argument.");
      return failure();
    }

    cleanupSet.insert(defOp);

    TypeSwitch<Operation *>(defOp)
        .Case<InitMemoryFilledOp>([&](auto op) {
          hasFailed |= failed(lowerFilledInitialization(initOp, op));
        })
        .Default([&](auto) {
          defOp->emitOpError("is not a supported memory intitializer.");
          hasFailed = true;
        });
  }

  if (hasFailed)
    return failure();

  for (auto initOp : initOps)
    initOp->erase();
  for (auto cleanupOp : cleanupSet)
    if (cleanupOp->getResult(0).getUses().empty())
      cleanupOp->erase();

  return success();
}

void LowerMemoryInitializersPass::runOnOperation() {
  symbolTable = nullptr;
  auto theModule = getOperation();
  for (auto modelOp : theModule.getOps<arc::ModelOp>()) {
    if (auto intitFnAttr = modelOp.getInitialFnAttr()) {
      if (!symbolTable)
        symbolTable = &getAnalysis<SymbolTable>();
      auto initFn = llvm::dyn_cast_or_null<func::FuncOp>(
          symbolTable->lookupSymbolIn(theModule, intitFnAttr));
      assert(!!initFn && "Failed to look-up initializer function.");
      if (failed(processInitializerFunction(initFn))) {
        signalPassFailure();
        return;
      }
    }
  }
}
