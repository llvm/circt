//===- MemoryBanking.cpp - Code to perform memory bnaking in parallel loops
//--------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements parallel loop memory banking.
//
//===----------------------------------------------------------------------===//

#include "circt/Transforms/Passes.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Affine/LoopUtils.h"
#include "mlir/Dialect/Affine/Passes.h"
#include "mlir/Dialect/Affine/Utils.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/OperationSupport.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/ADT/TypeSwitch.h"

namespace circt {
#define GEN_PASS_DEF_MEMORYBANKING
#include "circt/Transforms/Passes.h.inc"
} // namespace circt

using namespace mlir;
using namespace circt;

namespace {

/// Partition memories used in `affine.parallel` operation by the
/// `bankingFactor` throughout the program.
struct MemoryBankingPass
    : public circt::impl::MemoryBankingBase<MemoryBankingPass> {
  const std::function<unsigned(mlir::affine::AffineParallelOp)>
      getBankingFactor;
  MemoryBankingPass() : getBankingFactor(nullptr) {}
  MemoryBankingPass(const MemoryBankingPass &other) = default;
  explicit MemoryBankingPass(
      std::optional<unsigned> bankingFactor = std::nullopt,
      const std::function<unsigned(mlir::affine::AffineParallelOp)>
          &getBankingFactor = nullptr)
      : getBankingFactor(getBankingFactor) {
    if (bankingFactor)
      this->bankingFactor = *bankingFactor;
  }

  void runOnOperation() override;
  LogicalResult parallelBankingByFactor(mlir::affine::AffineParallelOp parOp,
                                        uint64_t bankingFactor);

private:
  // map from original memory definition to newly allocated banks
  DenseMap<Value, SmallVector<Value>> memoryToBanks;
};
} // namespace

// Collect all memref in the `parOp`'s region'
DenseSet<Value> collectMemRefs(mlir::affine::AffineParallelOp parOp) {
  DenseSet<Value> memrefVals;
  parOp.walk([&](Operation *op) {
    for (auto operand : op->getOperands()) {
      if (isa<MemRefType>(operand.getType()))
        memrefVals.insert(operand);
    }
    return WalkResult::advance();
  });
  return memrefVals;
}

MemRefType computeBankedMemRefType(MemRefType originalType,
                                   uint64_t bankingFactor) {
  ArrayRef<int64_t> originalShape = originalType.getShape();
  assert(!originalShape.empty() && "memref shape should not be empty");
  assert(originalType.getRank() == 1 &&
         "currently only support one dimension memories");
  SmallVector<int64_t, 4> newShape(originalShape.begin(), originalShape.end());
  assert(newShape.front() % bankingFactor == 0 &&
         "memref shape must be divided by the banking factor");
  newShape.front() /= bankingFactor;
  MemRefType newMemRefType =
      MemRefType::get(newShape, originalType.getElementType(),
                      originalType.getLayout(), originalType.getMemorySpace());

  return newMemRefType;
}

SmallVector<Value, 4> createBanks(Value originalMem, uint64_t bankingFactor) {
  MemRefType originalMemRefType = cast<MemRefType>(originalMem.getType());
  MemRefType newMemRefType =
      computeBankedMemRefType(originalMemRefType, bankingFactor);
  SmallVector<Value, 4> banks;
  if (auto blockArgMem = dyn_cast<BlockArgument>(originalMem)) {
    Block *block = blockArgMem.getOwner();
    unsigned blockArgNum = blockArgMem.getArgNumber();

    SmallVector<Type> banksType;
    for (unsigned i = 0; i < bankingFactor; ++i) {
      block->insertArgument(blockArgNum + 1 + i, newMemRefType,
                            blockArgMem.getLoc());
    }

    auto blockArgs =
        block->getArguments().slice(blockArgNum + 1, bankingFactor);
    banks.append(blockArgs.begin(), blockArgs.end());
  } else {
    Operation *originalDef = originalMem.getDefiningOp();
    Location loc = originalDef->getLoc();
    OpBuilder builder(originalDef);
    builder.setInsertionPointAfter(originalDef);
    TypeSwitch<Operation *>(originalDef)
        .Case<memref::AllocOp>([&](memref::AllocOp allocOp) {
          for (uint bankCnt = 0; bankCnt < bankingFactor; bankCnt++) {
            auto bankAllocOp =
                builder.create<memref::AllocOp>(loc, newMemRefType);
            banks.push_back(bankAllocOp);
          }
        })
        .Case<memref::AllocaOp>([&](memref::AllocaOp allocaOp) {
          for (uint bankCnt = 0; bankCnt < bankingFactor; bankCnt++) {
            auto bankAllocaOp =
                builder.create<memref::AllocaOp>(loc, newMemRefType);
            banks.push_back(bankAllocaOp);
          }
        })
        .Default([](Operation *) {
          llvm_unreachable("Unhandled memory operation type");
        });
  }
  return banks;
}

struct BankAffineLoadPattern
    : public OpRewritePattern<mlir::affine::AffineLoadOp> {
  BankAffineLoadPattern(MLIRContext *context, uint64_t bankingFactor,
                        DenseMap<Value, SmallVector<Value>> &memoryToBanks)
      : OpRewritePattern<mlir::affine::AffineLoadOp>(context),
        bankingFactor(bankingFactor), memoryToBanks(memoryToBanks) {}

  LogicalResult matchAndRewrite(mlir::affine::AffineLoadOp loadOp,
                                PatternRewriter &rewriter) const override {
    Location loc = loadOp.getLoc();
    auto banks = memoryToBanks[loadOp.getMemref()];
    Value loadIndex = loadOp.getIndices().front();
    auto modMap =
        AffineMap::get(1, 0, {rewriter.getAffineDimExpr(0) % bankingFactor});
    auto divMap = AffineMap::get(
        1, 0, {rewriter.getAffineDimExpr(0).floorDiv(bankingFactor)});

    Value bankIndex = rewriter.create<affine::AffineApplyOp>(
        loc, modMap, loadIndex); // assuming one-dim
    Value offset =
        rewriter.create<affine::AffineApplyOp>(loc, divMap, loadIndex);

    SmallVector<Type> resultTypes = {loadOp.getResult().getType()};

    SmallVector<int64_t, 4> caseValues;
    for (unsigned i = 0; i < bankingFactor; ++i)
      caseValues.push_back(i);

    rewriter.setInsertionPoint(loadOp);
    scf::IndexSwitchOp switchOp = rewriter.create<scf::IndexSwitchOp>(
        loc, resultTypes, bankIndex, caseValues,
        /*numRegions=*/bankingFactor);

    for (unsigned i = 0; i < bankingFactor; ++i) {
      Region &caseRegion = switchOp.getCaseRegions()[i];
      rewriter.setInsertionPointToStart(&caseRegion.emplaceBlock());
      Value bankedLoad =
          rewriter.create<mlir::affine::AffineLoadOp>(loc, banks[i], offset);
      rewriter.create<scf::YieldOp>(loc, bankedLoad);
    }

    Region &defaultRegion = switchOp.getDefaultRegion();
    assert(defaultRegion.empty() && "Default region should be empty");
    rewriter.setInsertionPointToStart(&defaultRegion.emplaceBlock());

    TypedAttr zeroAttr =
        cast<TypedAttr>(rewriter.getZeroAttr(loadOp.getType()));
    auto defaultValue = rewriter.create<arith::ConstantOp>(loc, zeroAttr);
    rewriter.create<scf::YieldOp>(loc, defaultValue.getResult());

    loadOp.getResult().replaceAllUsesWith(switchOp.getResult(0));

    rewriter.eraseOp(loadOp);
    return success();
  }

private:
  uint64_t bankingFactor;
  DenseMap<Value, SmallVector<Value>> &memoryToBanks;
};

struct BankAffineStorePattern
    : public OpRewritePattern<mlir::affine::AffineStoreOp> {
  BankAffineStorePattern(MLIRContext *context, uint64_t bankingFactor,
                         DenseMap<Value, SmallVector<Value>> &memoryToBanks)
      : OpRewritePattern<mlir::affine::AffineStoreOp>(context),
        bankingFactor(bankingFactor), memoryToBanks(memoryToBanks) {}

  LogicalResult matchAndRewrite(mlir::affine::AffineStoreOp storeOp,
                                PatternRewriter &rewriter) const override {
    Location loc = storeOp.getLoc();
    auto banks = memoryToBanks[storeOp.getMemref()];
    Value storeIndex = storeOp.getIndices().front();

    auto modMap =
        AffineMap::get(1, 0, {rewriter.getAffineDimExpr(0) % bankingFactor});
    auto divMap = AffineMap::get(
        1, 0, {rewriter.getAffineDimExpr(0).floorDiv(bankingFactor)});

    Value bankIndex = rewriter.create<affine::AffineApplyOp>(
        loc, modMap, storeIndex); // assuming one-dim
    Value offset =
        rewriter.create<affine::AffineApplyOp>(loc, divMap, storeIndex);

    SmallVector<Type> resultTypes = {};

    SmallVector<int64_t, 4> caseValues;
    for (unsigned i = 0; i < bankingFactor; ++i)
      caseValues.push_back(i);

    rewriter.setInsertionPoint(storeOp);
    scf::IndexSwitchOp switchOp = rewriter.create<scf::IndexSwitchOp>(
        loc, resultTypes, bankIndex, caseValues,
        /*numRegions=*/bankingFactor);

    for (unsigned i = 0; i < bankingFactor; ++i) {
      Region &caseRegion = switchOp.getCaseRegions()[i];
      rewriter.setInsertionPointToStart(&caseRegion.emplaceBlock());
      rewriter.create<mlir::affine::AffineStoreOp>(
          loc, storeOp.getValueToStore(), banks[i], offset);
      rewriter.create<scf::YieldOp>(loc);
    }

    Region &defaultRegion = switchOp.getDefaultRegion();
    assert(defaultRegion.empty() && "Default region should be empty");
    rewriter.setInsertionPointToStart(&defaultRegion.emplaceBlock());

    rewriter.create<scf::YieldOp>(loc);

    rewriter.eraseOp(storeOp);
    return success();
  }

private:
  uint64_t bankingFactor;
  DenseMap<Value, SmallVector<Value>> &memoryToBanks;
};

struct BankReturnPattern : public OpRewritePattern<func::ReturnOp> {
  BankReturnPattern(MLIRContext *context,
                    DenseMap<Value, SmallVector<Value>> &memoryToBanks)
      : OpRewritePattern<func::ReturnOp>(context),
        memoryToBanks(memoryToBanks) {}

  LogicalResult matchAndRewrite(func::ReturnOp returnOp,
                                PatternRewriter &rewriter) const override {
    Location loc = returnOp.getLoc();
    SmallVector<Value, 4> newReturnOperands;
    bool allOrigMemsUsedByReturn = true;
    for (auto operand : returnOp.getOperands()) {
      if (!memoryToBanks.contains(operand)) {
        newReturnOperands.push_back(operand);
        continue;
      }
      if (operand.hasOneUse())
        allOrigMemsUsedByReturn = false;
      auto banks = memoryToBanks[operand];
      newReturnOperands.append(banks.begin(), banks.end());
    }

    func::FuncOp funcOp = returnOp.getParentOp();
    rewriter.setInsertionPointToEnd(&funcOp.getBlocks().front());
    auto newReturnOp =
        rewriter.create<func::ReturnOp>(loc, ValueRange(newReturnOperands));
    TypeRange newReturnType = TypeRange(newReturnOperands);
    FunctionType newFuncType =
        FunctionType::get(funcOp.getContext(),
                          funcOp.getFunctionType().getInputs(), newReturnType);
    funcOp.setType(newFuncType);

    if (allOrigMemsUsedByReturn)
      rewriter.replaceOp(returnOp, newReturnOp);

    return success();
  }

private:
  DenseMap<Value, SmallVector<Value>> &memoryToBanks;
};

LogicalResult cleanUpOldMemRefs(DenseSet<Value> &oldMemRefVals) {
  DenseSet<func::FuncOp> funcsToModify;
  for (auto &memrefVal : oldMemRefVals) {
    if (!memrefVal.use_empty())
      continue;
    if (auto blockArg = dyn_cast<BlockArgument>(memrefVal)) {
      Block *block = blockArg.getOwner();
      block->eraseArgument(blockArg.getArgNumber());
      if (auto funcOp = dyn_cast<func::FuncOp>(block->getParentOp()))
        funcsToModify.insert(funcOp);
    } else
      memrefVal.getDefiningOp()->erase();
  }

  // Modify the function type accordingly
  for (auto funcOp : funcsToModify) {
    SmallVector<Type, 4> newArgTypes;
    for (BlockArgument arg : funcOp.getArguments()) {
      newArgTypes.push_back(arg.getType());
    }
    FunctionType newFuncType =
        FunctionType::get(funcOp.getContext(), newArgTypes,
                          funcOp.getFunctionType().getResults());
    funcOp.setType(newFuncType);
  }
  return success();
}

void MemoryBankingPass::runOnOperation() {
  if (getOperation().isExternal()) {
    return;
  }

  getOperation().walk([&](mlir::affine::AffineParallelOp parOp) {
    DenseSet<Value> memrefsInPar = collectMemRefs(parOp);

    for (auto memrefVal : memrefsInPar) {
      SmallVector<Value> banks = createBanks(memrefVal, bankingFactor);
      memoryToBanks[memrefVal] = banks;
    }
  });

  auto *ctx = &getContext();
  RewritePatternSet patterns(ctx);

  patterns.add<BankAffineLoadPattern>(ctx, bankingFactor, memoryToBanks);
  patterns.add<BankAffineStorePattern>(ctx, bankingFactor, memoryToBanks);
  patterns.add<BankReturnPattern>(ctx, memoryToBanks);

  GreedyRewriteConfig config;
  config.strictMode = GreedyRewriteStrictness::ExistingOps;

  if (failed(applyPatternsAndFoldGreedily(getOperation(), std::move(patterns),
                                          config))) {
    signalPassFailure();
  }

  // Clean up the old memref values
  DenseSet<Value> oldMemRefVals;
  for (const auto &pair : memoryToBanks)
    oldMemRefVals.insert(pair.first);

  if (failed(cleanUpOldMemRefs(oldMemRefVals))) {
    signalPassFailure();
  }
}

namespace circt {
std::unique_ptr<mlir::Pass> createMemoryBankingPass(
    int bankingFactor,
    const std::function<unsigned(mlir::affine::AffineParallelOp)>
        &getBankingFactor) {
  return std::make_unique<MemoryBankingPass>(
      bankingFactor == -1 ? std::nullopt
                          : std::optional<unsigned>(bankingFactor),
      getBankingFactor);
}
} // namespace circt
