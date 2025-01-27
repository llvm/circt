//===- MemoryBanking.cpp - memory bank parallel loops -----------*- C++ -*-===//
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

#include "circt/Support/LLVM.h"
#include "circt/Transforms/Passes.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/OperationSupport.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/FormatVariadic.h"
#include <numeric>

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
  MemoryBankingPass(const MemoryBankingPass &other) = default;
  explicit MemoryBankingPass(
      std::optional<unsigned> bankingFactor = std::nullopt,
      std::optional<int> bankingDimension = std::nullopt) {}

  void runOnOperation() override;

private:
  // map from original memory definition to newly allocated banks
  DenseMap<Value, SmallVector<Value>> memoryToBanks;
  DenseSet<Operation *> opsToErase;
  // Track memory references that need to be cleaned up after memory banking is
  // complete.
  DenseSet<Value> oldMemRefVals;
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
                                   uint64_t bankingFactor,
                                   unsigned bankingDimension) {
  ArrayRef<int64_t> originalShape = originalType.getShape();
  assert(!originalShape.empty() && "memref shape should not be empty");

  assert(bankingDimension < originalType.getRank() &&
         "dimension must be within the memref rank");
  assert(originalShape[bankingDimension] % bankingFactor == 0 &&
         "memref shape must be evenly divided by the banking factor");
  SmallVector<int64_t, 4> newShape(originalShape.begin(), originalShape.end());
  newShape[bankingDimension] /= bankingFactor;
  MemRefType newMemRefType =
      MemRefType::get(newShape, originalType.getElementType(),
                      originalType.getLayout(), originalType.getMemorySpace());

  return newMemRefType;
}

// Decodes the flat index `linIndex` into an n-dimensional index based on the
// given `shape` of the array in row-major order. Returns an array to represent
// the n-dimensional indices.
SmallVector<int64_t> decodeIndex(int64_t linIndex, ArrayRef<int64_t> shape) {
  const unsigned rank = shape.size();
  SmallVector<int64_t> ndIndex(rank, 0);

  // Compute from last dimension to first because we assume row-major.
  for (int64_t d = rank - 1; d >= 0; --d) {
    ndIndex[d] = linIndex % shape[d];
    linIndex /= shape[d];
  }

  return ndIndex;
}

// Performs multi-dimensional slicing on `allAttrs` by extracting all elements
// whose coordinates range from `bankCnt`*`bankingDimension` to
// (`bankCnt`+1)*`bankingDimension` from `bankingDimension`'s dimension, leaving
// other dimensions alone.
SmallVector<SmallVector<Attribute>> sliceSubBlock(ArrayRef<Attribute> allAttrs,
                                                  ArrayRef<int64_t> memShape,
                                                  unsigned bankingDimension,
                                                  unsigned bankingFactor) {
  size_t numElements = std::reduce(memShape.begin(), memShape.end(), 1,
                                   std::multiplies<size_t>());
  // `bankingFactor` number of flattened attributes that store the information
  // in the original globalOp.
  SmallVector<SmallVector<Attribute>> subBlocks;
  subBlocks.resize(bankingFactor);

  for (unsigned linIndex = 0; linIndex < numElements; ++linIndex) {
    SmallVector<int64_t> ndIndex = decodeIndex(linIndex, memShape);
    unsigned subBlockIndex = ndIndex[bankingDimension] % bankingFactor;
    subBlocks[subBlockIndex].push_back(allAttrs[linIndex]);
  }

  return subBlocks;
}

// Handles the splitting of a GetGlobalOp into multiple banked memory and
// creates new GetGlobalOp to represent each banked memory by slicing the data
// in the original GetGlobalOp.
SmallVector<Value, 4> handleGetGlobalOp(memref::GetGlobalOp getGlobalOp,
                                        uint64_t bankingFactor,
                                        unsigned bankingDimension,
                                        MemRefType newMemRefType,
                                        OpBuilder &builder) {
  SmallVector<Value, 4> banks;
  auto memTy = cast<MemRefType>(getGlobalOp.getType());
  ArrayRef<int64_t> originalShape = memTy.getShape();
  auto newShape =
      SmallVector<int64_t>(originalShape.begin(), originalShape.end());
  newShape[bankingDimension] = originalShape[bankingDimension] / bankingFactor;

  auto *symbolTableOp = getGlobalOp->getParentWithTrait<OpTrait::SymbolTable>();
  auto globalOpNameAttr = getGlobalOp.getNameAttr();
  auto globalOp = dyn_cast_or_null<memref::GlobalOp>(
      SymbolTable::lookupSymbolIn(symbolTableOp, globalOpNameAttr));
  assert(globalOp && "The corresponding GlobalOp should exist in the module");
  MemRefType globalOpTy = globalOp.getType();

  auto cstAttr =
      dyn_cast_or_null<DenseElementsAttr>(globalOp.getConstantInitValue());
  auto attributes = cstAttr.getValues<Attribute>();
  SmallVector<Attribute, 8> allAttrs(attributes.begin(), attributes.end());

  auto subBlocks =
      sliceSubBlock(allAttrs, originalShape, bankingDimension, bankingFactor);

  // Initialize globalOp and getGlobalOp's insertion points. Since
  // bankingFactor is guaranteed to be greater than zero as it would
  // have early exited if not, the loop below will execute at least
  // once. So it's safe to manipulate the insertion points here.
  builder.setInsertionPointAfter(globalOp);
  OpBuilder::InsertPoint globalOpsInsertPt = builder.saveInsertionPoint();
  builder.setInsertionPointAfter(getGlobalOp);
  OpBuilder::InsertPoint getGlobalOpsInsertPt = builder.saveInsertionPoint();

  for (size_t bankCnt = 0; bankCnt < bankingFactor; ++bankCnt) {
    // Prepare relevant information to create a new GlobalOp
    auto newMemRefTy = MemRefType::get(newShape, globalOpTy.getElementType());
    auto newTypeAttr = TypeAttr::get(newMemRefTy);
    std::string newName = llvm::formatv(
        "{0}_{1}_{2}", globalOpNameAttr.getValue(), "bank", bankCnt);
    RankedTensorType tensorType =
        RankedTensorType::get({newShape}, globalOpTy.getElementType());
    auto newInitValue = DenseElementsAttr::get(tensorType, subBlocks[bankCnt]);

    builder.restoreInsertionPoint(globalOpsInsertPt);
    auto newGlobalOp = builder.create<memref::GlobalOp>(
        globalOp.getLoc(), builder.getStringAttr(newName),
        globalOp.getSymVisibilityAttr(), newTypeAttr, newInitValue,
        globalOp.getConstantAttr(), globalOp.getAlignmentAttr());
    builder.setInsertionPointAfter(newGlobalOp);
    globalOpsInsertPt = builder.saveInsertionPoint();

    builder.restoreInsertionPoint(getGlobalOpsInsertPt);
    auto newGetGlobalOp = builder.create<memref::GetGlobalOp>(
        getGlobalOp.getLoc(), newMemRefTy, newGlobalOp.getName());
    builder.setInsertionPointAfter(newGetGlobalOp);
    getGlobalOpsInsertPt = builder.saveInsertionPoint();

    banks.push_back(newGetGlobalOp);
  }

  globalOp.erase();
  return banks;
}

unsigned getBankingDimension(std::optional<int> bankingDimensionOpt,
                             int64_t rank, ArrayRef<int64_t> shape) {
  // If the banking dimension is already specified, return it.
  // Note, the banking dimension will always be nonempty because TableGen will
  // assign it with a default value -1 if it's not specified by the user. Thus,
  // -1 is the sentinel value to indicate the default behavior, which is the
  // innermost dimension with shape greater than 1.
  if (bankingDimensionOpt.has_value() && *bankingDimensionOpt >= 0) {
    return static_cast<unsigned>(*bankingDimensionOpt);
  }

  // Otherwise, find the innermost dimension with size > 1.
  // For example, [[1], [2], [3], [4]] with `bankingFactor`=2 will be banked to
  // [[1], [3]] and [[2], [4]].
  int bankingDimension = -1;
  for (int dim = rank - 1; dim >= 0; --dim) {
    if (shape[dim] > 1) {
      bankingDimension = dim;
      break;
    }
  }

  assert(bankingDimension >= 0 && "No eligible dimension for banking");
  return static_cast<unsigned>(bankingDimension);
}

SmallVector<Value, 4> createBanks(Value originalMem, uint64_t bankingFactor,
                                  std::optional<int> bankingDimensionOpt) {
  MemRefType originalMemRefType = cast<MemRefType>(originalMem.getType());
  unsigned rank = originalMemRefType.getRank();
  ArrayRef<int64_t> shape = originalMemRefType.getShape();

  auto bankingDimension = getBankingDimension(bankingDimensionOpt, rank, shape);

  MemRefType newMemRefType = computeBankedMemRefType(
      originalMemRefType, bankingFactor, bankingDimension);
  SmallVector<Value, 4> banks;
  if (auto blockArgMem = dyn_cast<BlockArgument>(originalMem)) {
    Block *block = blockArgMem.getOwner();
    unsigned blockArgNum = blockArgMem.getArgNumber();

    for (unsigned i = 0; i < bankingFactor; ++i)
      block->insertArgument(blockArgNum + 1 + i, newMemRefType,
                            blockArgMem.getLoc());

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
          for (uint64_t bankCnt = 0; bankCnt < bankingFactor; ++bankCnt) {
            auto bankAllocOp =
                builder.create<memref::AllocOp>(loc, newMemRefType);
            banks.push_back(bankAllocOp);
          }
        })
        .Case<memref::AllocaOp>([&](memref::AllocaOp allocaOp) {
          for (uint64_t bankCnt = 0; bankCnt < bankingFactor; ++bankCnt) {
            auto bankAllocaOp =
                builder.create<memref::AllocaOp>(loc, newMemRefType);
            banks.push_back(bankAllocaOp);
          }
        })
        .Case<memref::GetGlobalOp>([&](memref::GetGlobalOp getGlobalOp) {
          auto newBanks =
              handleGetGlobalOp(getGlobalOp, bankingFactor, bankingDimension,
                                newMemRefType, builder);
          banks.append(newBanks.begin(), newBanks.end());
        })
        .Default([](Operation *) {
          llvm_unreachable("Unhandled memory operation type");
        });
  }
  return banks;
}

// Replace the original load operations with newly created memory banks
struct BankAffineLoadPattern
    : public OpRewritePattern<mlir::affine::AffineLoadOp> {
  BankAffineLoadPattern(MLIRContext *context, uint64_t bankingFactor,
                        std::optional<int> bankingDimensionOpt,
                        DenseMap<Value, SmallVector<Value>> &memoryToBanks,
                        DenseSet<Value> &oldMemRefVals)
      : OpRewritePattern<mlir::affine::AffineLoadOp>(context),
        bankingFactor(bankingFactor), bankingDimensionOpt(bankingDimensionOpt),
        memoryToBanks(memoryToBanks), oldMemRefVals(oldMemRefVals) {}

  LogicalResult matchAndRewrite(mlir::affine::AffineLoadOp loadOp,
                                PatternRewriter &rewriter) const override {
    Location loc = loadOp.getLoc();
    auto banks = memoryToBanks[loadOp.getMemref()];
    auto loadIndices = loadOp.getIndices();
    int64_t memrefRank = loadOp.getMemRefType().getRank();
    ArrayRef<int64_t> shape = loadOp.getMemRefType().getShape();

    auto bankingDimension =
        getBankingDimension(bankingDimensionOpt, memrefRank, shape);

    auto modMap = AffineMap::get(
        /*dimCount=*/memrefRank, /*symbolCount=*/0,
        {rewriter.getAffineDimExpr(bankingDimension) % bankingFactor});
    auto divMap = AffineMap::get(
        memrefRank, 0,
        {rewriter.getAffineDimExpr(bankingDimension).floorDiv(bankingFactor)});

    Value bankIndex =
        rewriter.create<affine::AffineApplyOp>(loc, modMap, loadIndices);
    Value offset =
        rewriter.create<affine::AffineApplyOp>(loc, divMap, loadIndices);
    SmallVector<Value, 4> newIndices(loadIndices.begin(), loadIndices.end());
    newIndices[bankingDimension] = offset;

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
      Value bankedLoad = rewriter.create<mlir::affine::AffineLoadOp>(
          loc, banks[i], newIndices);
      rewriter.create<scf::YieldOp>(loc, bankedLoad);
    }

    Region &defaultRegion = switchOp.getDefaultRegion();
    assert(defaultRegion.empty() && "Default region should be empty");
    rewriter.setInsertionPointToStart(&defaultRegion.emplaceBlock());

    TypedAttr zeroAttr =
        cast<TypedAttr>(rewriter.getZeroAttr(loadOp.getType()));
    auto defaultValue = rewriter.create<arith::ConstantOp>(loc, zeroAttr);
    rewriter.create<scf::YieldOp>(loc, defaultValue.getResult());

    // We track Load's memory reference only if it is a block argument - this is
    // the only case where the reference isn't replaced.
    if (Value memRef = loadOp.getMemref(); isa<BlockArgument>(memRef))
      oldMemRefVals.insert(memRef);
    rewriter.replaceOp(loadOp, switchOp.getResult(0));

    return success();
  }

private:
  uint64_t bankingFactor;
  std::optional<int> bankingDimensionOpt;
  DenseMap<Value, SmallVector<Value>> &memoryToBanks;
  DenseSet<Value> &oldMemRefVals;
};

// Replace the original store operations with newly created memory banks
struct BankAffineStorePattern
    : public OpRewritePattern<mlir::affine::AffineStoreOp> {
  BankAffineStorePattern(MLIRContext *context, uint64_t bankingFactor,
                         std::optional<int> bankingDimensionOpt,
                         DenseMap<Value, SmallVector<Value>> &memoryToBanks,
                         DenseSet<Operation *> &opsToErase,
                         DenseSet<Operation *> &processedOps,
                         DenseSet<Value> &oldMemRefVals)
      : OpRewritePattern<mlir::affine::AffineStoreOp>(context),
        bankingFactor(bankingFactor), bankingDimensionOpt(bankingDimensionOpt),
        memoryToBanks(memoryToBanks), opsToErase(opsToErase),
        processedOps(processedOps), oldMemRefVals(oldMemRefVals) {}

  LogicalResult matchAndRewrite(mlir::affine::AffineStoreOp storeOp,
                                PatternRewriter &rewriter) const override {
    if (processedOps.contains(storeOp)) {
      return failure();
    }
    Location loc = storeOp.getLoc();
    auto banks = memoryToBanks[storeOp.getMemref()];
    auto storeIndices = storeOp.getIndices();
    int64_t memrefRank = storeOp.getMemRefType().getRank();
    ArrayRef<int64_t> shape = storeOp.getMemRefType().getShape();

    auto bankingDimension =
        getBankingDimension(bankingDimensionOpt, memrefRank, shape);

    auto modMap = AffineMap::get(
        /*dimCount=*/memrefRank, /*symbolCount=*/0,
        {rewriter.getAffineDimExpr(bankingDimension) % bankingFactor});
    auto divMap = AffineMap::get(
        memrefRank, 0,
        {rewriter.getAffineDimExpr(bankingDimension).floorDiv(bankingFactor)});

    Value bankIndex =
        rewriter.create<affine::AffineApplyOp>(loc, modMap, storeIndices);
    Value offset =
        rewriter.create<affine::AffineApplyOp>(loc, divMap, storeIndices);
    SmallVector<Value, 4> newIndices(storeIndices.begin(), storeIndices.end());
    newIndices[bankingDimension] = offset;

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
          loc, storeOp.getValueToStore(), banks[i], newIndices);
      rewriter.create<scf::YieldOp>(loc);
    }

    Region &defaultRegion = switchOp.getDefaultRegion();
    assert(defaultRegion.empty() && "Default region should be empty");
    rewriter.setInsertionPointToStart(&defaultRegion.emplaceBlock());

    rewriter.create<scf::YieldOp>(loc);

    processedOps.insert(storeOp);
    opsToErase.insert(storeOp);
    oldMemRefVals.insert(storeOp.getMemref());

    return success();
  }

private:
  uint64_t bankingFactor;
  std::optional<int> bankingDimensionOpt;
  DenseMap<Value, SmallVector<Value>> &memoryToBanks;
  DenseSet<Operation *> &opsToErase;
  DenseSet<Operation *> &processedOps;
  DenseSet<Value> &oldMemRefVals;
};

// Replace the original return operation with newly created memory banks
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

// Clean up the empty uses old memory values by either erasing the defining
// operation or replace the block arguments with new ones that corresponds to
// the newly created banks. Change the function signature if the old memory
// values are used as function arguments and/or return values.
LogicalResult cleanUpOldMemRefs(DenseSet<Value> &oldMemRefVals,
                                DenseSet<Operation *> &opsToErase) {
  DenseSet<func::FuncOp> funcsToModify;
  SmallVector<Value, 4> valuesToErase;
  for (auto &memrefVal : oldMemRefVals) {
    valuesToErase.push_back(memrefVal);
    if (auto blockArg = dyn_cast<BlockArgument>(memrefVal)) {
      if (auto funcOp =
              dyn_cast<func::FuncOp>(blockArg.getOwner()->getParentOp()))
        funcsToModify.insert(funcOp);
    }
  }

  for (auto *op : opsToErase) {
    op->erase();
  }
  // Erase values safely.
  for (auto &memrefVal : valuesToErase) {
    assert(memrefVal.use_empty() && "use must be empty");
    if (auto blockArg = dyn_cast<BlockArgument>(memrefVal)) {
      blockArg.getOwner()->eraseArgument(blockArg.getArgNumber());
    } else if (auto *op = memrefVal.getDefiningOp()) {
      op->erase();
    }
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
  if (getOperation().isExternal() || bankingFactor == 1)
    return;

  if (bankingFactor == 0) {
    getOperation().emitError("banking factor must be greater than 1");
    signalPassFailure();
    return;
  }

  getOperation().walk([&](mlir::affine::AffineParallelOp parOp) {
    DenseSet<Value> memrefsInPar = collectMemRefs(parOp);

    for (auto memrefVal : memrefsInPar) {
      auto [it, inserted] =
          memoryToBanks.insert(std::make_pair(memrefVal, SmallVector<Value>{}));
      if (inserted)
        it->second = createBanks(memrefVal, bankingFactor, bankingDimension);
    }
  });

  auto *ctx = &getContext();
  RewritePatternSet patterns(ctx);

  DenseSet<Operation *> processedOps;
  patterns.add<BankAffineLoadPattern>(ctx, bankingFactor, bankingDimension,
                                      memoryToBanks, oldMemRefVals);
  patterns.add<BankAffineStorePattern>(ctx, bankingFactor, bankingDimension,
                                       memoryToBanks, opsToErase, processedOps,
                                       oldMemRefVals);
  patterns.add<BankReturnPattern>(ctx, memoryToBanks);

  GreedyRewriteConfig config;
  config.strictMode = GreedyRewriteStrictness::ExistingOps;
  if (failed(
          applyPatternsGreedily(getOperation(), std::move(patterns), config))) {
    signalPassFailure();
  }

  // Clean up the old memref values
  if (failed(cleanUpOldMemRefs(oldMemRefVals, opsToErase))) {
    signalPassFailure();
  }
}

namespace circt {
std::unique_ptr<mlir::Pass>
createMemoryBankingPass(std::optional<unsigned> bankingFactor,
                        std::optional<int> bankingDimension) {
  return std::make_unique<MemoryBankingPass>(bankingFactor, bankingDimension);
}
} // namespace circt
