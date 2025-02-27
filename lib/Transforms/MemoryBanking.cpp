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
struct BankingConfigAttributes {
  Attribute factors;
  Attribute dimensions;
};

constexpr std::string_view bankingFactorsStr = "banking.factors";
constexpr std::string_view bankingDimensionsStr = "banking.dimensions";

/// Partition memories used in `affine.parallel` operation by the
/// `bankingFactor` throughout the program.
struct MemoryBankingPass
    : public circt::impl::MemoryBankingBase<MemoryBankingPass> {
  MemoryBankingPass(const MemoryBankingPass &other) = default;
  explicit MemoryBankingPass(ArrayRef<unsigned> bankingFactors = {},
                             ArrayRef<unsigned> bankingDimensions = {}) {}

  void runOnOperation() override;

  LogicalResult applyMemoryBanking(Operation *, MLIRContext *);

  SmallVector<Value, 4> createBanks(OpBuilder &builder, Value originalMem);

  void setAllBankingAttributes(Operation *, MLIRContext *);

private:
  SmallVector<unsigned, 4> bankingFactors;
  SmallVector<unsigned, 4> bankingDimensions;
  // map from original memory definition to newly allocated banks
  DenseMap<Value, SmallVector<Value>> memoryToBanks;
  DenseSet<Operation *> opsToErase;
  // Track memory references that need to be cleaned up after memory banking is
  // complete.
  DenseSet<Value> oldMemRefVals;
};
} // namespace

BankingConfigAttributes getMemRefBankingConfig(Value originalMem) {
  Attribute bankingFactorsAttr, bankingDimensionsAttr;
  if (auto blockArg = dyn_cast<BlockArgument>(originalMem)) {
    Block *block = blockArg.getOwner();

    auto *parentOp = block->getParentOp();
    auto funcOp = dyn_cast<func::FuncOp>(parentOp);
    assert(funcOp &&
           "Expected the original memory to be a FuncOp block argument!");
    unsigned argIndex = blockArg.getArgNumber();
    if (auto argAttrs = funcOp.getArgAttrDict(argIndex)) {
      bankingFactorsAttr = argAttrs.get(bankingFactorsStr);
      bankingDimensionsAttr = argAttrs.get(bankingDimensionsStr);
    }
  } else {
    Operation *originalDef = originalMem.getDefiningOp();
    bankingFactorsAttr = originalDef->getAttr(bankingFactorsStr);
    bankingDimensionsAttr = originalDef->getAttr(bankingDimensionsStr);
  }
  return BankingConfigAttributes{bankingFactorsAttr, bankingDimensionsAttr};
}

// Collect all memref in the `parOp`'s region'
DenseSet<Value> collectMemRefs(affine::AffineParallelOp affineParallelOp) {
  DenseSet<Value> memrefVals;
  affineParallelOp.walk([&](Operation *op) {
    if (!isa<affine::AffineWriteOpInterface>(op) &&
        !isa<affine::AffineReadOpInterface>(op))
      return WalkResult::advance();

    auto read = dyn_cast<affine::AffineReadOpInterface>(op);
    Value memref = read ? read.getMemRef()
                        : cast<affine::AffineWriteOpInterface>(op).getMemRef();
    memrefVals.insert(memref);
    return WalkResult::advance();
  });
  return memrefVals;
}

// Verify the banking configuration with different conditions.
void verifyBankingConfigurations(unsigned bankingFactor,
                                 unsigned bankingDimension,
                                 MemRefType originalType) {
  ArrayRef<int64_t> originalShape = originalType.getShape();
  assert(!originalShape.empty() && "memref shape should not be empty");
  assert(bankingDimension < originalType.getRank() &&
         "dimension must be within the memref rank");
  assert(originalShape[bankingDimension] % bankingFactor == 0 &&
         "memref shape must be evenly divided by the banking factor");
}

MemRefType computeBankedMemRefType(MemRefType originalType,
                                   uint64_t bankingFactor,
                                   unsigned bankingDimension) {
  ArrayRef<int64_t> originalShape = originalType.getShape();
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
SmallVector<Value, 4>
handleGetGlobalOp(memref::GetGlobalOp getGlobalOp, uint64_t bankingFactor,
                  unsigned bankingDimension, MemRefType newMemRefType,
                  OpBuilder &builder, DictionaryAttr remainingAttrs) {
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
    newGetGlobalOp->setAttrs(remainingAttrs);
    builder.setInsertionPointAfter(newGetGlobalOp);
    getGlobalOpsInsertPt = builder.saveInsertionPoint();

    banks.push_back(newGetGlobalOp);
  }

  globalOp.erase();
  return banks;
}

SmallVector<unsigned, 4>
getSpecifiedOrDefaultBankingDim(const ArrayRef<unsigned> bankingDimensions,
                                int64_t rank, ArrayRef<int64_t> shape) {
  // If the banking dimension is already specified, return it.
  if (!bankingDimensions.empty()) {
    return SmallVector<unsigned, 4>(bankingDimensions.begin(),
                                    bankingDimensions.end());
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
  return SmallVector<unsigned, 4>{static_cast<unsigned>(bankingDimension)};
}

// Update the argument types of `funcOp` by inserting `numInsertedArgs` number
// of `newMemRefType` after `argIndex`.
void updateFuncOpArgumentTypes(func::FuncOp funcOp, unsigned argIndex,
                               MemRefType newMemRefType,
                               unsigned numInsertedArgs) {
  auto originalArgTypes = funcOp.getFunctionType().getInputs();
  SmallVector<Type, 4> updatedArgTypes;

  // Rebuild the argument types, inserting new types for the newly added
  // arguments
  for (unsigned i = 0; i < originalArgTypes.size(); ++i) {
    updatedArgTypes.push_back(originalArgTypes[i]);

    // Insert new argument types after the specified argument index
    if (i == argIndex) {
      for (unsigned j = 0; j < numInsertedArgs; ++j) {
        updatedArgTypes.push_back(newMemRefType);
      }
    }
  }

  // Update the function type with the new argument types
  auto resultTypes = funcOp.getFunctionType().getResults();
  auto newFuncType =
      FunctionType::get(funcOp.getContext(), updatedArgTypes, resultTypes);
  funcOp.setType(newFuncType);
}

// Update `funcOp`'s "arg_attrs" by inserting `numInsertedArgs` number of
// `remainingAttrs` after `argIndex`.
void updateFuncOpArgAttrs(func::FuncOp funcOp, unsigned argIndex,
                          unsigned numInsertedArgs,
                          DictionaryAttr remainingAttrs) {
  ArrayAttr existingArgAttrs = funcOp->getAttrOfType<ArrayAttr>("arg_attrs");
  SmallVector<Attribute, 4> updatedArgAttrs;
  unsigned numArguments = funcOp.getNumArguments();
  unsigned newNumArguments = numArguments + numInsertedArgs;
  updatedArgAttrs.resize(newNumArguments);

  // Copy existing attributes, adjusting for the new arguments
  for (unsigned i = 0; i < numArguments; ++i) {
    // Shift attributes for arguments after the inserted ones.
    unsigned newIndex = (i > argIndex) ? i + numInsertedArgs : i;
    updatedArgAttrs[newIndex] = existingArgAttrs
                                    ? existingArgAttrs[i]
                                    : DictionaryAttr::get(funcOp.getContext());
  }

  // Initialize new attributes for the inserted arguments as empty dictionaries
  for (unsigned i = 0; i < numInsertedArgs; ++i) {
    updatedArgAttrs[argIndex + 1 + i] = remainingAttrs;
  }

  // Set the updated attributes.
  funcOp->setAttr("arg_attrs",
                  ArrayAttr::get(funcOp.getContext(), updatedArgAttrs));
}

unsigned getCurrBankingInfo(BankingConfigAttributes bankingConfigAttrs,
                            StringRef attrName) {
  auto getFirstInteger = [](Attribute attr) -> unsigned {
    if (auto arrayAttr = dyn_cast<ArrayAttr>(attr)) {
      assert(!arrayAttr.empty() &&
             "BankingConfig ArrayAttr should not be empty");
      auto intAttr = dyn_cast<IntegerAttr>(arrayAttr.getValue().front());
      assert(intAttr && "BankingConfig elements must be integers");
      return intAttr.getInt();
    }
    auto intAttr = dyn_cast<IntegerAttr>(attr);
    assert(intAttr && "BankingConfig attribute must be an integer");
    return intAttr.getInt();
  };

  if (attrName.str() == bankingFactorsStr) {
    return getFirstInteger(bankingConfigAttrs.factors);
  }

  assert(attrName.str() == bankingDimensionsStr &&
         "BankingConfig only contains 'factors' and 'dimensions' attributes");
  return getFirstInteger(bankingConfigAttrs.dimensions);
}

Attribute getRemainingBankingInfo(MLIRContext *context,
                                  BankingConfigAttributes bankingConfigAttrs,
                                  StringRef attrName) {
  if (attrName.str() == bankingFactorsStr) {
    if (auto arrayAttr = dyn_cast<ArrayAttr>(bankingConfigAttrs.factors)) {
      assert(!arrayAttr.empty() &&
             "BankingConfig factors ArrayAttr should not be empty");
      return arrayAttr.size() > 1
                 ? ArrayAttr::get(context, arrayAttr.getValue().take_back(
                                               arrayAttr.size() - 1))
                 : nullptr;
    }
    auto intAttr = dyn_cast<IntegerAttr>(bankingConfigAttrs.factors);
    assert(intAttr && "BankingConfig factor must be an integer");
    return nullptr;
  }

  assert(attrName.str() == bankingDimensionsStr &&
         "BankingConfig only contains 'factors' and 'dimensions' attributes");
  if (auto arrayAttr = dyn_cast<ArrayAttr>(bankingConfigAttrs.dimensions)) {
    assert(!arrayAttr.empty() &&
           "BankingConfig dimensions ArrayAttr should not be empty");
    return arrayAttr.size() > 1
               ? ArrayAttr::get(context, arrayAttr.getValue().take_back(
                                             arrayAttr.size() - 1))
               : nullptr;
  }
  auto intAttr = dyn_cast<IntegerAttr>(bankingConfigAttrs.dimensions);
  assert(intAttr && "BankingConfig dimension must be an integer");
  return nullptr;
}

SmallVector<Value, 4> MemoryBankingPass::createBanks(OpBuilder &builder,
                                                     Value originalMem) {
  MemRefType originalMemRefType = cast<MemRefType>(originalMem.getType());

  MLIRContext *context = builder.getContext();

  BankingConfigAttributes currBankingConfig =
      getMemRefBankingConfig(originalMem);

  unsigned currFactor =
      getCurrBankingInfo(currBankingConfig, bankingFactorsStr);
  unsigned currDimension =
      getCurrBankingInfo(currBankingConfig, bankingDimensionsStr);

  verifyBankingConfigurations(currFactor, currDimension, originalMemRefType);

  Attribute remainingFactors =
      getRemainingBankingInfo(context, currBankingConfig, bankingFactorsStr);
  Attribute remainingDimensions =
      getRemainingBankingInfo(context, currBankingConfig, bankingDimensionsStr);
  DictionaryAttr remainingAttrs =
      remainingFactors
          ? DictionaryAttr::get(
                context,
                {builder.getNamedAttr(bankingFactorsStr, remainingFactors),
                 builder.getNamedAttr(bankingDimensionsStr,
                                      remainingDimensions)})
          : DictionaryAttr::get(context);

  MemRefType newMemRefType =
      computeBankedMemRefType(originalMemRefType, currFactor, currDimension);
  SmallVector<Value, 4> banks;
  if (auto blockArgMem = dyn_cast<BlockArgument>(originalMem)) {
    Block *block = blockArgMem.getOwner();
    unsigned blockArgNum = blockArgMem.getArgNumber();

    for (unsigned i = 0; i < currFactor; ++i)
      block->insertArgument(blockArgNum + 1 + i, newMemRefType,
                            blockArgMem.getLoc());

    auto blockArgs = block->getArguments().slice(blockArgNum + 1, currFactor);
    banks.append(blockArgs.begin(), blockArgs.end());

    auto *parentOp = block->getParentOp();
    auto funcOp = dyn_cast<func::FuncOp>(parentOp);
    assert(funcOp && "BlockArgument is not part of a FuncOp");
    // Update the ArgumentTypes of `funcOp` so that we can correctly get
    // `getArgAttrDict` when resolving banking attributes across the iterations
    // of creating new banks.
    updateFuncOpArgumentTypes(funcOp, blockArgNum, newMemRefType, currFactor);
    updateFuncOpArgAttrs(funcOp, blockArgNum, currFactor, remainingAttrs);
  } else {
    Operation *originalDef = originalMem.getDefiningOp();
    Location loc = originalDef->getLoc();
    builder.setInsertionPointAfter(originalDef);
    TypeSwitch<Operation *>(originalDef)
        .Case<memref::AllocOp>([&](memref::AllocOp allocOp) {
          for (uint64_t bankCnt = 0; bankCnt < currFactor; ++bankCnt) {
            auto bankAllocOp =
                builder.create<memref::AllocOp>(loc, newMemRefType);
            bankAllocOp->setAttrs(remainingAttrs);
            banks.push_back(bankAllocOp);
          }
        })
        .Case<memref::AllocaOp>([&](memref::AllocaOp allocaOp) {
          for (uint64_t bankCnt = 0; bankCnt < currFactor; ++bankCnt) {
            auto bankAllocaOp =
                builder.create<memref::AllocaOp>(loc, newMemRefType);
            bankAllocaOp->setAttrs(remainingAttrs);
            banks.push_back(bankAllocaOp);
          }
        })
        .Case<memref::GetGlobalOp>([&](memref::GetGlobalOp getGlobalOp) {
          auto newBanks =
              handleGetGlobalOp(getGlobalOp, currFactor, currDimension,
                                newMemRefType, builder, remainingAttrs);
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
  BankAffineLoadPattern(MLIRContext *context,
                        DenseMap<Value, SmallVector<Value>> &memoryToBanks,
                        DenseSet<Value> &oldMemRefVals)
      : OpRewritePattern<mlir::affine::AffineLoadOp>(context),
        memoryToBanks(memoryToBanks), oldMemRefVals(oldMemRefVals) {}

  LogicalResult matchAndRewrite(mlir::affine::AffineLoadOp loadOp,
                                PatternRewriter &rewriter) const override {
    Location loc = loadOp.getLoc();
    auto originalMem = loadOp.getMemref();
    auto banks = memoryToBanks[originalMem];
    auto loadIndices = loadOp.getIndices();
    MemRefType originalMemRefType = loadOp.getMemRefType();
    int64_t memrefRank = originalMemRefType.getRank();

    BankingConfigAttributes currBankingConfig =
        getMemRefBankingConfig(originalMem);
    if (!currBankingConfig.factors) {
      // No need to rewrite anymore.
      return failure();
    }

    unsigned currFactor =
        getCurrBankingInfo(currBankingConfig, bankingFactorsStr);
    unsigned currDimension =
        getCurrBankingInfo(currBankingConfig, bankingDimensionsStr);

    verifyBankingConfigurations(currFactor, currDimension, originalMemRefType);

    auto modMap = AffineMap::get(
        /*dimCount=*/memrefRank, /*symbolCount=*/0,
        {rewriter.getAffineDimExpr(currDimension) % currFactor});
    auto divMap = AffineMap::get(
        memrefRank, 0,
        {rewriter.getAffineDimExpr(currDimension).floorDiv(currFactor)});

    Value bankIndex =
        rewriter.create<affine::AffineApplyOp>(loc, modMap, loadIndices);
    Value offset =
        rewriter.create<affine::AffineApplyOp>(loc, divMap, loadIndices);
    SmallVector<Value, 4> newIndices(loadIndices.begin(), loadIndices.end());
    newIndices[currDimension] = offset;

    SmallVector<Type> resultTypes = {loadOp.getResult().getType()};

    SmallVector<int64_t, 4> caseValues;
    for (unsigned i = 0; i < currFactor; ++i)
      caseValues.push_back(i);

    rewriter.setInsertionPoint(loadOp);
    scf::IndexSwitchOp switchOp = rewriter.create<scf::IndexSwitchOp>(
        loc, resultTypes, bankIndex, caseValues,
        /*numRegions=*/currFactor);

    for (unsigned i = 0; i < currFactor; ++i) {
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
    if (Value memRef = loadOp.getMemref(); isa<BlockArgument>(memRef)) {
      oldMemRefVals.insert(memRef);
    }
    rewriter.replaceOp(loadOp, switchOp.getResult(0));

    return success();
  }

private:
  DenseMap<Value, SmallVector<Value>> &memoryToBanks;
  DenseSet<Value> &oldMemRefVals;
};

// Replace the original store operations with newly created memory banks
struct BankAffineStorePattern
    : public OpRewritePattern<mlir::affine::AffineStoreOp> {
  BankAffineStorePattern(MLIRContext *context,
                         DenseMap<Value, SmallVector<Value>> &memoryToBanks,
                         DenseSet<Operation *> &opsToErase,
                         DenseSet<Operation *> &processedOps,
                         DenseSet<Value> &oldMemRefVals)
      : OpRewritePattern<mlir::affine::AffineStoreOp>(context),
        memoryToBanks(memoryToBanks), opsToErase(opsToErase),
        processedOps(processedOps), oldMemRefVals(oldMemRefVals) {}

  LogicalResult matchAndRewrite(mlir::affine::AffineStoreOp storeOp,
                                PatternRewriter &rewriter) const override {
    if (processedOps.contains(storeOp)) {
      return failure();
    }
    auto currConfig = getMemRefBankingConfig(storeOp.getMemref());
    if (!currConfig.factors) {
      // No need to rewrite anymore.
      return failure();
    }
    Location loc = storeOp.getLoc();
    auto originalMem = storeOp.getMemref();
    auto banks = memoryToBanks[originalMem];
    auto storeIndices = storeOp.getIndices();
    auto originalMemRefType = storeOp.getMemRefType();
    int64_t memrefRank = originalMemRefType.getRank();

    BankingConfigAttributes currBankingConfig =
        getMemRefBankingConfig(originalMem);

    unsigned currFactor =
        getCurrBankingInfo(currBankingConfig, bankingFactorsStr);
    unsigned currDimension =
        getCurrBankingInfo(currBankingConfig, bankingDimensionsStr);

    verifyBankingConfigurations(currFactor, currDimension, originalMemRefType);

    auto modMap = AffineMap::get(
        /*dimCount=*/memrefRank, /*symbolCount=*/0,
        {rewriter.getAffineDimExpr(currDimension) % currFactor});
    auto divMap = AffineMap::get(
        memrefRank, 0,
        {rewriter.getAffineDimExpr(currDimension).floorDiv(currFactor)});

    Value bankIndex =
        rewriter.create<affine::AffineApplyOp>(loc, modMap, storeIndices);
    Value offset =
        rewriter.create<affine::AffineApplyOp>(loc, divMap, storeIndices);
    SmallVector<Value, 4> newIndices(storeIndices.begin(), storeIndices.end());
    newIndices[currDimension] = offset;

    SmallVector<Type> resultTypes = {};

    SmallVector<int64_t, 4> caseValues;
    for (unsigned i = 0; i < currFactor; ++i)
      caseValues.push_back(i);

    rewriter.setInsertionPoint(storeOp);
    scf::IndexSwitchOp switchOp = rewriter.create<scf::IndexSwitchOp>(
        loc, resultTypes, bankIndex, caseValues,
        /*numRegions=*/currFactor);

    for (unsigned i = 0; i < currFactor; ++i) {
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
  DenseMap<func::FuncOp, SmallVector<unsigned, 4>> erasedArgIndices;
  for (auto &memrefVal : oldMemRefVals) {
    valuesToErase.push_back(memrefVal);
    if (auto blockArg = dyn_cast<BlockArgument>(memrefVal)) {
      if (auto funcOp =
              dyn_cast<func::FuncOp>(blockArg.getOwner()->getParentOp())) {
        funcsToModify.insert(funcOp);
        erasedArgIndices[funcOp].push_back(blockArg.getArgNumber());
      }
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

  // Modify the function argument attributes and function type accordingly
  for (auto funcOp : funcsToModify) {
    ArrayAttr existingArgAttrs = funcOp->getAttrOfType<ArrayAttr>("arg_attrs");
    if (existingArgAttrs) {
      SmallVector<Attribute, 4> updatedArgAttrs;
      auto erasedIndices = erasedArgIndices[funcOp];
      DenseSet<unsigned> indicesToErase(erasedIndices.begin(),
                                        erasedIndices.end());
      for (unsigned i = 0; i < existingArgAttrs.size(); ++i) {
        if (!indicesToErase.contains(i))
          updatedArgAttrs.push_back(existingArgAttrs[i]);
      }

      funcOp->setAttr("arg_attrs",
                      ArrayAttr::get(funcOp.getContext(), updatedArgAttrs));
    }

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

void verifyBankingAttributesSize(Attribute bankingFactorsAttr,
                                 Attribute bankingDimensionsAttr) {
  if (auto factorsArrayAttr = dyn_cast<ArrayAttr>(bankingFactorsAttr)) {
    assert(!factorsArrayAttr.empty() && "Banking factors should not be empty");
    if (auto dimsArrayAttr = dyn_cast<ArrayAttr>(bankingDimensionsAttr)) {
      assert(factorsArrayAttr.size() == dimsArrayAttr.size() &&
             "Banking factors/dimensions must be paired together");
    } else {
      auto dimsIntAttr = dyn_cast<IntegerAttr>(bankingDimensionsAttr);
      assert(dimsIntAttr && "banking.dimensions can either be an integer or an "
                            "array of integers");
      assert(factorsArrayAttr.size() == 1 &&
             "Banking factors/dimensions must be paired together");
    }
  } else {
    auto factorsIntAttr = dyn_cast<IntegerAttr>(bankingFactorsAttr);
    assert(factorsIntAttr &&
           "banking.factors can either be an integer or an array of integers");
    if (auto dimsArrayAttr = dyn_cast<ArrayAttr>(bankingDimensionsAttr)) {
      assert(dimsArrayAttr.size() == 1 &&
             "Banking factors/dimensions must be paired together");
    } else {
      auto dimsIntAttr = dyn_cast<IntegerAttr>(bankingDimensionsAttr);
      assert(dimsIntAttr && "banking.dimensions can either be an integer or an "
                            "array of integers");
    }
  }
}

void MemoryBankingPass::setAllBankingAttributes(Operation *operation,
                                                MLIRContext *context) {
  ArrayAttr defaultFactorsAttr = ArrayAttr::get(
      context,
      llvm::map_to_vector(bankingFactors, [&](unsigned factor) -> Attribute {
        return IntegerAttr::get(IntegerType::get(context, 32), factor);
      }));

  auto getDimensionsAttr =
      [&](SmallVector<unsigned, 4> specifiedOrDefaultDims) -> ArrayAttr {
    return ArrayAttr::get(
        context, llvm::map_to_vector(specifiedOrDefaultDims,
                                     [&](unsigned dim) -> Attribute {
                                       return IntegerAttr::get(
                                           IntegerType::get(context, 32), dim);
                                     }));
  };

  // Set or keep the memory banking related attributes for every memory-involved
  // affine operation.
  operation->walk([&](affine::AffineParallelOp affineParallelOp) {
    affineParallelOp.walk([&](Operation *op) {
      if (!isa<affine::AffineWriteOpInterface, affine::AffineReadOpInterface>(
              op))
        return WalkResult::advance();

      auto read = dyn_cast<affine::AffineReadOpInterface>(op);
      Value memref = read
                         ? read.getMemRef()
                         : cast<affine::AffineWriteOpInterface>(op).getMemRef();
      MemRefType memrefType =
          read ? read.getMemRefType()
               : cast<affine::AffineWriteOpInterface>(op).getMemRefType();

      if (auto *originalDef = memref.getDefiningOp()) {
        // Set the default factors using the command line option.
        if (!originalDef->getAttr(bankingFactorsStr)) {
          originalDef->setAttr(bankingFactorsStr, defaultFactorsAttr);
        }

        // Set the default `dimensions` either by the command line option or
        // inferencing if unspecified.
        if (!originalDef->getAttr(bankingDimensionsStr)) {
          SmallVector<unsigned, 4> specifiedOrDefaultDims =
              getSpecifiedOrDefaultBankingDim(bankingDimensions,
                                              memrefType.getRank(),
                                              memrefType.getShape());

          originalDef->setAttr(bankingDimensionsStr,
                               getDimensionsAttr(specifiedOrDefaultDims));
        }

        verifyBankingAttributesSize(originalDef->getAttr(bankingFactorsStr),
                                    originalDef->getAttr(bankingDimensionsStr));
      } else if (isa<BlockArgument>(memref)) {
        auto blockArg = cast<BlockArgument>(memref);
        auto *parentOp = blockArg.getOwner()->getParentOp();
        auto funcOp = dyn_cast<func::FuncOp>(parentOp);
        assert(funcOp &&
               "Expected the original memory to be a FuncOp block argument!");
        unsigned argIndex = blockArg.getArgNumber();
        SmallVector<unsigned, 4> specifiedOrDefaultDims =
            getSpecifiedOrDefaultBankingDim(
                bankingDimensions, memrefType.getRank(), memrefType.getShape());

        if (!funcOp.getArgAttr(argIndex, bankingFactorsStr))
          funcOp.setArgAttr(argIndex, bankingFactorsStr, defaultFactorsAttr);
        if (!funcOp.getArgAttr(argIndex, bankingDimensionsStr))
          funcOp.setArgAttr(argIndex, bankingDimensionsStr,
                            getDimensionsAttr(specifiedOrDefaultDims));

        verifyBankingAttributesSize(
            funcOp.getArgAttr(argIndex, bankingFactorsStr),
            funcOp.getArgAttr(argIndex, bankingDimensionsStr));
      }
      return WalkResult::advance();
    });
  });
}

void MemoryBankingPass::runOnOperation() {
  this->bankingFactors = {bankingFactorsList.begin(), bankingFactorsList.end()};
  this->bankingDimensions = {bankingDimensionsList.begin(),
                             bankingDimensionsList.end()};

  if (getOperation().isExternal() ||
      (bankingFactors.empty() ||
       std::all_of(bankingFactors.begin(), bankingFactors.end(),
                   [](unsigned f) { return f == 1; })))
    return;

  if (std::any_of(bankingFactors.begin(), bankingFactors.end(),
                  [](int f) { return f == 0; })) {
    getOperation().emitError("banking factor must be greater than 1");
    signalPassFailure();
    return;
  }

  if (bankingDimensions.size() > bankingFactors.size()) {
    getOperation().emitError(
        "A banking dimension must be paired with a factor");
    signalPassFailure();
    return;
  }
  // `bankingFactors` is guaranteed to have elements and at least one of them is
  // greater than 1 beyond this point.

  setAllBankingAttributes(getOperation(), &getContext());

  OpBuilder builder(getOperation());
  // We run this pass until convergence, i.e., `applyMemoryBanking` has reached
  // its fixed point, which means every memory read/write operation has been
  // rewritten to be using the newly created banks, and that the old memory
  // references are erased.
  bool banksCreated = false;
  do {
    memoryToBanks.clear();
    oldMemRefVals.clear();
    opsToErase.clear();

    banksCreated = false;
    getOperation().walk([&](mlir::affine::AffineParallelOp parOp) {
      DenseSet<Value> memrefsInPar = collectMemRefs(parOp);
      // We run `createBanks` iff there exists some `memrefVal` s.t. it has
      // banking attributes attached to it.
      for (auto memrefVal : memrefsInPar) {
        auto currConfig = getMemRefBankingConfig(memrefVal);
        if (!currConfig.factors) {
          continue;
        }
        auto [it, inserted] = memoryToBanks.insert(
            std::make_pair(memrefVal, SmallVector<Value>{}));
        if (inserted)
          it->second = createBanks(builder, memrefVal);
        banksCreated = true;
      }
    });

    if (failed(applyMemoryBanking(getOperation(), &getContext()))) {
      signalPassFailure();
      break;
    }
  } while (banksCreated);
};

LogicalResult MemoryBankingPass::applyMemoryBanking(Operation *operation,
                                                    MLIRContext *ctx) {
  RewritePatternSet patterns(ctx);

  DenseSet<Operation *> processedOps;
  patterns.add<BankAffineLoadPattern>(ctx, memoryToBanks, oldMemRefVals);
  patterns.add<BankAffineStorePattern>(ctx, memoryToBanks, opsToErase,
                                       processedOps, oldMemRefVals);
  patterns.add<BankReturnPattern>(ctx, memoryToBanks);

  GreedyRewriteConfig config;
  config.strictMode = GreedyRewriteStrictness::ExistingOps;
  if (failed(applyPatternsGreedily(operation, std::move(patterns), config))) {
    return failure();
  }

  // Clean up the old memref values
  if (failed(cleanUpOldMemRefs(oldMemRefVals, opsToErase))) {
    return failure();
  }

  return success();
}

namespace circt {
std::unique_ptr<mlir::Pass>
createMemoryBankingPass(ArrayRef<unsigned> bankingFactors,
                        ArrayRef<unsigned> bankingDimensions) {
  return std::make_unique<MemoryBankingPass>(bankingFactors, bankingDimensions);
}
} // namespace circt
