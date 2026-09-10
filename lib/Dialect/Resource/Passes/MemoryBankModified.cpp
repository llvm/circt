//===- MemoryBanking.cpp - memory bank affine memrefs -----------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements memref banking driven by `hls.array_partition`
// attributes. It is NOT restricted to memories used inside `affine.parallel`:
// every memref carrying an `hls.array_partition` spec (on its defining op, or
// as a function-argument attribute) is banked, and every affine load/store of
// it is rewritten. Supports cyclic / block / complete partitioning and
// Vitis-style 1-based dimension numbering.
//
//===----------------------------------------------------------------------===//

#include "circt/Support/LLVM.h"
#include "circt/Transforms/Passes.h"
#include "circt/Dialect/Resource/Interfaces/MemoryDS.h"
#include "circt/Dialect/Resource/HLS/HLSOps.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/OperationSupport.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/FormatVariadic.h"

#include <numeric>
#include <optional>
#include <utility>
#include <circt/Dialect/Resource/Interfaces/MemoryOpInterface.h.inc>
#include <mlir/IR/IntegerSet.h>

namespace circt::hls_analysis {
#define GEN_PASS_DEF_MEMORYBANKING
#include "circt/Dialect/Resource/Passes/Passes.h.inc"
} // namespace circt

using namespace mlir;
using namespace circt;
using namespace circt::hls_analysis;

// The factor actually used for banking. For Complete this is the dimension
// extent (one bank per element); otherwise the user-specified factor.
static unsigned effectiveFactor(const PartitionSpec &spec, MemRefType ty) {
  if (spec.kind == PartitionSpec::Complete)
    return static_cast<unsigned>(ty.getShape()[spec.dim]);
  return spec.factor;
}

// Compute the (bankIndex, intra-bank offset) affine expressions for an
// effective access expression `idxExpr` along the banked dimension.
//   cyclic / complete:  bank = idx % factor      offset = idx floordiv factor
//   block:              bank = idx floordiv bs   offset = idx % bs   (bs=ext/f)
static std::pair<AffineExpr, AffineExpr>
computeBankAndOffset(AffineExpr idxExpr, const PartitionSpec &spec,
                     MemRefType originalType) {
  unsigned f = effectiveFactor(spec, originalType);
  if (spec.kind == PartitionSpec::Block) {
    // Block size is ceil(extent / factor); element i lives in bank i / bs at
    // offset i % bs. The final bank may hold fewer than bs live elements.
    int64_t extent = originalType.getShape()[spec.dim];
    int64_t blockSize =
        (extent + static_cast<int64_t>(f) - 1) / static_cast<int64_t>(f);
    return {idxExpr.floorDiv(blockSize), idxExpr % blockSize};
  }
  // Cyclic, and Complete (cyclic with factor == dim extent).
  return {idxExpr % static_cast<int64_t>(f),
          idxExpr.floorDiv(static_cast<int64_t>(f))};
}

namespace {


// Resolve a non-affine (memref) access along the banked dim.
struct MemrefBankAccess {
  SmallVector<Value, 4> bankedIndices;  // original indices, banked dim -> offset
  Value bankIdx;                        // runtime bank index (null if constBank)
  std::optional<unsigned> constBank;    // set when the index folds
};

static MemrefBankAccess
computeMemrefBankAccess(OpBuilder &b, Location loc, ValueRange indices,
                        const PartitionSpec &spec, MemRefType origTy) {
  unsigned f = effectiveFactor(spec, origTy);
  Value idx = indices[spec.dim];

  // block:  bank = idx / bs,  offset = idx % bs   (bs = ceil(extent/f))
  // cyclic: bank = idx % f,   offset = idx / f
  bool bankIsDiv = spec.kind == PartitionSpec::Block;
  int64_t divisor;
  if (bankIsDiv) {
    int64_t extent = origTy.getShape()[spec.dim];
    divisor = (extent + (int64_t)f - 1) / (int64_t)f;
  } else {
    divisor = (int64_t)f;
  }

  MemrefBankAccess mba;
  mba.bankedIndices.assign(indices.begin(), indices.end());

  if (auto c = getConstantIntValue(idx)) {
    int64_t v = *c;
    int64_t bank = bankIsDiv ? v / divisor : v % divisor;
    int64_t off  = bankIsDiv ? v % divisor : v / divisor;
    mba.constBank = (unsigned)bank;
    mba.bankedIndices[spec.dim] = arith::ConstantIndexOp::create(b, loc, off);
    return mba;
  }

  // memref indices are non-negative index-typed, so unsigned div/rem is correct.
  Value d = arith::ConstantIndexOp::create(b, loc, divisor);
  Value bankV = bankIsDiv ? arith::DivUIOp::create(b, loc, idx, d).getResult()
                          : arith::RemUIOp::create(b, loc, idx, d).getResult();
  Value offV  = bankIsDiv ? arith::RemUIOp::create(b, loc, idx, d).getResult()
                          : arith::DivUIOp::create(b, loc, idx, d).getResult();
  mba.bankIdx = bankV;
  mba.bankedIndices[spec.dim] = offV;
  return mba;
}

// Resolved banking for one access: the bank-index map, the banked access map
// (original map with the banked dim replaced by the intra-bank offset), and a
// constant bank if the index folds.
struct BankAccess {
  AffineMap bankMap;             // single-result, = simplified bank expr
  AffineMap bankedAccessMap;     // original results, banked dim -> offset
  std::optional<unsigned> constBank;
};

BankAccess computeBankAccess(AffineMap accessMap,
                                    const PartitionSpec &spec,
                                    MemRefType origTy, MLIRContext *ctx) {
  
  AffineExpr idx = accessMap.getResult(spec.dim);
  auto [bankExpr, offsetExpr] = computeBankAndOffset(idx, spec, origTy);
 
  unsigned nDims = accessMap.getNumDims();
  unsigned nSyms = accessMap.getNumSymbols();
 
  BankAccess ba;
  AffineExpr simp = simplifyAffineExpr(bankExpr, nDims, nSyms);
  ba.bankMap = AffineMap::get(nDims, nSyms, simp);
 
  SmallVector<AffineExpr, 4> results(accessMap.getResults().begin(),
                                     accessMap.getResults().end());
  results[spec.dim] = offsetExpr;
  ba.bankedAccessMap = AffineMap::get(nDims, nSyms, results, ctx);
 
  if (auto c = dyn_cast<AffineConstantExpr>(simp))
    ba.constBank = static_cast<unsigned>(c.getValue());
  return ba;
}

// Partition memories carrying `hls.array_partition` throughout the function.
struct MemoryBankingPass
    : public hls_analysis::impl::MemoryBankingBase<MemoryBankingPass> {
  MemoryBankingPass(const MemoryBankingPass &other) = default;
  explicit MemoryBankingPass() {}

  void runOnOperation() override;

  LogicalResult applyMemoryBanking(Operation *, MLIRContext *);

  SmallVector<Value, 4> createBanks(OpBuilder &builder, Value originalMem);

private:
  // map from original memory definition to newly allocated banks
  DenseMap<Value, SmallVector<Value>> memoryToBanks;
  DenseSet<Operation *> opsToErase;
  // Track memory references that need cleanup after banking is complete.
  DenseSet<Value> oldMemRefVals;
};
} // namespace

// Collect every memref referenced by an affine read/write in `funcOp`.
static DenseSet<Value> collectMemRefs(func::FuncOp funcOp) {
  DenseSet<Value> memrefVals;
  funcOp.walk([&](Operation *op) {
    Value memref;
    if (auto read = dyn_cast<affine::AffineReadOpInterface>(op))
      memref = read.getMemRef();
    else if (auto write = dyn_cast<affine::AffineWriteOpInterface>(op))
      memref = write.getMemRef();
    else
      return WalkResult::advance();
    memrefVals.insert(memref);
    return WalkResult::advance();
  });
  return memrefVals;
}

static void verifyBankingConfigurations(unsigned bankingFactor,
                                        unsigned bankingDimension,
                                        MemRefType originalType) {
  [[maybe_unused]] ArrayRef<int64_t> originalShape = originalType.getShape();
  assert(!originalShape.empty() && "memref shape should not be empty");
  assert(bankingDimension < originalType.getRank() &&
         "dimension must be within the memref rank");
  // Vitis does not require even division: an uneven factor yields banks of
  // depth ceil(extent / factor), with the trailing slots of some banks unused.
  // We only require the factor to fit within the dimension. factor == extent is
  // complete partitioning (depth-1 banks), so the bound is inclusive.
  assert(bankingFactor >= 1 && "banking factor must be positive");
  assert(bankingFactor <= originalShape[bankingDimension] &&
         "banking factor must not exceed the dimension extent");
}

static MemRefType computeBankedMemRefType(MemRefType originalType,
                                          uint64_t bankingFactor,
                                          unsigned bankingDimension) {
  ArrayRef<int64_t> originalShape = originalType.getShape();
  SmallVector<int64_t, 4> newShape(originalShape.begin(), originalShape.end());
  // ceil(extent / factor): banks have uniform depth, matching Vitis. When the
  // division is uneven the trailing slots of some banks are simply unused.
  int64_t extent = newShape[bankingDimension];
  newShape[bankingDimension] =
      (extent + static_cast<int64_t>(bankingFactor) - 1) /
      static_cast<int64_t>(bankingFactor);
  return MemRefType::get(newShape, originalType.getElementType(),
                         originalType.getLayout(),
                         originalType.getMemorySpace());
}

// Decodes a flat row-major index into an n-dimensional index.
static SmallVector<int64_t> decodeIndex(int64_t linIndex,
                                        ArrayRef<int64_t> shape) {
  const unsigned rank = shape.size();
  SmallVector<int64_t> ndIndex(rank, 0);
  for (int64_t d = rank - 1; d >= 0; --d) {
    ndIndex[d] = linIndex % shape[d];
    linIndex /= shape[d];
  }
  return ndIndex;
}

// Slice constant-init data into per-bank sub-blocks, honoring `kind`. Only used
// for constant-initialized (GetGlobal) partitioned arrays; verify the block
// ordering if you rely on it.
static SmallVector<SmallVector<Attribute>>
sliceSubBlock(ArrayRef<Attribute> allAttrs, ArrayRef<int64_t> memShape,
              ArrayRef<int64_t> newShape, unsigned bankingDimension,
              unsigned bankingFactor, PartitionSpec::Kind kind,
              Attribute zeroAttr) {
  size_t numElements = std::reduce(memShape.begin(), memShape.end(), size_t{1},
                                   std::multiplies<size_t>());
  size_t perBank = std::reduce(newShape.begin(), newShape.end(), size_t{1},
                               std::multiplies<size_t>());
  int64_t blockSize =
      (memShape[bankingDimension] + static_cast<int64_t>(bankingFactor) - 1) /
      static_cast<int64_t>(bankingFactor); // ceil

  // Zero-initialize every bank to ceil-sized depth, then place each element at
  // its banked coordinate. Uneven divisions leave the trailing slots of some
  // banks as zero; in-bounds accesses never address those offsets.
  SmallVector<SmallVector<Attribute>> subBlocks(
      bankingFactor, SmallVector<Attribute>(perBank, zeroAttr));

  for (unsigned linIndex = 0; linIndex < numElements; ++linIndex) {
    SmallVector<int64_t> ndIndex = decodeIndex(linIndex, memShape);
    int64_t d = ndIndex[bankingDimension];
    unsigned bank;
    int64_t offset;
    if (kind == PartitionSpec::Block) {
      bank = static_cast<unsigned>(d / blockSize);
      offset = d % blockSize;
    } else {
      bank = static_cast<unsigned>(d % bankingFactor);
      offset = d / static_cast<int64_t>(bankingFactor);
    }
    ndIndex[bankingDimension] = offset;
    // Row-major encode into the banked shape.
    int64_t newLin = 0;
    for (unsigned k = 0; k < newShape.size(); ++k)
      newLin = newLin * newShape[k] + ndIndex[k];
    subBlocks[bank][newLin] = allAttrs[linIndex];
  }
  return subBlocks;
}

static SmallVector<Value, 4>
handleGetGlobalOp(memref::GetGlobalOp getGlobalOp, uint64_t bankingFactor,
                  unsigned bankingDimension, PartitionSpec::Kind kind,
                  MemRefType newMemRefType, OpBuilder &builder,
                  DictionaryAttr remainingAttrs) {
  SmallVector<Value, 4> banks;
  auto memTy = cast<MemRefType>(getGlobalOp.getType());
  ArrayRef<int64_t> originalShape = memTy.getShape();
  auto newShape =
      SmallVector<int64_t>(originalShape.begin(), originalShape.end());
  newShape[bankingDimension] =
      (originalShape[bankingDimension] + static_cast<int64_t>(bankingFactor) -
       1) /
      static_cast<int64_t>(bankingFactor); // ceil

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

  Attribute zeroAttr = builder.getZeroAttr(globalOpTy.getElementType());
  auto subBlocks = sliceSubBlock(allAttrs, originalShape, newShape,
                                 bankingDimension, bankingFactor, kind,
                                 zeroAttr);

  builder.setInsertionPointAfter(globalOp);
  OpBuilder::InsertPoint globalOpsInsertPt = builder.saveInsertionPoint();
  builder.setInsertionPointAfter(getGlobalOp);
  OpBuilder::InsertPoint getGlobalOpsInsertPt = builder.saveInsertionPoint();

  for (size_t bankCnt = 0; bankCnt < bankingFactor; ++bankCnt) {
    auto newMemRefTy = MemRefType::get(newShape, globalOpTy.getElementType());
    auto newTypeAttr = TypeAttr::get(newMemRefTy);
    std::string newName = llvm::formatv(
        "{0}_{1}_{2}", globalOpNameAttr.getValue(), "bank", bankCnt);
    RankedTensorType tensorType =
        RankedTensorType::get({newShape}, globalOpTy.getElementType());
    auto newInitValue = DenseElementsAttr::get(tensorType, subBlocks[bankCnt]);

    builder.restoreInsertionPoint(globalOpsInsertPt);
    auto newGlobalOp = memref::GlobalOp::create(
        builder, globalOp.getLoc(), builder.getStringAttr(newName),
        globalOp.getSymVisibilityAttr(), newTypeAttr, newInitValue,
        globalOp.getConstantAttr(), globalOp.getAlignmentAttr());
    builder.setInsertionPointAfter(newGlobalOp);
    globalOpsInsertPt = builder.saveInsertionPoint();

    builder.restoreInsertionPoint(getGlobalOpsInsertPt);
    auto newGetGlobalOp = memref::GetGlobalOp::create(
        builder, getGlobalOp.getLoc(), newMemRefTy, newGlobalOp.getName());
    newGetGlobalOp->setAttrs(remainingAttrs);
    builder.setInsertionPointAfter(newGetGlobalOp);
    getGlobalOpsInsertPt = builder.saveInsertionPoint();

    banks.push_back(newGetGlobalOp);
  }

  globalOp.erase();
  return banks;
}

static void updateFuncOpArgumentTypes(func::FuncOp funcOp, unsigned argIndex,
                                      MemRefType newMemRefType,
                                      unsigned numInsertedArgs) {
  auto originalArgTypes = funcOp.getFunctionType().getInputs();
  SmallVector<Type, 4> updatedArgTypes;
  for (unsigned i = 0; i < originalArgTypes.size(); ++i) {
    updatedArgTypes.push_back(originalArgTypes[i]);
    if (i == argIndex)
      for (unsigned j = 0; j < numInsertedArgs; ++j)
        updatedArgTypes.push_back(newMemRefType);
  }
  auto resultTypes = funcOp.getFunctionType().getResults();
  funcOp.setType(
      FunctionType::get(funcOp.getContext(), updatedArgTypes, resultTypes));
}

static void updateFuncOpArgAttrs(func::FuncOp funcOp, unsigned argIndex,
                                 unsigned numInsertedArgs,
                                 DictionaryAttr remainingAttrs) {
  ArrayAttr existingArgAttrs = funcOp->getAttrOfType<ArrayAttr>("arg_attrs");
  SmallVector<Attribute, 4> updatedArgAttrs;
  unsigned numArguments = funcOp.getNumArguments();
  unsigned newNumArguments = numArguments + numInsertedArgs;
  updatedArgAttrs.resize(newNumArguments);

  for (unsigned i = 0; i < numArguments; ++i) {
    unsigned newIndex = (i > argIndex) ? i + numInsertedArgs : i;
    updatedArgAttrs[newIndex] = existingArgAttrs
                                    ? existingArgAttrs[i]
                                    : DictionaryAttr::get(funcOp.getContext());
  }
  for (unsigned i = 0; i < numInsertedArgs; ++i)
    updatedArgAttrs[argIndex + 1 + i] = remainingAttrs;

  funcOp->setAttr("arg_attrs",
                  ArrayAttr::get(funcOp.getContext(), updatedArgAttrs));
}

SmallVector<Value, 4> MemoryBankingPass::createBanks(OpBuilder &builder,
                                                     Value originalMem) {

  SmallVector<Value, 4> banks;
  MemRefType originalMemRefType = cast<MemRefType>(originalMem.getType());
  MLIRContext *context = builder.getContext();

  PartitionSpec spec;
  auto iface = dyn_cast_or_null<MemoryResourceOpInterface>(originalMem.getDefiningOp());
  if (!iface)
    return banks;
  auto specOpt = iface.getPartitionSpec();
  if (!specOpt.has_value())
    return banks;
  spec = *specOpt;
    
  unsigned currFactor = effectiveFactor(spec, originalMemRefType);
  unsigned currDimension = spec.dim;
  verifyBankingConfigurations(currFactor, currDimension, originalMemRefType);

  // One spec per array: the new banks carry no partition attribute.
  DictionaryAttr emptyAttrs = DictionaryAttr::get(context);

  MemRefType newMemRefType =
      computeBankedMemRefType(originalMemRefType, currFactor, currDimension);


  if (auto blockArgMem = dyn_cast<BlockArgument>(originalMem)) {
    Block *block = blockArgMem.getOwner();
    unsigned blockArgNum = blockArgMem.getArgNumber();

    for (unsigned i = 0; i < currFactor; ++i)
      block->insertArgument(blockArgNum + 1 + i, newMemRefType,
                            blockArgMem.getLoc());

    auto blockArgs = block->getArguments().slice(blockArgNum + 1, currFactor);
    banks.append(blockArgs.begin(), blockArgs.end());

    auto funcOp = dyn_cast<func::FuncOp>(block->getParentOp());
    assert(funcOp && "BlockArgument is not part of a FuncOp");
    updateFuncOpArgumentTypes(funcOp, blockArgNum, newMemRefType, currFactor);
    updateFuncOpArgAttrs(funcOp, blockArgNum, currFactor, emptyAttrs);
  } else {
    Operation *originalDef = originalMem.getDefiningOp();
    Location loc = originalDef->getLoc();
    builder.setInsertionPointAfter(originalDef);
    TypeSwitch<Operation *>(originalDef)
        .Case<memref::AllocOp>([&](memref::AllocOp) {
          for (uint64_t b = 0; b < currFactor; ++b) {
            auto bankAllocOp =
                memref::AllocOp::create(builder, loc, newMemRefType);
            bankAllocOp->setAttrs(emptyAttrs);
            banks.push_back(bankAllocOp);
          }
        })
        .Case<memref::AllocaOp>([&](memref::AllocaOp) {
          for (uint64_t b = 0; b < currFactor; ++b) {
            auto bankAllocaOp =
                memref::AllocaOp::create(builder, loc, newMemRefType);
            bankAllocaOp->setAttrs(emptyAttrs);
            banks.push_back(bankAllocaOp);
          }
        })
        .Case<memref::GetGlobalOp>([&](memref::GetGlobalOp getGlobalOp) {
          auto newBanks =
              handleGetGlobalOp(getGlobalOp, currFactor, currDimension,
                                spec.kind, newMemRefType, builder, emptyAttrs);
          banks.append(newBanks.begin(), newBanks.end());
        })
        .Default([](Operation *) {
          llvm_unreachable("Unhandled memory operation type");
        });
  }
  return banks;
}

//===----------------------------------------------------------------------===//
// Load / store rewriting
//===----------------------------------------------------------------------===//
// ---- loads -----------------------------------------------------------------
struct BankMemRefLoadPattern : OpRewritePattern<memref::LoadOp> {
  BankMemRefLoadPattern(MLIRContext *ctx,
                        DenseMap<Value, SmallVector<Value>> &memoryToBanks,
                        DenseSet<Value> &oldMemRefVals)
      : OpRewritePattern<memref::LoadOp>(ctx),
        memoryToBanks(memoryToBanks), oldMemRefVals(oldMemRefVals) {}

  LogicalResult matchAndRewrite(memref::LoadOp loadOp,
                                PatternRewriter &rewriter) const override {
    Value mem = loadOp.getMemRef();
    auto iface = dyn_cast_or_null<MemoryResourceOpInterface>(mem.getDefiningOp());
    if (!iface) return failure();
    auto specOpt = iface.getPartitionSpec();
    if (!specOpt) return failure();
    PartitionSpec spec = *specOpt;
    auto banksIt = memoryToBanks.find(mem);
    if (banksIt == memoryToBanks.end()) return failure();
    ArrayRef<Value> banks = banksIt->second;
    MemRefType origTy = loadOp.getMemRefType();
    unsigned factor = effectiveFactor(spec, origTy);
    verifyBankingConfigurations(factor, spec.dim, origTy);

    Location loc = loadOp.getLoc();
    rewriter.setInsertionPoint(loadOp);
    MemrefBankAccess ba =
        computeMemrefBankAccess(rewriter, loc, loadOp.getIndices(), spec, origTy);

    if (ba.constBank) {
      rewriter.replaceOpWithNewOp<memref::LoadOp>(
          loadOp, banks[*ba.constBank], ba.bankedIndices);
      if (isa<BlockArgument>(mem)) oldMemRefVals.insert(mem);
      return success();
    }

    SmallVector<Value, 4> bankVals;
    for (unsigned k = 0; k < factor; ++k)
      bankVals.push_back(
          memref::LoadOp::create(rewriter, loc, banks[k], ba.bankedIndices));

    Value result = bankVals[factor - 1];
    for (unsigned k = factor - 1; k-- > 0;) {
      Value kc = arith::ConstantIndexOp::create(rewriter, loc, k);
      Value eq = arith::CmpIOp::create(rewriter, loc,
                                       arith::CmpIPredicate::eq, ba.bankIdx, kc);
      result = arith::SelectOp::create(rewriter, loc, eq, bankVals[k], result);
    }
    if (isa<BlockArgument>(mem)) oldMemRefVals.insert(mem);
    rewriter.replaceOp(loadOp, result);
    return success();
  }

private:
  DenseMap<Value, SmallVector<Value>> &memoryToBanks;
  DenseSet<Value> &oldMemRefVals;
};

struct BankAffineLoadPattern
    : public OpRewritePattern<mlir::affine::AffineLoadOp> {
  BankAffineLoadPattern(MLIRContext *context,
                        DenseMap<Value, SmallVector<Value>> &memoryToBanks,
                        DenseSet<Value> &oldMemRefVals)
      : OpRewritePattern<mlir::affine::AffineLoadOp>(context),
        memoryToBanks(memoryToBanks), oldMemRefVals(oldMemRefVals) {}
 
  LogicalResult matchAndRewrite(mlir::affine::AffineLoadOp loadOp,
                                PatternRewriter &rewriter) const override {
    Value mem = loadOp.getMemref();
    auto iface = dyn_cast_or_null<MemoryResourceOpInterface>(mem.getDefiningOp());
    if (!iface)
      return failure();
    auto specOpt = iface.getPartitionSpec();
    if (!specOpt.has_value())
      return failure();
    PartitionSpec spec = *specOpt;
    
    auto banksIt = memoryToBanks.find(mem);
    if (banksIt == memoryToBanks.end())
      return failure();
    ArrayRef<Value> banks = banksIt->second;
    MemRefType origTy = loadOp.getMemRefType();
    unsigned f = effectiveFactor(spec, origTy);
    verifyBankingConfigurations(f, spec.dim, origTy);
 
    Location loc = loadOp.getLoc();
    AffineMap accessMap = loadOp.getAffineMap();
    SmallVector<Value,4> operands(loadOp.getMapOperands().begin(),
                                  loadOp.getMapOperands().end());
    affine::fullyComposeAffineMapAndOperands(&accessMap, &operands);
    accessMap = simplifyAffineMap(accessMap);
    BankAccess ba = computeBankAccess(accessMap, spec, origTy,
                                      rewriter.getContext());
 
    // (1) constant bank -> single direct load.
    if (ba.constBank) {
      auto newLoad = rewriter.replaceOpWithNewOp<mlir::affine::AffineLoadOp>(
          loadOp, banks[*ba.constBank], ba.bankedAccessMap, operands);
      (void)newLoad;
      if (isa<BlockArgument>(mem))
        oldMemRefVals.insert(mem);
      return success();
    }
 
    // (2) affine rotating bank -> read every bank at the shared offset, mux
    //     with a select chain on the (runtime) bank index. No switch.
    Value bankIdx =
        affine::AffineApplyOp::create(rewriter, loc, ba.bankMap, operands);
 
    SmallVector<Value, 4> bankVals;
    for (unsigned k = 0; k < f; ++k)
      bankVals.push_back(mlir::affine::AffineLoadOp::create(
          rewriter, loc, banks[k], ba.bankedAccessMap, operands));
 
    // result = (idx==0)?v0 : (idx==1)?v1 : ... : v_{f-1}
    Value result = bankVals[f - 1];
    for (unsigned k = f - 1; k-- > 0;) {
      Value kc = arith::ConstantIndexOp::create(rewriter, loc, k);
      Value eq = arith::CmpIOp::create(rewriter, loc,
                                       arith::CmpIPredicate::eq, bankIdx, kc);
      result = arith::SelectOp::create(rewriter, loc, eq, bankVals[k], result);
    }
 
    if (isa<BlockArgument>(mem))
      oldMemRefVals.insert(mem);
    rewriter.replaceOp(loadOp, result);
    return success();
  }
 
private:
  DenseMap<Value, SmallVector<Value>> &memoryToBanks;
  DenseSet<Value> &oldMemRefVals;
};

// ---- stores ----------------------------------------------------------------


struct BankMemRefStorePattern : public OpRewritePattern<memref::StoreOp> {
  BankMemRefStorePattern(MLIRContext *ctx,
                         DenseMap<Value, SmallVector<Value>> &memoryToBanks,
                         DenseSet<Operation *> &opsToErase,
                         DenseSet<Operation *> &processedOps,
                         DenseSet<Value> &oldMemRefVals)
      : OpRewritePattern<memref::StoreOp>(ctx),
        memoryToBanks(memoryToBanks), opsToErase(opsToErase),
        processedOps(processedOps), oldMemRefVals(oldMemRefVals) {}

  LogicalResult matchAndRewrite(memref::StoreOp storeOp,
                                PatternRewriter &rewriter) const override {
    if (processedOps.contains(storeOp)) return failure();
    Value mem = storeOp.getMemRef();
    auto iface = dyn_cast_or_null<MemoryResourceOpInterface>(mem.getDefiningOp());
    if (!iface) return failure();
    auto specOpt = iface.getPartitionSpec();
    if (!specOpt) return failure();
    PartitionSpec spec = *specOpt;
    auto banksIt = memoryToBanks.find(mem);
    if (banksIt == memoryToBanks.end()) return failure();
    ArrayRef<Value> banks = banksIt->second;
    MemRefType origTy = storeOp.getMemRefType();
    unsigned f = effectiveFactor(spec, origTy);
    verifyBankingConfigurations(f, spec.dim, origTy);

    Location loc = storeOp.getLoc();
    Value val = storeOp.getValueToStore();
    rewriter.setInsertionPoint(storeOp);
    MemrefBankAccess ba =
        computeMemrefBankAccess(rewriter, loc, storeOp.getIndices(), spec, origTy);

    if (ba.constBank) {
      memref::StoreOp::create(rewriter, loc, val, banks[*ba.constBank],
                              ba.bankedIndices);
    } else {
      for (unsigned k = 0; k < f; ++k) {
        Value kc = arith::ConstantIndexOp::create(rewriter, loc, k);
        Value en = arith::CmpIOp::create(rewriter, loc,
                                         arith::CmpIPredicate::eq, ba.bankIdx, kc);
        // Mirror AffineStoreEnableLowering's StoreEnableOp build order:
        // (value, enable_i1, memref, indices...)
        StoreEnableOp::create(rewriter, loc, val, en, banks[k],
                              ba.bankedIndices);
      }
    }
    processedOps.insert(storeOp);
    opsToErase.insert(storeOp);
    oldMemRefVals.insert(mem);
    return success();
  }

private:
  DenseMap<Value, SmallVector<Value>> &memoryToBanks;
  DenseSet<Operation *> &opsToErase;
  DenseSet<Operation *> &processedOps;
  DenseSet<Value> &oldMemRefVals;
};

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
    if (processedOps.contains(storeOp))
      return failure();
    
    Value mem = storeOp.getMemref();
    auto iface = dyn_cast_or_null<MemoryResourceOpInterface>(mem.getDefiningOp());
    if (!iface)
      return failure();
    auto specOpt = iface.getPartitionSpec();
    if (!specOpt.has_value())
      return failure();
    
    PartitionSpec spec = *specOpt;
    
    auto banksIt = memoryToBanks.find(mem);
    if (banksIt == memoryToBanks.end())
      return failure();
    
    ArrayRef<Value> banks = banksIt->second;
    MemRefType origTy = storeOp.getMemRefType();
    unsigned factor = effectiveFactor(spec, origTy);
    verifyBankingConfigurations(factor, spec.dim, origTy);
 
    Location loc = storeOp.getLoc();
    Value val = storeOp.getValueToStore();
    AffineMap accessMap = storeOp.getAffineMap();
    SmallVector<Value,4> operands(storeOp.getMapOperands().begin(),
                                  storeOp.getMapOperands().end());
    affine::fullyComposeAffineMapAndOperands(&accessMap, &operands);

    
    accessMap = simplifyAffineMap(accessMap);   // canonicalize after compose
    BankAccess ba = computeBankAccess(accessMap, spec, origTy,
                                      rewriter.getContext());
 
    rewriter.setInsertionPoint(storeOp);
 
    // (1) constant bank -> single direct store.
    if (ba.constBank) {
      affine::AffineStoreOp::create(rewriter, loc, val,
                                          banks[*ba.constBank],
                                          ba.bankedAccessMap, operands);
    } else {
      // (2) affine rotating bank -> affine predicated demux. Each bank gets ONE
      //     affine.store_enable carrying its guard as an IntegerSet and its
      //     address as the banked access map. Staying affine keeps the write
      //     visible to MemoryDependenceAnalysis (auto-anchored, loop-carried deps
      //     for accumulation), and lowers to the flat memref store_enable later.
      AffineExpr be = ba.bankMap.getResult(0);  // bank index expr = idx mod f
      unsigned nDims = ba.bankMap.getNumDims();
      unsigned nSyms = ba.bankMap.getNumSymbols();

      for (unsigned k = 0; k < factor; ++k) {
        // Guard set:  (be - k) == 0   over the SAME operands the maps use.
        IntegerSet cond = IntegerSet::get(nDims, nSyms,
                                          /*constraints=*/{be - k},
                                          /*eqFlags=*/{true});
        AffineStoreEnableOp::create(
            rewriter, loc,
            val,                       // $value
            banks[k],                  // $memref
            /*mapOperands=*/operands,  // feed ba.bankedAccessMap
            /*setOperands=*/operands,  // feed cond
            ba.bankedAccessMap,        // $map  (banked dim -> offset)
            cond);                     // $condition
      }
    }
 
    processedOps.insert(storeOp);
    opsToErase.insert(storeOp);
    oldMemRefVals.insert(mem);
    return success();
  }
 
private:
  DenseMap<Value, SmallVector<Value>> &memoryToBanks;
  DenseSet<Operation *> &opsToErase;
  DenseSet<Operation *> &processedOps;
  DenseSet<Value> &oldMemRefVals;
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

    func::FuncOp funcOp = returnOp->getParentOfType<func::FuncOp>();
    rewriter.setInsertionPointToEnd(&funcOp.getBlocks().front());
    auto newReturnOp =
        func::ReturnOp::create(rewriter, loc, ValueRange(newReturnOperands));
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

static LogicalResult cleanUpOldMemRefs(DenseSet<Value> &oldMemRefVals,
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

  for (auto *op : opsToErase)
    op->erase();

  for (auto &memrefVal : valuesToErase) {
    assert(memrefVal.use_empty() && "use must be empty");
    if (auto blockArg = dyn_cast<BlockArgument>(memrefVal))
      blockArg.getOwner()->eraseArgument(blockArg.getArgNumber());
    else if (auto *op = memrefVal.getDefiningOp())
      op->erase();
  }

  for (auto funcOp : funcsToModify) {
    ArrayAttr existingArgAttrs = funcOp->getAttrOfType<ArrayAttr>("arg_attrs");
    if (existingArgAttrs) {
      SmallVector<Attribute, 4> updatedArgAttrs;
      auto erasedIndices = erasedArgIndices[funcOp];
      DenseSet<unsigned> indicesToErase(erasedIndices.begin(),
                                        erasedIndices.end());
      for (unsigned i = 0; i < existingArgAttrs.size(); ++i)
        if (!indicesToErase.contains(i))
          updatedArgAttrs.push_back(existingArgAttrs[i]);
      funcOp->setAttr("arg_attrs",
                      ArrayAttr::get(funcOp.getContext(), updatedArgAttrs));
    }

    SmallVector<Type, 4> newArgTypes;
    for (BlockArgument arg : funcOp.getArguments())
      newArgTypes.push_back(arg.getType());
    funcOp.setType(FunctionType::get(funcOp.getContext(), newArgTypes,
                                     funcOp.getFunctionType().getResults()));
  }

  return success();
}

void MemoryBankingPass::runOnOperation() {
  func::FuncOp funcOp = getOperation();
  if (funcOp.isExternal())
    return;

  memoryToBanks.clear();
  oldMemRefVals.clear();
  opsToErase.clear();

  OpBuilder builder(funcOp);

  // Each partitioned array has a single spec, so one banking pass suffices:
  // create all banks first, then rewrite every affine load/store in one go.
  DenseSet<Value> memrefs = collectMemRefs(funcOp);
  for (Value memrefVal : memrefs) {
    auto iface =
      dyn_cast_or_null<MemoryResourceOpInterface>(memrefVal.getDefiningOp());
    if (!iface)
      continue;
    auto spec = iface.getPartitionSpec();
    if (!spec || (spec->factor == 1 && spec->kind != PartitionSpec::Complete))
      continue;
    auto [it, inserted] =
        memoryToBanks.insert(std::make_pair(memrefVal, SmallVector<Value>{}));
    if (inserted)
      it->second = createBanks(builder, memrefVal);
  }

  if (failed(applyMemoryBanking(funcOp, &getContext())))
    signalPassFailure();
}

LogicalResult MemoryBankingPass::applyMemoryBanking(Operation *operation,
                                                    MLIRContext *ctx) {
  RewritePatternSet patterns(ctx);
  DenseSet<Operation *> processedOps;
  patterns.add<BankAffineLoadPattern>(ctx, memoryToBanks, oldMemRefVals);
  patterns.add<BankAffineStorePattern>(ctx, memoryToBanks, opsToErase,
                                       processedOps, oldMemRefVals);
  patterns.add<BankReturnPattern>(ctx, memoryToBanks);
  patterns.add<BankMemRefLoadPattern>(ctx, memoryToBanks, oldMemRefVals);
  patterns.add<BankMemRefStorePattern>(ctx, memoryToBanks, opsToErase,
                                       processedOps, oldMemRefVals);

  GreedyRewriteConfig config;
  config.setStrictness(GreedyRewriteStrictness::ExistingOps);
  if (failed(applyPatternsGreedily(operation, std::move(patterns), config)))
    return failure();

  if (failed(cleanUpOldMemRefs(oldMemRefVals, opsToErase)))
    return failure();


  // Composition folded operand-defining affine.apply ops (e.g. 2*j) into the
  // banked access maps, leaving them dead. ExistingOps strictness means the
  // greedy driver won't reap them; do it here so they don't survive as
  // disconnected sinks for the downstream scheduler.
  {
    SmallVector<Operation *> dead;
    operation->walk([&](affine::AffineApplyOp ap) {
      if (isOpTriviallyDead(ap))
        dead.push_back(ap);
    });
    for (Operation *op : llvm::reverse(dead))
      op->erase();
  }
  
  return success();
}

namespace circt::hls_analysis {

static std::unique_ptr<mlir::Pass>
createMemoryBankingPassModified() {
  return std::make_unique<MemoryBankingPass>();
}

void registerMemoryBankingPassModified() {
  mlir::PassRegistration<MemoryBankingPass>();
}

} // namespace circt