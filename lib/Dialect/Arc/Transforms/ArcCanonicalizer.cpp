//===- ArcCanonicalizer.cpp -------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//===----------------------------------------------------------------------===//
//
// Simulation centric canonicalizations for non-arc operations and
// canonicalizations that require efficient symbol lookups.
//
//===----------------------------------------------------------------------===//

#include "PassDetails.h"
#include "circt/Dialect/Arc/ArcOps.h"
#include "circt/Dialect/Comb/CombOps.h"
#include "circt/Dialect/HW/HWOps.h"
#include "circt/Support/Namespace.h"
#include "circt/Support/SymCache.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "arc-canonicalizer"

using namespace circt;
using namespace arc;

//===----------------------------------------------------------------------===//
// Datastructures
//===----------------------------------------------------------------------===//

/// A combination of SymbolCache and SymbolUserMap that also allows to add users
/// and remove symbols on-demand.
class SymbolHandler : public SymbolCache {
public:
  /// Return the users of the provided symbol operation.
  ArrayRef<Operation *> getUsers(Operation *symbol) const {
    auto it = userMap.find(symbol);
    return it != userMap.end() ? it->second.getArrayRef() : std::nullopt;
  }

  /// Return true if the given symbol has no uses.
  bool useEmpty(Operation *symbol) {
    return !userMap.count(symbol) || userMap[symbol].empty();
  }

  void addUser(Operation *def, Operation *user) {
    assert(isa<mlir::SymbolOpInterface>(def));
    if (!symbolCache.contains(cast<mlir::SymbolOpInterface>(def).getNameAttr()))
      symbolCache.insert(
          {cast<mlir::SymbolOpInterface>(def).getNameAttr(), def});
    userMap[def].insert(user);
  }

  void removeUser(Operation *def, Operation *user) {
    assert(isa<mlir::SymbolOpInterface>(def));
    if (symbolCache.contains(cast<mlir::SymbolOpInterface>(def).getNameAttr()))
      userMap[def].remove(user);
    if (userMap[def].empty())
      userMap.erase(def);
  }

  void removeDefinitionAndAllUsers(Operation *def) {
    assert(isa<mlir::SymbolOpInterface>(def));
    symbolCache.erase(cast<mlir::SymbolOpInterface>(def).getNameAttr());
    userMap.erase(def);
  }

  void collectAllSymbolUses(Operation *symbolTableOp,
                            SymbolTableCollection &symbolTable) {
    // NOTE: the following is almost 1-1 taken from the SymbolUserMap
    // constructor. They made it difficult to extend the implementation by
    // having a lot of members private and non-virtual methods.
    SmallVector<Operation *> symbols;
    auto walkFn = [&](Operation *symbolTableOp, bool allUsesVisible) {
      for (Operation &nestedOp : symbolTableOp->getRegion(0).getOps()) {
        auto symbolUses = SymbolTable::getSymbolUses(&nestedOp);
        assert(symbolUses && "expected uses to be valid");

        for (const SymbolTable::SymbolUse &use : *symbolUses) {
          symbols.clear();
          (void)symbolTable.lookupSymbolIn(symbolTableOp, use.getSymbolRef(),
                                           symbols);
          for (Operation *symbolOp : symbols)
            userMap[symbolOp].insert(use.getUser());
        }
      }
    };
    // We just set `allSymUsesVisible` to false here because it isn't necessary
    // for building the user map.
    SymbolTable::walkSymbolTables(symbolTableOp, /*allSymUsesVisible=*/false,
                                  walkFn);
  }

private:
  DenseMap<Operation *, SetVector<Operation *>> userMap;
};

//===----------------------------------------------------------------------===//
// Canonicalization patterns
//===----------------------------------------------------------------------===//

namespace {
/// A rewrite pattern that has access to a symbol cache to access and modify the
/// symbol-defining op and symbol users as well as a namespace to query new
/// names. Each pattern has to make sure that the symbol handler is kept
/// up-to-date no matter whether the pattern succeeds of fails.
template <typename SourceOp>
class SymOpRewritePattern : public OpRewritePattern<SourceOp> {
public:
  SymOpRewritePattern(MLIRContext *ctxt, SymbolHandler &symbolCache,
                      Namespace &names, mlir::PatternBenefit benefit = 1,
                      ArrayRef<StringRef> generatedNames = {})
      : OpRewritePattern<SourceOp>(ctxt, benefit, generatedNames), names(names),
        symbolCache(symbolCache) {}

protected:
  Namespace &names;
  SymbolHandler &symbolCache;
};

class MemWritePortEnableAndMaskCanonicalizer
    : public SymOpRewritePattern<MemoryWritePortOp> {
public:
  MemWritePortEnableAndMaskCanonicalizer(
      MLIRContext *ctxt, SymbolHandler &symbolCache, Namespace &names,
      DenseMap<StringAttr, StringAttr> &arcMapping)
      : SymOpRewritePattern<MemoryWritePortOp>(ctxt, symbolCache, names),
        arcMapping(arcMapping) {}
  LogicalResult matchAndRewrite(MemoryWritePortOp op,
                                PatternRewriter &rewriter) const final;

private:
  DenseMap<StringAttr, StringAttr> &arcMapping;
};

struct CallPassthroughArc : public SymOpRewritePattern<CallOp> {
  using SymOpRewritePattern::SymOpRewritePattern;
  LogicalResult matchAndRewrite(CallOp op,
                                PatternRewriter &rewriter) const final;
};

struct StatePassthroughArc : public SymOpRewritePattern<StateOp> {
  using SymOpRewritePattern::SymOpRewritePattern;
  LogicalResult matchAndRewrite(StateOp op,
                                PatternRewriter &rewriter) const final;
};

struct RemoveUnusedArcs : public SymOpRewritePattern<DefineOp> {
  using SymOpRewritePattern::SymOpRewritePattern;
  LogicalResult matchAndRewrite(DefineOp op,
                                PatternRewriter &rewriter) const final;
};

struct ICMPCanonicalizer : public OpRewritePattern<comb::ICmpOp> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(comb::ICmpOp op,
                                PatternRewriter &rewriter) const final;
};

} // namespace

//===----------------------------------------------------------------------===//
// Helpers
//===----------------------------------------------------------------------===//

LogicalResult canonicalizePassthoughCall(mlir::CallOpInterface callOp,
                                         SymbolHandler &symbolCache,
                                         PatternRewriter &rewriter) {
  auto defOp = cast<DefineOp>(symbolCache.getDefinition(
      callOp.getCallableForCallee().get<SymbolRefAttr>().getLeafReference()));
  if (defOp.isPassthrough()) {
    symbolCache.removeUser(defOp, callOp);
    rewriter.replaceOp(callOp, callOp.getArgOperands());
    return success();
  }
  return failure();
}

//===----------------------------------------------------------------------===//
// Canonicalization pattern implementations
//===----------------------------------------------------------------------===//

LogicalResult MemWritePortEnableAndMaskCanonicalizer::matchAndRewrite(
    MemoryWritePortOp op, PatternRewriter &rewriter) const {
  auto defOp = cast<DefineOp>(symbolCache.getDefinition(op.getArcAttr()));
  APInt enable;

  if (op.getEnable() &&
      mlir::matchPattern(
          defOp.getBodyBlock().getTerminator()->getOperand(op.getEnableIdx()),
          mlir::m_ConstantInt(&enable))) {
    if (enable.isZero()) {
      symbolCache.removeUser(defOp, op);
      rewriter.eraseOp(op);
      if (symbolCache.useEmpty(defOp)) {
        symbolCache.removeDefinitionAndAllUsers(defOp);
        rewriter.eraseOp(defOp);
      }
      return success();
    }
    if (enable.isAllOnes()) {
      if (arcMapping.count(defOp.getNameAttr())) {
        auto arcWithoutEnable = arcMapping[defOp.getNameAttr()];
        // Remove the enable attribute
        rewriter.updateRootInPlace(op, [&]() {
          op.setEnable(false);
          op.setArc(arcWithoutEnable.getValue());
        });
        symbolCache.removeUser(defOp, op);
        symbolCache.addUser(symbolCache.getDefinition(arcWithoutEnable), op);
        return success();
      }

      auto newName = names.newName(defOp.getName());
      auto users = SmallVector<Operation *>(symbolCache.getUsers(defOp));
      symbolCache.removeDefinitionAndAllUsers(defOp);

      // Remove the enable attribute
      rewriter.updateRootInPlace(op, [&]() {
        op.setEnable(false);
        op.setArc(newName);
      });

      auto newResultTypes = op.getArcResultTypes();

      // Create a new arc that acts as replacement for other users
      rewriter.setInsertionPoint(defOp);
      auto newDefOp = rewriter.cloneWithoutRegions(defOp);
      auto *block = rewriter.createBlock(
          &newDefOp.getBody(), newDefOp.getBody().end(),
          newDefOp.getArgumentTypes(),
          SmallVector<Location>(newDefOp.getNumArguments(), defOp.getLoc()));
      auto callOp = rewriter.create<CallOp>(newDefOp.getLoc(), newResultTypes,
                                            newName, block->getArguments());
      SmallVector<Value> results(callOp->getResults());
      Value constTrue = rewriter.create<hw::ConstantOp>(
          newDefOp.getLoc(), rewriter.getI1Type(), 1);
      results.insert(results.begin() + op.getEnableIdx(), constTrue);
      rewriter.create<OutputOp>(newDefOp.getLoc(), results);

      // Remove the enable output from the current arc
      auto *terminator = defOp.getBodyBlock().getTerminator();
      rewriter.updateRootInPlace(
          terminator, [&]() { terminator->eraseOperand(op.getEnableIdx()); });
      rewriter.updateRootInPlace(defOp, [&]() {
        defOp.setName(newName);
        defOp.setFunctionType(
            rewriter.getFunctionType(defOp.getArgumentTypes(), newResultTypes));
      });

      // Update symbol cache
      symbolCache.addDefinition(defOp.getNameAttr(), defOp);
      symbolCache.addDefinition(newDefOp.getNameAttr(), newDefOp);
      symbolCache.addUser(defOp, callOp);
      for (auto *user : users)
        symbolCache.addUser(user == op ? defOp : newDefOp, user);

      arcMapping[newDefOp.getNameAttr()] = defOp.getNameAttr();
      return success();
    }
  }
  return failure();
}

LogicalResult
CallPassthroughArc::matchAndRewrite(CallOp op,
                                    PatternRewriter &rewriter) const {
  return canonicalizePassthoughCall(op, symbolCache, rewriter);
}

LogicalResult
StatePassthroughArc::matchAndRewrite(StateOp op,
                                     PatternRewriter &rewriter) const {
  if (op.getLatency() == 0)
    return canonicalizePassthoughCall(op, symbolCache, rewriter);
  return failure();
}

LogicalResult
RemoveUnusedArcs::matchAndRewrite(DefineOp op,
                                  PatternRewriter &rewriter) const {
  if (symbolCache.useEmpty(op)) {
    op.getBody().walk([&](mlir::CallOpInterface user) {
      if (auto symbol = user.getCallableForCallee().dyn_cast<SymbolRefAttr>())
        if (auto *defOp = symbolCache.getDefinition(symbol.getLeafReference()))
          symbolCache.removeUser(defOp, user);
    });
    symbolCache.removeDefinitionAndAllUsers(op);
    rewriter.eraseOp(op);
    return success();
  }
  return failure();
}

LogicalResult
ICMPCanonicalizer::matchAndRewrite(comb::ICmpOp op,
                                   PatternRewriter &rewriter) const {
  auto getConstant = [&](const APInt &constant) -> Value {
    return rewriter.create<hw::ConstantOp>(op.getLoc(), constant);
  };
  auto sameWidthIntegers = [](TypeRange types) -> std::optional<unsigned> {
    if (llvm::all_equal(types) && !types.empty())
      if (auto intType = dyn_cast<IntegerType>(*types.begin()))
        return intType.getWidth();
    return std::nullopt;
  };

  APInt rhs;
  if (matchPattern(op.getRhs(), mlir::m_ConstantInt(&rhs))) {
    if (auto concatOp = op.getLhs().getDefiningOp<comb::ConcatOp>()) {
      if (auto optionalWidth =
              sameWidthIntegers(concatOp->getOperands().getTypes())) {
        if (op.getPredicate() == comb::ICmpPredicate::eq && rhs.isAllOnes()) {
          auto andOp = rewriter.create<comb::AndOp>(
              op.getLoc(), concatOp.getInputs(), op.getTwoState());
          if (*optionalWidth == 1)
            return rewriter.replaceOp(op, andOp->getResults()), success();
          rewriter.replaceOpWithNewOp<comb::ICmpOp>(
              op, comb::ICmpPredicate::eq, andOp,
              getConstant(APInt::getAllOnes(*optionalWidth)), op.getTwoState());
          return success();
        }

        if (op.getPredicate() == comb::ICmpPredicate::ne && rhs.isZero()) {
          auto orOp = rewriter.create<comb::OrOp>(
              op.getLoc(), concatOp.getInputs(), op.getTwoState());
          if (*optionalWidth == 1)
            return rewriter.replaceOp(op, orOp->getResults()), success();
          rewriter.replaceOpWithNewOp<comb::ICmpOp>(
              op, comb::ICmpPredicate::ne, orOp,
              getConstant(APInt::getZero(*optionalWidth)), op.getTwoState());
          return success();
        }
      }
    }
  }
  return failure();
}

//===----------------------------------------------------------------------===//
// ArcCanonicalizerPass implementation
//===----------------------------------------------------------------------===//

namespace {
struct ArcCanonicalizerPass
    : public ArcCanonicalizerBase<ArcCanonicalizerPass> {
  void runOnOperation() override;
};
} // namespace

void ArcCanonicalizerPass::runOnOperation() {
  MLIRContext &ctxt = getContext();
  SymbolTableCollection symbolTable;
  SymbolHandler cache;
  cache.addDefinitions(getOperation());
  cache.collectAllSymbolUses(getOperation(), symbolTable);
  Namespace names;
  names.add(cache);
  DenseMap<StringAttr, StringAttr> arcMapping;

  mlir::GreedyRewriteConfig config;
  config.enableRegionSimplification = false;
  config.maxIterations = 10;
  config.useTopDownTraversal = true;

  RewritePatternSet symbolPatterns(&getContext());
  symbolPatterns.add<CallPassthroughArc, StatePassthroughArc, RemoveUnusedArcs>(
      &getContext(), cache, names);
  symbolPatterns.add<MemWritePortEnableAndMaskCanonicalizer>(
      &getContext(), cache, names, arcMapping);

  if (failed(mlir::applyPatternsAndFoldGreedily(
          getOperation(), std::move(symbolPatterns), config)))
    return signalPassFailure();

  RewritePatternSet patterns(&ctxt);
  for (auto *dialect : ctxt.getLoadedDialects())
    dialect->getCanonicalizationPatterns(patterns);
  for (mlir::RegisteredOperationName op : ctxt.getRegisteredOperations())
    op.getCanonicalizationPatterns(patterns, &ctxt);
  patterns.add<ICMPCanonicalizer>(&getContext());

  // Don't test for convergence since it is often not reached.
  (void)mlir::applyPatternsAndFoldGreedily(getOperation(), std::move(patterns),
                                           config);
}

std::unique_ptr<mlir::Pass> arc::createArcCanonicalizerPass() {
  return std::make_unique<ArcCanonicalizerPass>();
}
