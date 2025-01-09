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

#include "circt/Dialect/Arc/ArcOps.h"
#include "circt/Dialect/Arc/ArcPasses.h"
#include "circt/Dialect/Comb/CombOps.h"
#include "circt/Dialect/HW/HWOps.h"
#include "circt/Dialect/Seq/SeqOps.h"
#include "circt/Support/Namespace.h"
#include "circt/Support/SymCache.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "arc-canonicalizer"

namespace circt {
namespace arc {
#define GEN_PASS_DEF_ARCCANONICALIZER
#include "circt/Dialect/Arc/ArcPasses.h.inc"
} // namespace arc
} // namespace circt

using namespace circt;
using namespace arc;

//===----------------------------------------------------------------------===//
// Datastructures
//===----------------------------------------------------------------------===//

namespace {

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

/// A Listener keeping the provided SymbolHandler up-to-date. This is especially
/// important for simplifications (e.g. DCE) the rewriter performs automatically
/// that we cannot or do not want to turn off.
class ArcListener : public mlir::RewriterBase::Listener {
public:
  explicit ArcListener(SymbolHandler *handler) : Listener(), handler(handler) {}

  void notifyOperationReplaced(Operation *op, Operation *replacement) override {
    // If, e.g., a DefineOp is replaced with another DefineOp but with the same
    // symbol, we don't want to drop the list of users.
    auto symOp = dyn_cast<mlir::SymbolOpInterface>(op);
    auto symReplacement = dyn_cast<mlir::SymbolOpInterface>(replacement);
    if (symOp && symReplacement &&
        symOp.getNameAttr() == symReplacement.getNameAttr())
      return;

    remove(op);
    // TODO: if an operation is inserted that defines a symbol and the symbol
    // already has uses, those users are not added.
    add(replacement);
  }

  void notifyOperationReplaced(Operation *op, ValueRange replacement) override {
    remove(op);
  }

  void notifyOperationErased(Operation *op) override { remove(op); }

  void notifyOperationInserted(Operation *op,
                               mlir::IRRewriter::InsertPoint) override {
    // TODO: if an operation is inserted that defines a symbol and the symbol
    // already has uses, those users are not added.
    add(op);
  }

private:
  FailureOr<Operation *> maybeGetDefinition(Operation *op) {
    if (auto callOp = dyn_cast<mlir::CallOpInterface>(op)) {
      auto symAttr =
          dyn_cast<mlir::SymbolRefAttr>(callOp.getCallableForCallee());
      if (!symAttr)
        return failure();
      if (auto *def = handler->getDefinition(symAttr.getLeafReference()))
        return def;
    }
    return failure();
  }

  void remove(Operation *op) {
    auto maybeDef = maybeGetDefinition(op);
    if (!failed(maybeDef))
      handler->removeUser(*maybeDef, op);

    if (isa<mlir::SymbolOpInterface>(op))
      handler->removeDefinitionAndAllUsers(op);
  }

  void add(Operation *op) {
    auto maybeDef = maybeGetDefinition(op);
    if (!failed(maybeDef))
      handler->addUser(*maybeDef, op);

    if (auto defOp = dyn_cast<mlir::SymbolOpInterface>(op))
      handler->addDefinition(defOp.getNameAttr(), op);
  }

  SymbolHandler *handler;
};

struct PatternStatistics {
  unsigned removeUnusedArcArgumentsPatternNumArgsRemoved = 0;
};

} // namespace

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
                      Namespace &names, PatternStatistics &stats,
                      mlir::PatternBenefit benefit = 1,
                      ArrayRef<StringRef> generatedNames = {})
      : OpRewritePattern<SourceOp>(ctxt, benefit, generatedNames), names(names),
        symbolCache(symbolCache), statistics(stats) {}

protected:
  Namespace &names;
  SymbolHandler &symbolCache;
  PatternStatistics &statistics;
};

class MemWritePortEnableAndMaskCanonicalizer
    : public SymOpRewritePattern<MemoryWritePortOp> {
public:
  MemWritePortEnableAndMaskCanonicalizer(
      MLIRContext *ctxt, SymbolHandler &symbolCache, Namespace &names,
      PatternStatistics &stats, DenseMap<StringAttr, StringAttr> &arcMapping)
      : SymOpRewritePattern<MemoryWritePortOp>(ctxt, symbolCache, names, stats),
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

struct CompRegCanonicalizer : public OpRewritePattern<seq::CompRegOp> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(seq::CompRegOp op,
                                PatternRewriter &rewriter) const final;
};

struct RemoveUnusedArcArgumentsPattern : public SymOpRewritePattern<DefineOp> {
  using SymOpRewritePattern::SymOpRewritePattern;
  LogicalResult matchAndRewrite(DefineOp op,
                                PatternRewriter &rewriter) const final;
};

struct SinkArcInputsPattern : public SymOpRewritePattern<DefineOp> {
  using SymOpRewritePattern::SymOpRewritePattern;
  LogicalResult matchAndRewrite(DefineOp op,
                                PatternRewriter &rewriter) const final;
};

struct MergeVectorizeOps : public OpRewritePattern<VectorizeOp> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(VectorizeOp op,
                                PatternRewriter &rewriter) const final;
};

struct KeepOneVecOp : public OpRewritePattern<VectorizeOp> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(VectorizeOp op,
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

LogicalResult updateInputOperands(VectorizeOp &vecOp,
                                  const SmallVector<Value> &newOperands) {
  // Set the new inputOperandSegments value
  unsigned groupSize = vecOp.getResults().size();
  unsigned numOfGroups = newOperands.size() / groupSize;
  SmallVector<int32_t> newAttr(numOfGroups, groupSize);
  vecOp.setInputOperandSegments(newAttr);
  vecOp.getOperation()->setOperands(ValueRange(newOperands));
  return success();
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
        rewriter.modifyOpInPlace(op, [&]() {
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
      rewriter.modifyOpInPlace(op, [&]() {
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
      rewriter.modifyOpInPlace(
          terminator, [&]() { terminator->eraseOperand(op.getEnableIdx()); });
      rewriter.modifyOpInPlace(defOp, [&]() {
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
RemoveUnusedArcs::matchAndRewrite(DefineOp op,
                                  PatternRewriter &rewriter) const {
  if (symbolCache.useEmpty(op)) {
    op.getBody().walk([&](mlir::CallOpInterface user) {
      if (auto symbol = dyn_cast<SymbolRefAttr>(user.getCallableForCallee()))
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
  auto negate = [&](Value input) -> Value {
    auto constTrue = rewriter.create<hw::ConstantOp>(op.getLoc(), APInt(1, 1));
    return rewriter.create<comb::XorOp>(op.getLoc(), input, constTrue,
                                        op.getTwoState());
  };

  APInt rhs;
  if (matchPattern(op.getRhs(), mlir::m_ConstantInt(&rhs))) {
    if (auto concatOp = op.getLhs().getDefiningOp<comb::ConcatOp>()) {
      if (auto optionalWidth =
              sameWidthIntegers(concatOp->getOperands().getTypes())) {
        if ((op.getPredicate() == comb::ICmpPredicate::eq ||
             op.getPredicate() == comb::ICmpPredicate::ne) &&
            rhs.isAllOnes()) {
          Value andOp = rewriter.create<comb::AndOp>(
              op.getLoc(), concatOp.getInputs(), op.getTwoState());
          if (*optionalWidth == 1) {
            if (op.getPredicate() == comb::ICmpPredicate::ne)
              andOp = negate(andOp);
            rewriter.replaceOp(op, andOp);
            return success();
          }
          rewriter.replaceOpWithNewOp<comb::ICmpOp>(
              op, op.getPredicate(), andOp,
              getConstant(APInt(*optionalWidth, rhs.getZExtValue(),
                                /*isSigned=*/false, /*implicitTrunc=*/true)),
              op.getTwoState());
          return success();
        }

        if ((op.getPredicate() == comb::ICmpPredicate::ne ||
             op.getPredicate() == comb::ICmpPredicate::eq) &&
            rhs.isZero()) {
          Value orOp = rewriter.create<comb::OrOp>(
              op.getLoc(), concatOp.getInputs(), op.getTwoState());
          if (*optionalWidth == 1) {
            if (op.getPredicate() == comb::ICmpPredicate::eq)
              orOp = negate(orOp);
            rewriter.replaceOp(op, orOp);
            return success();
          }
          rewriter.replaceOpWithNewOp<comb::ICmpOp>(
              op, op.getPredicate(), orOp,
              getConstant(APInt(*optionalWidth, rhs.getZExtValue(),
                                /*isSigned=*/false, /*implicitTrunc=*/true)),
              op.getTwoState());
          return success();
        }
      }
    }
  }
  return failure();
}

LogicalResult RemoveUnusedArcArgumentsPattern::matchAndRewrite(
    DefineOp op, PatternRewriter &rewriter) const {
  BitVector toDelete(op.getNumArguments());
  for (auto [i, arg] : llvm::enumerate(op.getArguments()))
    if (arg.use_empty())
      toDelete.set(i);

  if (toDelete.none())
    return failure();

  // Collect the mutable callers in a first iteration. If there is a user that
  // does not implement the interface, we have to abort the rewrite and have to
  // make sure that we didn't change anything so far.
  SmallVector<mlir::CallOpInterface> mutableUsers;
  for (auto *user : symbolCache.getUsers(op)) {
    auto callOpMutable = dyn_cast<mlir::CallOpInterface>(user);
    if (!callOpMutable)
      return failure();
    mutableUsers.push_back(callOpMutable);
  }

  // Do the actual rewrites.
  for (auto user : mutableUsers)
    for (int i = toDelete.size() - 1; i >= 0; --i)
      if (toDelete[i])
        user.getArgOperandsMutable().erase(i);

  op.eraseArguments(toDelete);
  op.setFunctionType(
      rewriter.getFunctionType(op.getArgumentTypes(), op.getResultTypes()));

  statistics.removeUnusedArcArgumentsPatternNumArgsRemoved += toDelete.count();
  return success();
}

LogicalResult
SinkArcInputsPattern::matchAndRewrite(DefineOp op,
                                      PatternRewriter &rewriter) const {
  // First check that all users implement the interface we need to be able to
  // modify the users.
  auto users = symbolCache.getUsers(op);
  if (llvm::any_of(
          users, [](auto *user) { return !isa<mlir::CallOpInterface>(user); }))
    return failure();

  // Find all arguments that use constant operands only.
  SmallVector<Operation *> stateConsts(op.getNumArguments());
  bool first = true;
  for (auto *user : users) {
    auto callOp = cast<mlir::CallOpInterface>(user);
    for (auto [constArg, input] :
         llvm::zip(stateConsts, callOp.getArgOperands())) {
      if (auto *constOp = input.getDefiningOp();
          constOp && constOp->template hasTrait<OpTrait::ConstantLike>()) {
        if (first) {
          constArg = constOp;
          continue;
        }
        if (constArg &&
            constArg->getName() == input.getDefiningOp()->getName() &&
            constArg->getAttrDictionary() ==
                input.getDefiningOp()->getAttrDictionary())
          continue;
      }
      constArg = nullptr;
    }
    first = false;
  }

  // Move the constants into the arc and erase the block arguments.
  rewriter.setInsertionPointToStart(&op.getBodyBlock());
  llvm::BitVector toDelete(op.getBodyBlock().getNumArguments());
  for (auto [constArg, arg] : llvm::zip(stateConsts, op.getArguments())) {
    if (!constArg)
      continue;
    auto *inlinedConst = rewriter.clone(*constArg);
    rewriter.replaceAllUsesWith(arg, inlinedConst->getResult(0));
    toDelete.set(arg.getArgNumber());
  }
  op.getBodyBlock().eraseArguments(toDelete);
  op.setType(rewriter.getFunctionType(op.getBodyBlock().getArgumentTypes(),
                                      op.getResultTypes()));

  // Rewrite all arc uses to not pass in the constant anymore.
  for (auto *user : users) {
    auto callOp = cast<mlir::CallOpInterface>(user);
    SmallPtrSet<Value, 4> maybeUnusedValues;
    SmallVector<Value> newInputs;
    for (auto [index, value] : llvm::enumerate(callOp.getArgOperands())) {
      if (toDelete[index])
        maybeUnusedValues.insert(value);
      else
        newInputs.push_back(value);
    }
    rewriter.modifyOpInPlace(
        callOp, [&]() { callOp.getArgOperandsMutable().assign(newInputs); });
    for (auto value : maybeUnusedValues)
      if (value.use_empty())
        rewriter.eraseOp(value.getDefiningOp());
  }

  return success(toDelete.any());
}

LogicalResult
CompRegCanonicalizer::matchAndRewrite(seq::CompRegOp op,
                                      PatternRewriter &rewriter) const {
  if (!op.getReset())
    return failure();

  // Because Arcilator supports constant zero reset values, skip them.
  APInt constant;
  if (mlir::matchPattern(op.getResetValue(), mlir::m_ConstantInt(&constant)))
    if (constant.isZero())
      return failure();

  Value newInput = rewriter.create<comb::MuxOp>(
      op->getLoc(), op.getReset(), op.getResetValue(), op.getInput());
  rewriter.modifyOpInPlace(op, [&]() {
    op.getInputMutable().set(newInput);
    op.getResetMutable().clear();
    op.getResetValueMutable().clear();
  });

  return success();
}

LogicalResult
MergeVectorizeOps::matchAndRewrite(VectorizeOp vecOp,
                                   PatternRewriter &rewriter) const {
  auto &currentBlock = vecOp.getBody().front();
  IRMapping argMapping;
  SmallVector<Value> newOperands;
  SmallVector<VectorizeOp> vecOpsToRemove;
  bool canBeMerged = false;
  // Used to calculate the new positions of args after insertions and removals
  unsigned paddedBy = 0;

  for (unsigned argIdx = 0, numArgs = vecOp.getInputs().size();
       argIdx < numArgs; ++argIdx) {
    auto inputVec = vecOp.getInputs()[argIdx];
    // Make sure that the input comes from a `VectorizeOp`
    // Ensure that the input vector matches the output of the `otherVecOp`
    // Make sure that the results of the otherVecOp have only one use
    auto otherVecOp = inputVec[0].getDefiningOp<VectorizeOp>();
    if (!otherVecOp || otherVecOp == vecOp ||
        !llvm::all_of(otherVecOp.getResults(),
                      [](auto result) { return result.hasOneUse(); }) ||
        !llvm::all_of(inputVec, [&](auto result) {
          return result.template getDefiningOp<VectorizeOp>() == otherVecOp;
        })) {
      newOperands.insert(newOperands.end(), inputVec.begin(), inputVec.end());
      continue;
    }

    // Here, all elements are from the same `VectorizeOp`.
    // If all elements of the input vector come from the same `VectorizeOp`
    // sort the vectors by their indices
    DenseMap<Value, size_t> resultIdxMap;
    for (auto [resultIdx, result] : llvm::enumerate(otherVecOp.getResults()))
      resultIdxMap[result] = resultIdx;

    SmallVector<Value> tempVec(inputVec.begin(), inputVec.end());
    llvm::sort(tempVec, [&](Value a, Value b) {
      return resultIdxMap[a] < resultIdxMap[b];
    });

    // Check if inputVec matches the result after sorting.
    if (tempVec != SmallVector<Value>(otherVecOp.getResults().begin(),
                                      otherVecOp.getResults().end())) {
      newOperands.insert(newOperands.end(), inputVec.begin(), inputVec.end());
      continue;
    }

    DenseMap<size_t, size_t> fromRealIdxToSortedIdx;
    for (auto [inIdx, in] : llvm::enumerate(inputVec))
      fromRealIdxToSortedIdx[inIdx] = resultIdxMap[in];

    // If this flag is set that means we changed the IR so we cannot return
    // failure
    canBeMerged = true;

    // If the results got shuffled, then shuffle the operands before merging.
    if (inputVec != otherVecOp.getResults()) {
      for (auto otherVecOpInputVec : otherVecOp.getInputs()) {
        // use the tempVec again instead of creating another one.
        tempVec = SmallVector<Value>(inputVec.size());
        for (auto [realIdx, opernad] : llvm::enumerate(otherVecOpInputVec))
          tempVec[realIdx] =
              otherVecOpInputVec[fromRealIdxToSortedIdx[realIdx]];

        newOperands.insert(newOperands.end(), tempVec.begin(), tempVec.end());
      }

    } else
      newOperands.insert(newOperands.end(), otherVecOp.getOperands().begin(),
                         otherVecOp.getOperands().end());

    auto &otherBlock = otherVecOp.getBody().front();
    for (auto &otherArg : otherBlock.getArguments()) {
      auto newArg = currentBlock.insertArgument(
          argIdx + paddedBy, otherArg.getType(), otherArg.getLoc());
      argMapping.map(otherArg, newArg);
      ++paddedBy;
    }

    rewriter.setInsertionPointToStart(&currentBlock);
    for (auto &op : otherBlock.without_terminator())
      rewriter.clone(op, argMapping);

    unsigned argNewPos = paddedBy + argIdx;
    // Get the result of the return value and use it in all places the
    // the `otherVecOp` results were used
    auto retOp = cast<VectorizeReturnOp>(otherBlock.getTerminator());
    rewriter.replaceAllUsesWith(currentBlock.getArgument(argNewPos),
                                argMapping.lookupOrDefault(retOp.getValue()));
    currentBlock.eraseArgument(argNewPos);
    vecOpsToRemove.push_back(otherVecOp);
    // We erased an arg so the padding decreased by 1
    paddedBy--;
  }

  // We didn't change the IR as there were no vectors to merge
  if (!canBeMerged)
    return failure();

  (void)updateInputOperands(vecOp, newOperands);

  // Erase dead VectorizeOps
  for (auto deadOp : vecOpsToRemove)
    rewriter.eraseOp(deadOp);

  return success();
}

namespace llvm {
static unsigned hashValue(const SmallVector<Value> &inputs) {
  unsigned hash = hash_value(inputs.size());
  for (auto input : inputs)
    hash = hash_combine(hash, input);
  return hash;
}

template <>
struct DenseMapInfo<SmallVector<Value>> {
  static inline SmallVector<Value> getEmptyKey() {
    return SmallVector<Value>();
  }

  static inline SmallVector<Value> getTombstoneKey() {
    return SmallVector<Value>();
  }

  static unsigned getHashValue(const SmallVector<Value> &inputs) {
    return hashValue(inputs);
  }

  static bool isEqual(const SmallVector<Value> &lhs,
                      const SmallVector<Value> &rhs) {
    return lhs == rhs;
  }
};
} // namespace llvm

LogicalResult KeepOneVecOp::matchAndRewrite(VectorizeOp vecOp,
                                            PatternRewriter &rewriter) const {
  DenseMap<SmallVector<Value>, unsigned> inExists;
  auto &currentBlock = vecOp.getBody().front();
  SmallVector<Value> newOperands;
  BitVector argsToRemove(vecOp.getInputs().size(), false);
  for (size_t argIdx = 0; argIdx < vecOp.getInputs().size(); ++argIdx) {
    auto input = SmallVector<Value>(vecOp.getInputs()[argIdx].begin(),
                                    vecOp.getInputs()[argIdx].end());
    if (auto in = inExists.find(input); in != inExists.end()) {
      rewriter.replaceAllUsesWith(currentBlock.getArgument(argIdx),
                                  currentBlock.getArgument(in->second));
      argsToRemove.set(argIdx);
      continue;
    }
    inExists[input] = argIdx;
    newOperands.insert(newOperands.end(), input.begin(), input.end());
  }

  if (argsToRemove.none())
    return failure();

  currentBlock.eraseArguments(argsToRemove);
  return updateInputOperands(vecOp, newOperands);
}

//===----------------------------------------------------------------------===//
// ArcCanonicalizerPass implementation
//===----------------------------------------------------------------------===//

namespace {
struct ArcCanonicalizerPass
    : public arc::impl::ArcCanonicalizerBase<ArcCanonicalizerPass> {
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
  config.enableRegionSimplification = mlir::GreedySimplifyRegionLevel::Disabled;
  config.maxIterations = 10;
  config.useTopDownTraversal = true;
  ArcListener listener(&cache);
  config.listener = &listener;

  PatternStatistics statistics;
  RewritePatternSet symbolPatterns(&getContext());
  symbolPatterns.add<CallPassthroughArc, RemoveUnusedArcs,
                     RemoveUnusedArcArgumentsPattern, SinkArcInputsPattern>(
      &getContext(), cache, names, statistics);
  symbolPatterns.add<MemWritePortEnableAndMaskCanonicalizer>(
      &getContext(), cache, names, statistics, arcMapping);

  if (failed(mlir::applyPatternsGreedily(getOperation(),
                                         std::move(symbolPatterns), config)))
    return signalPassFailure();

  numArcArgsRemoved = statistics.removeUnusedArcArgumentsPatternNumArgsRemoved;

  RewritePatternSet patterns(&ctxt);
  for (auto *dialect : ctxt.getLoadedDialects())
    dialect->getCanonicalizationPatterns(patterns);
  for (mlir::RegisteredOperationName op : ctxt.getRegisteredOperations())
    op.getCanonicalizationPatterns(patterns, &ctxt);
  patterns.add<ICMPCanonicalizer, CompRegCanonicalizer, MergeVectorizeOps,
               KeepOneVecOp>(&getContext());

  // Don't test for convergence since it is often not reached.
  (void)mlir::applyPatternsGreedily(getOperation(), std::move(patterns),
                                    config);
}

std::unique_ptr<mlir::Pass> arc::createArcCanonicalizerPass() {
  return std::make_unique<ArcCanonicalizerPass>();
}
