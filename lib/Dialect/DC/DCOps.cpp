//===- DCOps.cpp ----------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "circt/Dialect/DC/DCOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/FunctionImplementation.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

using namespace circt;
using namespace dc;
using namespace mlir;

bool circt::dc::isI1ValueType(Type t) {
  auto vt = t.dyn_cast<ValueType>();
  if (!vt || vt.getInnerTypes().size() != 1)
    return false;

  auto innerWidth = vt.getInnerTypes()[0].getIntOrFloatBitWidth();
  return innerWidth == 1;
}

namespace circt {
namespace dc {

// =============================================================================
// JoinOp
// =============================================================================

class IdenticalJoinCanonicalizationPattern : public OpRewritePattern<JoinOp> {
  // Canonicalization of joins where all of the inputs are the same.
public:
  using OpRewritePattern<JoinOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(JoinOp join,
                                PatternRewriter &rewriter) const override {
    auto inputs = join.getTokens();
    if (inputs.size() < 2)
      return failure();

    // Coarse-grain find candidates which have a single similar input.
    llvm::SetVector<JoinOp> similarCandidates;
    for (auto input : inputs) {
      for (auto *user : input.getUsers()) {
        auto userJoin = dyn_cast<JoinOp>(user);
        if (!userJoin || userJoin == join)
          continue;
        similarCandidates.insert(userJoin);
      }
    }

    if (similarCandidates.empty())
      return failure();

    // Now filter that set based on whether all of the inputs are the same.
    llvm::SetVector<JoinOp> identicalCandidates;
    for (auto candidate : similarCandidates) {
      auto candidateInputs = candidate.getTokens();
      if (candidateInputs.size() != inputs.size())
        continue;

      bool allInputsIdentical = true;
      for (auto input : inputs) {
        if (!llvm::is_contained(candidateInputs, input)) {
          allInputsIdentical = false;
          break;
        }
      }

      if (allInputsIdentical)
        identicalCandidates.insert(candidate);
    }

    if (identicalCandidates.empty())
      return failure();

    // Replace all of the identical candidates with the original join.
    rewriter.updateRootInPlace(join, [&] {
      for (auto candidate : identicalCandidates) {
        rewriter.replaceOp(candidate, join.getResult());
      }
    });
    return success();
  }
};

class TransitiveJoinCanonicalizationPattern : public OpRewritePattern<JoinOp> {
  // Canonicalization staggered joins where the sink join contains inputs also
  // found in the source join.
public:
  using OpRewritePattern<JoinOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(JoinOp join,
                                PatternRewriter &rewriter) const override {
    SetVector<mlir::Value> newInputs;
    bool didSomething = false;
    for (auto operand : join.getTokens()) {
      auto inputJoin = operand.getDefiningOp<dc::JoinOp>();
      if (!inputJoin) {
        // Operand does not originate from a join, so just add it to the new
        // set of inputs directly.
        newInputs.insert(operand);
        continue;
      }

      // Operand originates from a join, so add all of its inputs to the new
      // set of inputs.
      for (auto input : inputJoin.getTokens())
        newInputs.insert(input);

      didSomething = true;
    }

    if (!didSomething)
      return failure();

    // We've now transitively added all of the next-level-up inputs to the
    // new set of inputs, and uniqued them via. the set.
    rewriter.replaceOpWithNewOp<dc::JoinOp>(join, newInputs.getArrayRef());
    return success();
  }
};

class JoinOnSourcePattern : public OpRewritePattern<JoinOp> {
  // Removes join operands which originate from source ops.
public:
  using OpRewritePattern<JoinOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(JoinOp join,
                                PatternRewriter &rewriter) const override {
    SetVector<mlir::Value> newInputs;
    bool didSomething = false;
    for (auto operand : join.getTokens()) {
      auto inputSource = operand.getDefiningOp<dc::SourceOp>();
      if (!inputSource) {
        // Operand does not originate from a source, so just add it to the new
        // set of inputs directly.
        newInputs.insert(operand);
        continue;
      }

      didSomething = true;
    }

    if (!didSomething)
      return failure();

    rewriter.replaceOpWithNewOp<dc::JoinOp>(join, newInputs.getArrayRef());
    return success();
  }
};

class RedundantJoinOperandsPattern : public OpRewritePattern<JoinOp> {
  // Removes duplicate operands to joins.
public:
  using OpRewritePattern<JoinOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(JoinOp join,
                                PatternRewriter &rewriter) const override {
    llvm::SetVector<Value> uniqueInputs;

    for (auto operand : join.getTokens())
      uniqueInputs.insert(operand);

    if (uniqueInputs.size() == join.getTokens().size())
      return failure();

    rewriter.replaceOpWithNewOp<dc::JoinOp>(join, uniqueInputs.getArrayRef());
    return success();
  }
};

void JoinOp::getCanonicalizationPatterns(RewritePatternSet &results,
                                         MLIRContext *context) {
  results.insert<TransitiveJoinCanonicalizationPattern,
                 IdenticalJoinCanonicalizationPattern,
                 RedundantJoinOperandsPattern, JoinOnSourcePattern>(context);
}

OpFoldResult JoinOp::fold(FoldAdaptor adaptor) {
  // Fold simple joins (joins with 1 input).
  if (auto tokens = getTokens(); tokens.size() == 1)
    return tokens.front();

  return {};
}

// =============================================================================
// ForkOp
// =============================================================================

template <typename TInt>
static ParseResult parseIntInSquareBrackets(OpAsmParser &parser, TInt &v) {
  if (parser.parseLSquare() || parser.parseInteger(v) || parser.parseRSquare())
    return failure();
  return success();
}

ParseResult ForkOp::parse(OpAsmParser &parser, OperationState &result) {
  OpAsmParser::UnresolvedOperand operand;
  size_t size = 0;
  if (parseIntInSquareBrackets(parser, size))
    return failure();

  if (size == 0)
    return parser.emitError(parser.getNameLoc(),
                            "fork size must be greater than 0");

  if (parser.parseOperand(operand) ||
      parser.parseOptionalAttrDict(result.attributes))
    return failure();

  auto tt = dc::TokenType::get(parser.getContext());
  llvm::SmallVector<Type> operandTypes{tt};
  SmallVector<Type> resultTypes{size, tt};
  result.addTypes(resultTypes);
  if (parser.resolveOperand(operand, tt, result.operands))
    return failure();
  return success();
}

void ForkOp::print(OpAsmPrinter &p) {
  p << " [" << getNumResults() << "] ";
  p << getOperand() << " ";
  auto attrs = (*this)->getAttrs();
  if (!attrs.empty()) {
    p << " ";
    p.printOptionalAttrDict(attrs);
  }
}

class EliminateForkToForkPattern : public OpRewritePattern<ForkOp> {
  // Canonicalization of forks where the output is fed into another fork.
public:
  using OpRewritePattern<ForkOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(ForkOp fork,
                                PatternRewriter &rewriter) const override {
    for (auto output : fork.getOutputs()) {
      for (auto *user : output.getUsers()) {
        auto userFork = dyn_cast<ForkOp>(user);
        if (!userFork)
          return failure();

        // We have a fork feeding into another fork. Replace the output fork by
        // adding more outputs to the current fork.
        size_t totalForks = fork.getNumResults() + userFork.getNumResults() - 1;

        auto newFork = rewriter.create<dc::ForkOp>(fork.getLoc(),
                                                   fork.getToken(), totalForks);
        rewriter.replaceOp(
            fork, newFork.getResults().take_front(fork.getNumResults()));
        rewriter.replaceOp(
            userFork, newFork.getResults().take_back(userFork.getNumResults()));

        // Just stop the pattern here instead of trying to do more - let the
        // canonicalizer recurse if another run of the canonicalization applies.
        return success();
      }
    }
    return failure();
  }
};

class EliminateForkOfSourcePattern : public OpRewritePattern<ForkOp> {
  // Canonicalizes away forks on source ops, in favor of individual source
  // operations. Having standalone sources are a better alternative, since other
  // operations can canonicalize on it (e.g. joins) as well as being very cheap
  // to implement in hardware, if they do remain.
public:
  using OpRewritePattern<ForkOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(ForkOp fork,
                                PatternRewriter &rewriter) const override {
    auto source = fork.getToken().getDefiningOp<SourceOp>();
    if (!source)
      return failure();

    // We have a source feeding into a fork. Replace the fork by a source for
    // each output.
    llvm::SmallVector<Value> sources;
    for (size_t i = 0; i < fork.getNumResults(); ++i)
      sources.push_back(rewriter.create<dc::SourceOp>(fork.getLoc()));

    rewriter.replaceOp(fork, sources);
    return success();
  }
};

void ForkOp::getCanonicalizationPatterns(RewritePatternSet &results,
                                         MLIRContext *context) {
  results.insert<EliminateForkToForkPattern, EliminateForkOfSourcePattern>(
      context);
}

LogicalResult ForkOp::fold(FoldAdaptor adaptor,
                           SmallVectorImpl<OpFoldResult> &results) {
  // Fold simple forks (forks with 1 output).
  if (getOutputs().size() == 1) {
    results.push_back(getToken());
    return success();
  }

  return failure();
}

// =============================================================================
// UnpackOp
// =============================================================================

struct EliminateMultipleUnpackPattern : public OpRewritePattern<UnpackOp> {
  // Example:
  // %0, %1 = !dc.unpack %v : ...
  // %2, %3 = !dc.unpack %v : ...
  // ->
  // %0, %1 = !dc.unpack %v : ...
  // %2 -> %0, %3 -> %1
  using OpRewritePattern<UnpackOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(UnpackOp unpack,
                                PatternRewriter &rewriter) const override {
    auto unpackedValue = unpack.getInput();
    // Find other unpacks of the same value.
    llvm::SmallVector<UnpackOp> users;
    for (auto *user : unpackedValue.getUsers()) {
      auto unpack = dyn_cast<UnpackOp>(user);
      if (!unpack)
        return failure();
      users.push_back(unpack);
    }

    if (users.size() == 1)
      return failure();

    // Replace all unpacks with a single unpack - just create a new one.
    auto newUnpack = rewriter.create<UnpackOp>(unpack.getLoc(), unpackedValue);
    for (auto user : users)
      rewriter.replaceOp(user, newUnpack.getResults());

    return success();
  }
};

struct EliminateRedundantUnpackPattern : public OpRewritePattern<UnpackOp> {
  // Eliminates unpacks where only the token is used.
  using OpRewritePattern<UnpackOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(UnpackOp unpack,
                                PatternRewriter &rewriter) const override {
    // Is the value-side of the unpack used?
    if (!llvm::all_of(unpack.getOutputs(),
                      [](auto output) { return output.use_empty(); }))
      return failure();

    auto pack = unpack.getInput().getDefiningOp<PackOp>();
    if (!pack)
      return failure();

    // Replace all uses of the unpack token with the packed token.
    rewriter.replaceAllUsesWith(unpack.getToken(), pack.getToken());
    rewriter.eraseOp(unpack);
    return success();
  }
};

void UnpackOp::getCanonicalizationPatterns(RewritePatternSet &results,
                                           MLIRContext *context) {
  results
      .insert<EliminateMultipleUnpackPattern, EliminateRedundantUnpackPattern>(
          context);
}

LogicalResult UnpackOp::fold(FoldAdaptor adaptor,
                             SmallVectorImpl<OpFoldResult> &results) {
  // Unpack of a pack is a no-op.
  if (auto pack = getInput().getDefiningOp<PackOp>()) {
    results.push_back(pack.getToken());
    results.append(pack.getInputs().begin(), pack.getInputs().end());
    return success();
  }

  return failure();
}

LogicalResult UnpackOp::inferReturnTypes(
    MLIRContext *context, std::optional<Location> loc, ValueRange operands,
    DictionaryAttr attrs, mlir::OpaqueProperties properties,
    mlir::RegionRange regions, SmallVectorImpl<Type> &results) {
  auto inputType = operands.front().getType().cast<ValueType>();
  results.push_back(TokenType::get(context));
  results.append(inputType.getInnerTypes().begin(),
                 inputType.getInnerTypes().end());
  return success();
}

// =============================================================================
// PackOp
// =============================================================================

class EliminateMultiplePackPattern : public OpRewritePattern<PackOp> {
  // example:
  // %0 = dc.pack %token, %a, %b : !dc.value<...>
  // %1 = dc.pack %token, %a, %b : !dc.value<...>
  // ->
  // %0 = dc.pack %token, %a, %b : !dc.value<...>
  // %1 -> %0
public:
  using OpRewritePattern<PackOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(PackOp pack,
                                PatternRewriter &rewriter) const override {
    // Find other packs of the token and the input values - order matters.
    auto token = pack.getToken();
    auto inputs = pack.getInputs();
    llvm::SmallVector<PackOp> otherPacks;
    for (auto *user : token.getUsers()) {
      auto otherPack = dyn_cast<PackOp>(user);
      if (!otherPack)
        continue;

      if (llvm::any_of(llvm::zip(inputs, otherPack.getInputs()), [](auto it) {
            return std::get<0>(it) != std::get<1>(it);
          }))
        continue;

      otherPacks.push_back(otherPack);
    }

    if (otherPacks.size() == 1)
      return failure();

    // Replace all packs with a single pack - just create a new one.
    auto newPack = rewriter.create<PackOp>(pack.getLoc(), pack.getToken(),
                                           pack.getInputs());
    otherPacks.push_back(pack);
    for (auto otherPack : otherPacks)
      rewriter.replaceOp(otherPack, newPack.getResult());

    return success();
  }
};

void PackOp::getCanonicalizationPatterns(RewritePatternSet &results,
                                         MLIRContext *context) {
  results.insert<EliminateMultiplePackPattern>(context);
}

OpFoldResult PackOp::fold(FoldAdaptor adaptor) {
  auto token = getToken();
  auto inputs = getInputs();

  // Pack of an unpack is a no-op.
  if (auto unpack = token.getDefiningOp<UnpackOp>()) {
    llvm::SmallVector<Value> unpackResults = unpack.getResults();
    if (unpackResults.size() == inputs.size() &&
        llvm::all_of(llvm::zip(getInputs(), unpackResults), [&](auto it) {
          return std::get<0>(it) == std::get<1>(it);
        })) {
      return unpack.getInput();
    }
  }

  return {};
}

LogicalResult PackOp::inferReturnTypes(
    MLIRContext *context, std::optional<Location> loc, ValueRange operands,
    DictionaryAttr attrs, mlir::OpaqueProperties properties,
    mlir::RegionRange regions, SmallVectorImpl<Type> &results) {
  llvm::SmallVector<Type> inputTypes;
  for (auto t : operands.drop_front().getTypes())
    inputTypes.push_back(t);
  auto valueType = dc::ValueType::get(context, inputTypes);
  results.push_back(valueType);
  return success();
}

// =============================================================================
// SelectOp
// =============================================================================

class EliminateBranchToSelectPattern : public OpRewritePattern<SelectOp> {
  // Canonicalize away a select that is fed only by a single branch
  // example:
  //   %true, %false = dc.branch %sel1 %token
  //   %0 = dc.select %sel2, %true, %false
  // ->
  //   %0 = dc.join %sel1, %sel2, %token

public:
  using OpRewritePattern<SelectOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(SelectOp select,
                                PatternRewriter &rewriter) const override {
    // Do all the inputs come from a branch?
    BranchOp branchInput;
    for (auto operand : {select.getTrueToken(), select.getFalseToken()}) {
      auto br = operand.getDefiningOp<BranchOp>();
      if (!br)
        return failure();

      if (!branchInput)
        branchInput = br;
      else if (branchInput != br)
        return failure();
    }

    // Replace the select with a join (unpack the select conditions).
    rewriter.replaceOpWithNewOp<JoinOp>(
        select,
        llvm::SmallVector<Value>{
            rewriter.create<UnpackOp>(select.getLoc(), select.getCondition())
                .getToken(),
            rewriter
                .create<UnpackOp>(branchInput.getLoc(),
                                  branchInput.getCondition())
                .getToken()});

    return success();
  }
};

void SelectOp::getCanonicalizationPatterns(RewritePatternSet &results,
                                           MLIRContext *context) {
  results.insert<EliminateBranchToSelectPattern>(context);
}

// =============================================================================
// BufferOp
// =============================================================================

FailureOr<SmallVector<int64_t>> BufferOp::getInitValueArray() {
  assert(getInitValues() && "initValues attribute not set");
  SmallVector<int64_t> values;
  for (auto value : getInitValuesAttr()) {
    if (auto iValue = value.dyn_cast<IntegerAttr>()) {
      values.push_back(iValue.getValue().getSExtValue());
    } else {
      return emitError() << "initValues attribute must be an array of integers";
    }
  }
  return values;
}

LogicalResult BufferOp::verify() {
  // Verify that exactly 'size' number of initial values have been provided, if
  // an initializer list have been provided.
  if (auto initVals = getInitValuesAttr()) {
    auto nInits = initVals.size();
    if (nInits != getSize())
      return emitOpError() << "expected " << getSize()
                           << " init values but got " << nInits << ".";
  }

  return success();
}

} // namespace dc
} // namespace circt

#define GET_OP_CLASSES
#include "circt/Dialect/DC/DC.cpp.inc"
