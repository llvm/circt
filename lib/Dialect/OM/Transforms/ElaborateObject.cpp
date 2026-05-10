//===- ElaborateObject.cpp - OM compile-time evaluation pass --------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This pass performs evaluation of OM classes by inlining object
// instantiations and folding field accesses. It replaces the runtime Evaluator
// framework with static compile-time evaluation.
//
//===----------------------------------------------------------------------===//

#include "circt/Dialect/OM/OMDialect.h"
#include "circt/Dialect/OM/OMOps.h"
#include "circt/Dialect/OM/OMPasses.h"
#include "circt/Support/LLVM.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Support/WalkResult.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/LogicalResult.h"

namespace circt {
namespace om {
#define GEN_PASS_DEF_ELABORATEOBJECT
#include "circt/Dialect/OM/OMPasses.h.inc"
} // namespace om
} // namespace circt

using namespace mlir;
using namespace circt;
using namespace om;

namespace {
// A map from (class name, field name) to field index.
using FieldIndex = DenseMap<std::pair<StringAttr, StringAttr>, unsigned>;

/// Pattern to inline ObjectOp instances by cloning the class body and
/// replacing them with ElaboratedObjectOp.
struct ObjectOpInliningPattern : public OpRewritePattern<ObjectOp> {
  ObjectOpInliningPattern(MLIRContext *context, SymbolTable &symTable)
      : OpRewritePattern<ObjectOp>(context), symTable(symTable) {}

  LogicalResult matchAndRewrite(ObjectOp objOp,
                                PatternRewriter &rewriter) const override {
    auto classLike = symTable.lookup<ClassLike>(objOp.getClassNameAttr());
    assert(classLike);

    // External classes cannot be elaborated; replace with unknown values.
    if (isa<ClassExternOp>(classLike)) {
      rewriter.replaceOpWithNewOp<UnknownValueOp>(objOp, objOp.getType());
      return success();
    }

    auto classOp = dyn_cast<ClassOp>(classLike.getOperation());
    if (!classOp)
      return failure();

    IRMapping mapper;
    for (auto [formal, actual] : llvm::zip(
             classOp.getBodyBlock()->getArguments(), objOp.getActualParams()))
      mapper.map(formal, actual);

    // Clone the class body into a temporary region with argument substitution.
    Region clonedRegion;
    classOp.getBody().cloneInto(&clonedRegion, mapper);
    Block *clonedBlock = &clonedRegion.front();

    auto clonedFields = cast<ClassFieldsOp>(clonedBlock->getTerminator());
    SmallVector<Value> fieldValues(clonedFields.getFields());

    // Erase the terminator and inline the body at the object instantiation.
    rewriter.eraseOp(clonedFields);
    rewriter.inlineBlockBefore(clonedBlock, objOp);

    rewriter.replaceOpWithNewOp<ElaboratedObjectOp>(objOp, classLike,
                                                    fieldValues);

    return success();
  }

  const SymbolTable &symTable;
};

/// Pattern to fold ObjectFieldOp on ElaboratedObjectOp by directly accessing
/// the field value operands.
struct EvaluateObjectField : OpRewritePattern<ObjectFieldOp> {
  EvaluateObjectField(MLIRContext *context, const SymbolTable &symTable,
                      const FieldIndex &fieldIndexes)
      : OpRewritePattern<ObjectFieldOp>(context), symTable(symTable),
        fieldIndexes(fieldIndexes) {}

  LogicalResult matchAndRewrite(ObjectFieldOp op,
                                PatternRewriter &rewriter) const override {
    // Only fold if the object is an ElaboratedObjectOp.
    auto elaboratedOp = op.getObject().getDefiningOp<ElaboratedObjectOp>();
    if (!elaboratedOp)
      return failure();

    auto classLike =
        symTable.lookup<ClassLike>(elaboratedOp.getClassNameAttr());
    assert(classLike);

    // Find the field index and get the corresponding value.
    auto index =
        fieldIndexes.at({classLike.getSymNameAttr(), op.getFieldAttr()});
    auto result = elaboratedOp.getFieldValues()[index];

    // Skip cycles where a field references itself.
    // This will be raised as an error later.
    if (op.getResult() == result)
      return failure();

    rewriter.replaceOp(op, result);
    return success();
  }

  const SymbolTable &symTable;
  const FieldIndex &fieldIndexes;
};

/// Pattern to propagate UnknownValueOp through pure OM operations.
/// If any operand is unknown, all results become unknown.
struct UnknownPropagationPattern : RewritePattern {
  UnknownPropagationPattern(MLIRContext *context)
      : RewritePattern(MatchAnyOpTypeTag(), /*benefit=*/1, context) {}

  LogicalResult matchAndRewrite(Operation *op,
                                PatternRewriter &rewriter) const override {
    // Only target pure OM operations.
    // TODO: Consider add a trait for this if we want to have more explict
    // behavior.
    if (!isa_and_nonnull<OMDialect>(op->getDialect()) || !isPure(op) ||
        op->getNumResults() == 0)
      return failure();

    // Check if any operand is an UnknownValueOp.
    // TODO: This directly ports the existing Evaluator semantics, but it
    // causes inconsistent evaluation for operations that can reason about
    // known values, e.g., "and(0, unknown) -> 0".
    if (!llvm::any_of(op->getOperands(), [](Value operand) {
          return operand.getDefiningOp<UnknownValueOp>();
        }))
      return failure();

    // Replace all results with UnknownValueOp.
    SmallVector<Value> unknowns;
    for (Type resultType : op->getResultTypes())
      unknowns.push_back(
          UnknownValueOp::create(rewriter, op->getLoc(), resultType));

    rewriter.replaceOp(op, unknowns);
    return success();
  }
};

// Check if an operation can be evaluated at compile time and is valid to
// remain in the IR after elaboration.
bool isFullyEvaluated(Operation *op) {
  return isa<
      // Structure.
      ClassOp, ClassFieldsOp, ElaboratedObjectOp, AnyCastOp,
      // Constant-like.
      ConstantOp, UnknownValueOp,
      // Path.
      FrozenBasePathCreateOp, FrozenPathCreateOp, FrozenEmptyPathOp,
      // List.
      ListCreateOp, ListConcatOp>(op);
}

LogicalResult verifyResult(ClassOp module) {
  auto isLegal = [](Operation *op) -> LogicalResult {
    // Check assert satisfied.
    if (auto assertOp = dyn_cast<PropertyAssertOp>(op)) {
      // Check if the condition is a constant false, which means the assertion
      // is violated.
      auto *defOp = assertOp.getCondition().getDefiningOp();
      APInt value;
      auto checkAssert = [&](bool cond) -> LogicalResult {
        if (cond) {
          // Erase when success, serialization doesn't need to care about
          // this.
          op->erase();
          return success();
        }

        return op->emitError("OM property assertion failed: ")
               << assertOp.getMessage();
      };

      // Condition is a constant integer/bool - check if it's true.
      if (matchPattern(assertOp.getCondition(), m_ConstantInt(&value)))
        return checkAssert(!value.isZero());

      // Condition is unknown - treat as passing.
      if (auto unknownOp = dyn_cast_or_null<UnknownValueOp>(defOp))
        return checkAssert(true);

      // This means the condition was not fully evaluated.
      return emitError(op->getLoc(), "failed to evaluate assertion condition");
    }

    if (!isFullyEvaluated(op))
      return emitError(op->getLoc()) << "failed to evaluate " << op->getName();

    return success();
  };
  bool encounteredError = false;
  module.walk([&](Operation *op) { encounteredError |= failed(isLegal(op)); });

  return failure(encounteredError);
}

struct ElaborateObjectPass
    : public circt::om::impl::ElaborateObjectBase<ElaborateObjectPass> {
  using Base::Base;

  static LogicalResult elaborateClass(ClassOp classOp, SymbolTable &symTable,
                                      FieldIndex &fieldIndexes) {
    // Elaborate objects by inlining all ObjectOps and folding field accesses
    // using a greedy pattern rewriter. NOTE: The conversion framework is not
    // suitable here because inlining patterns need to be applied recursively to
    // fully evaluate nested object instantiations.
    RewritePatternSet patterns(classOp.getContext());
    patterns.add<ObjectOpInliningPattern>(classOp.getContext(), symTable);
    patterns.add<EvaluateObjectField>(classOp.getContext(), symTable,
                                      fieldIndexes);
    patterns.add<UnknownPropagationPattern>(classOp.getContext());
    GreedyRewriteConfig config;
    // Disable iteration limit to allow full recursive inlining.
    config.setMaxIterations(GreedyRewriteConfig::kNoLimit);
    if (failed(applyPatternsGreedily(classOp, std::move(patterns), config)))
      return failure();

    // Check if elaboration succeeded after saturation.
    return verifyResult(classOp);
  }

  LogicalResult initialize(MLIRContext *context) override {
    if (test.getValue() ^ targetClass.getValue().empty())
      return emitError(UnknownLoc::get(context))
             << "either 'test' or 'target-class' must be specified";
    return success();
  }

  void runOnOperation() override {
    auto module = getOperation();
    auto &symTable = getAnalysis<SymbolTable>();

    // Build a map from (class name, field name) to field index for all
    // classes.
    FieldIndex fieldIndexes;
    for (auto classOp : module.getOps<ClassLike>()) {
      auto name = classOp.getSymNameAttr();
      for (auto [idx, fieldName] :
           llvm::enumerate(classOp.getFieldNames().getAsRange<StringAttr>()))
        fieldIndexes[{name, fieldName}] = idx;
    }

    // Test mode: elaborate all zero-argument classes.
    if (test) {
      for (auto classOp : module.getOps<ClassOp>())
        if (classOp.getBodyBlock()->getNumArguments() == 0)
          if (failed(elaborateClass(classOp, symTable, fieldIndexes)))
            return signalPassFailure();
      return;
    }

    // Normal mode: elaborate the specified target class.
    auto classOp = symTable.lookup<ClassOp>(targetClass);
    if (!classOp) {
      emitError(module.getLoc())
          << "target class '" << targetClass << "' was not found";
      return signalPassFailure();
    }

    if (failed(elaborateClass(classOp, symTable, fieldIndexes)))
      return signalPassFailure();
  }
};

} // namespace
