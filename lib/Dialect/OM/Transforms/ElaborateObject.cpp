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
#include "llvm/ADT/DenseSet.h"
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

/// Conservatively identify collection capabilities exposed by a value's type.
/// An opaque !om.any may hide a collection; an explicit any_cast can be
/// unwrapped by the caller to establish a more precise type.
bool mayContainCollection(Type type, const SymbolTable &symTable,
                          SmallPtrSetImpl<Operation *> &visited) {
  if (isa<CollectionType, AnyType>(type))
    return true;
  if (auto list = dyn_cast<ListType>(type))
    return mayContainCollection(list.getElementType(), symTable, visited);
  if (auto classType = dyn_cast<ClassType>(type)) {
    auto cls = symTable.lookup<ClassLike>(classType.getClassName().getAttr());
    if (cls && visited.insert(cls.getOperation()).second)
      for (auto name : cls.getFieldNames())
        if (mayContainCollection(*cls.getFieldType(cast<StringAttr>(name)),
                                 symTable, visited))
          return true;
  }
  return false;
}

/// Prefer actual field/element operands to conservative type information when
/// available. A missing answer defers the external call until its object/field
/// operands have been elaborated. This avoids rejecting primitive !om.any
/// inputs.
std::optional<bool>
mayContainCollection(Value value, const SymbolTable &symTable,
                     llvm::SmallDenseSet<Value, 8> &visited) {
  if (!visited.insert(value).second)
    return false;
  if (auto cast = value.getDefiningOp<AnyCastOp>())
    return mayContainCollection(cast.getInput(), symTable, visited);
  // The evaluator imports collection-bearing inputs as IR, so its wrapper
  // class parameters never hide collections.
  if (auto arg = dyn_cast<BlockArgument>(value))
    if (arg.getOwner()->getParentOp()->hasAttr(
            "om.evaluator.collection_free_args"))
      return false;
  auto *op = value.getDefiningOp();
  if (isa_and_nonnull<ObjectOp, ObjectFieldOp>(op))
    return std::nullopt;
  if (isa_and_nonnull<ElaboratedObjectOp, ListCreateOp, ListConcatOp>(op)) {
    bool deferred = false;
    for (Value operand : op->getOperands()) {
      auto capability = mayContainCollection(operand, symTable, visited);
      if (capability && *capability)
        return true;
      deferred |= !capability.has_value();
    }
    if (deferred)
      return std::nullopt;
    return false;
  }
  SmallPtrSet<Operation *, 8> visitedClasses;
  return mayContainCollection(value.getType(), symTable, visitedClasses);
}

/// Pattern to inline ObjectOp instances by cloning the class body and
/// replacing them with ElaboratedObjectOp.
struct ObjectOpInliningPattern : public OpRewritePattern<ObjectOp> {
  ObjectOpInliningPattern(MLIRContext *context, SymbolTable &symTable,
                          bool replaceExternalWithUnknown,
                          bool &unsupportedEffects)
      : OpRewritePattern<ObjectOp>(context), symTable(symTable),
        replaceExternalWithUnknown(replaceExternalWithUnknown),
        unsupportedEffects(unsupportedEffects) {}

  LogicalResult matchAndRewrite(ObjectOp objOp,
                                PatternRewriter &rewriter) const override {
    if (unsupportedEffects)
      return failure();
    auto classLike =
        symTable.lookup<ClassLike>(objOp.getClassNameAttr().getAttr());
    assert(classLike);

    // External classes cannot be elaborated; replace with unknown values.
    if (isa<ClassExternOp>(classLike)) {
      if (!replaceExternalWithUnknown)
        return failure();
      for (Value param : objOp.getActualParams()) {
        llvm::SmallDenseSet<Value, 8> visited;
        auto capability = mayContainCollection(param, symTable, visited);
        if (!capability)
          return failure();
        if (*capability) {
          objOp.emitError("cannot evaluate external constructor with "
                          "collection capabilities");
          unsupportedEffects = true;
          return failure();
        }
      }
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
    // Propagate the class's per-field locations onto each field value, fused
    // with the value's existing location.
    if (auto classOp = dyn_cast<ClassOp>(classLike.getOperation()))
      for (auto [i, v] : llvm::enumerate(fieldValues)) {
        Location fieldLoc =
            rewriter.getFusedLoc({classOp.getFieldLocByIndex(i), v.getLoc()});
        if (auto *fieldOp = v.getDefiningOp()) {
          rewriter.modifyOpInPlace(fieldOp, [&] { fieldOp->setLoc(fieldLoc); });
        } else {
          auto &val = v;
          rewriter.modifyOpInPlace(
              cast<BlockArgument>(v).getOwner()->getParentOp(),
              [&] { val.setLoc(fieldLoc); });
        }
      }

    // Erase the terminator and inline the body at the object instantiation.
    rewriter.eraseOp(clonedFields);
    rewriter.inlineBlockBefore(clonedBlock, objOp);

    rewriter.replaceOpWithNewOp<ElaboratedObjectOp>(objOp, classLike,
                                                    fieldValues);

    return success();
  }

  const SymbolTable &symTable;
  bool replaceExternalWithUnknown;
  bool &unsupportedEffects;
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
        symTable.lookup<ClassLike>(elaboratedOp.getClassNameAttr().getAttr());
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

// Operations supported as residual IR after elaboration. Collection pushes
// only remain when they target a class parameter.
bool isSupportedResidual(Operation *op) {
  return isa<
      // Structure.
      ClassOp, ClassFieldsOp, ElaboratedObjectOp, AnyCastOp,
      // Constant-like.
      ConstantOp, UnknownValueOp,
      // Path.
      FrozenBasePathCreateOp, FrozenPathCreateOp, FrozenEmptyPathOp,
      // Containers.
      ListCreateOp, ListConcatOp, CollectionCreateOp, CollectionPushOp>(op);
}

LogicalResult verifyResult(ClassOp module, bool allowUnevaluated) {
  auto isLegal = [allowUnevaluated](Operation *op) -> LogicalResult {
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

        // The message is supposed to be fully evaluated at this point, though
        // it could be unknown.
        auto messageOp =
            dyn_cast_or_null<ConstantOp>(assertOp.getMessage().getDefiningOp());
        if (!messageOp) {
          if (allowUnevaluated)
            return op->emitError("OM property assertion failed: <unevaluated>");

          auto diag = emitError(op->getLoc(),
                                "OM property assertion failed, but no message "
                                "is available as the message is unevaluated");
          diag.attachNote(assertOp.getMessage().getLoc())
              << "unevaluated message operation is here";
          return failure();
        }

        StringAttr message;
        if (!matchPattern(assertOp.getMessage(), m_Constant(&message)))
          return op->emitError()
                 << "OM property assertion failed, but no message is available "
                    "because the message is not a constant string";
        return op->emitError("OM property assertion failed: ")
               << message.getValue();
      };

      // Condition is a constant integer/bool - check if it's true.
      if (matchPattern(assertOp.getCondition(), m_ConstantInt(&value)))
        return checkAssert(!value.isZero());

      // Condition is unknown - treat as passing.
      if (auto unknownOp = dyn_cast_or_null<UnknownValueOp>(defOp))
        return checkAssert(true);

      // This means the condition was not fully evaluated.
      if (allowUnevaluated)
        return success();
      return emitError(op->getLoc(), "failed to evaluate assertion condition");
    }

    if (!isSupportedResidual(op)) {
      if (allowUnevaluated)
        return success();
      return emitError(op->getLoc()) << "failed to evaluate " << op->getName();
    }

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
                                      FieldIndex &fieldIndexes,
                                      bool allowUnevaluated = false) {
    // Elaborate objects by inlining all ObjectOps and folding field accesses
    // using a greedy pattern rewriter. NOTE: The conversion framework is not
    // suitable here because inlining patterns need to be applied recursively to
    // fully evaluate nested object instantiations.
    bool unsupportedEffects = false;
    RewritePatternSet patterns(classOp.getContext());
    patterns.add<ObjectOpInliningPattern>(
        classOp.getContext(), symTable, !allowUnevaluated, unsupportedEffects);
    patterns.add<EvaluateObjectField>(classOp.getContext(), symTable,
                                      fieldIndexes);
    patterns.add<UnknownPropagationPattern>(classOp.getContext());
    CollectionPushOp::getCanonicalizationPatterns(patterns,
                                                  classOp.getContext());
    GreedyRewriteConfig config;
    // Disable iteration limit to allow full recursive inlining.
    config.setMaxIterations(GreedyRewriteConfig::kNoLimit);
    if (failed(applyPatternsGreedily(classOp, std::move(patterns), config)))
      return failure();

    if (unsupportedEffects)
      return failure();
    // Check if elaboration succeeded after saturation.
    return verifyResult(classOp, allowUnevaluated);
  }

  LogicalResult initialize(MLIRContext *context) override {
    unsigned numModes =
        allPublicClasses.getValue() + !targetClass.getValue().empty();
    if (numModes != 1)
      return emitError(UnknownLoc::get(context))
             << "exactly one of 'target-class' or 'all-public-classes' must "
                "be specified";
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

    // Elaborate all public classes.
    if (allPublicClasses) {
      for (auto classOp : module.getOps<ClassOp>()) {
        if (!classOp.isPublic())
          continue;
        if (failed(elaborateClass(classOp, symTable, fieldIndexes,
                                  allowUnevaluated)))
          return signalPassFailure();
      }
      return;
    }

    // Normal mode: elaborate the specified target class.
    auto classOp = symTable.lookup<ClassOp>(targetClass);
    if (!classOp) {
      emitError(module.getLoc())
          << "target class '" << targetClass << "' was not found";
      return signalPassFailure();
    }

    if (failed(
            elaborateClass(classOp, symTable, fieldIndexes, allowUnevaluated)))
      return signalPassFailure();
  }
};

} // namespace
