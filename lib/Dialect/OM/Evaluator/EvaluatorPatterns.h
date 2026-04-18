//===- EvaluatorPatterns.h - OM evaluator pattern details -------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Stuff shared between Evaluator.cpp and EvaluatorPatterns.cpp.
//
//===----------------------------------------------------------------------===//

// clang-tidy seems to expect the absolute path in the header guard on some
// systems, so just disable it.
// NOLINTNEXTLINE(llvm-header-guard)
#ifndef DIALECT_OM_EVALUATOR_EVALUATORPATTERNS_H
#define DIALECT_OM_EVALUATOR_EVALUATORPATTERNS_H

#include "circt/Dialect/OM/Evaluator/Evaluator.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/Operation.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/STLFunctionalExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/LogicalResult.h"
#include <memory>
#include <optional>
#include <utility>

namespace circt {
namespace om {
namespace detail {

using ResolutionState = evaluator::ResolutionState;
using ResolvedValue = evaluator::ResolvedValue;

ResolvedValue resolveValueState(evaluator::EvaluatorValuePtr currentValue);
evaluator::EvaluatorValue *
resolveReadyValue(evaluator::EvaluatorValuePtr value);
bool isUnknownReadyValue(evaluator::EvaluatorValuePtr value);

std::optional<ResolvedValue>
requireReady(const ResolvedValue &resolved,
             evaluator::EvaluatorValuePtr pendingValue,
             llvm::function_ref<void()> emitFailure,
             evaluator::EvaluatorValuePtr &readyValue);

std::optional<ResolvedValue> requireAllOperandsReady(
    ValueRange operands, evaluator::EvaluatorValuePtr pendingValue,
    llvm::function_ref<ResolvedValue(Value)> evaluateOperand,
    llvm::function_ref<void()> emitFailure,
    SmallVectorImpl<evaluator::EvaluatorValuePtr> &readyOperands,
    bool &existsUnknown);

ResolvedValue markUnknownAndReturn(evaluator::EvaluatorValuePtr value);
LogicalResult setAttrResult(evaluator::EvaluatorValuePtr resultValue,
                            Attribute attr);
LogicalResult foldSingleResultOperation(
    Operation *op, ArrayRef<evaluator::EvaluatorValuePtr> readyOperands,
    evaluator::EvaluatorValuePtr resultValue,
    StringRef failureMessage = "failed to evaluate operation");

template <typename ValueT>
ValueT *getReadyAs(evaluator::EvaluatorValuePtr value) {
  auto *typedValue =
      llvm::dyn_cast<ValueT>(resolveReadyValue(std::move(value)));
  assert(typedValue);
  return typedValue;
}

template <typename AttrT = Attribute>
AttrT getAsAttr(evaluator::EvaluatorValuePtr value) {
  return llvm::dyn_cast<AttrT>(
      getReadyAs<evaluator::AttributeValue>(std::move(value))->getAttr());
}

/// Base class for one OM operation in the evaluator.
/// A pattern picks an initial value for the result and later fills it in.
class OperationPattern {
public:
  using GetPartialValueForTypeFn =
      llvm::function_ref<FailureOr<evaluator::EvaluatorValuePtr>(Type,
                                                                 Location)>;
  using GetValueForFn =
      llvm::function_ref<FailureOr<evaluator::EvaluatorValuePtr>(Value,
                                                                 Location)>;

  explicit OperationPattern(StringRef operationName)
      : operationName(operationName) {}
  virtual ~OperationPattern() = default;

  StringRef getOperationName() const { return operationName; }
  FailureOr<evaluator::EvaluatorValuePtr>
  createInitialValue(Operation *op, Value value,
                  GetPartialValueForTypeFn getPartialValueForType,
                  GetValueForFn getValueFor, Location loc) const {
    return createInitialValueImpl(op, value, getPartialValueForType, getValueFor,
                               loc);
  }

  virtual ResolvedValue
  evaluate(Operation *op, evaluator::EvaluatorValuePtr resultValue,
           llvm::function_ref<ResolvedValue(Value)> evaluateValue,
           Location loc) const = 0;

protected:
  static FailureOr<evaluator::EvaluatorValuePtr>
  ccreateDefaultInitialValue(Value value,
                         GetPartialValueForTypeFn getPartialValueForType,
                         Location loc) {
    return getPartialValueForType(value.getType(), loc);
  }

private:
  virtual FailureOr<evaluator::EvaluatorValuePtr>
  createInitialValueImpl(Operation *op, Value value,
                      GetPartialValueForTypeFn getPartialValueForType,
                      GetValueForFn getValueFor, Location loc) const {
    return ccreateDefaultInitialValue(value, getPartialValueForType, loc);
  }

  StringRef operationName;
};

template <typename OpT>
class OpPattern : public OperationPattern {
public:
  using OperationPattern::OperationPattern;

private:
  FailureOr<evaluator::EvaluatorValuePtr>
  createInitialValueImpl(Operation *op, Value value,
                      GetPartialValueForTypeFn getPartialValueForType,
                      GetValueForFn getValueFor, Location loc) const final {
    return createInitialValueFor(cast<OpT>(op), value, getPartialValueForType,
                              getValueFor, loc);
  }

  ResolvedValue evaluate(Operation *op,
                         evaluator::EvaluatorValuePtr resultValue,
                         llvm::function_ref<ResolvedValue(Value)> evaluateValue,
                         Location loc) const override {
    return evaluateTyped(cast<OpT>(op), std::move(resultValue), evaluateValue,
                         loc);
  }

protected:
  virtual FailureOr<evaluator::EvaluatorValuePtr>
  createInitialValueFor(OpT op, Value value,
                     GetPartialValueForTypeFn getPartialValueForType,
                     GetValueForFn getValueFor, Location loc) const {
    return ccreateDefaultInitialValue(value, getPartialValueForType, loc);
  }

  virtual ResolvedValue
  evaluateTyped(OpT op, evaluator::EvaluatorValuePtr resultValue,
                llvm::function_ref<ResolvedValue(Value)> evaluateValue,
                Location loc) const {
    return resolveValueState(std::move(resultValue));
  }
};

/// Base class for operations that only run once all operands are ready.
/// This handles the shared ready/pending/failure/unknown logic so concrete
/// patterns only implement the successful case.
template <typename OpT>
class OpWhenOperandsReadyPattern : public OpPattern<OpT> {
public:
  using OpPattern<OpT>::OpPattern;
  using OpPattern<OpT>::evaluateTyped;

private:
  ResolvedValue evaluate(Operation *op,
                         evaluator::EvaluatorValuePtr resultValue,
                         llvm::function_ref<ResolvedValue(Value)> evaluateValue,
                         Location loc) const final {
    if (resultValue && resultValue->isSettled())
      return resolveValueState(std::move(resultValue));

    SmallVector<evaluator::EvaluatorValuePtr, 4> readyOperands;
    bool existsUnknown = false;
    if (auto early = requireAllOperandsReady(
            op->getOperands(), resultValue, evaluateValue,
            [&] {
              op->emitError() << "failed to resolve "
                              << this->getOperationName() << " operand";
            },
            readyOperands, existsUnknown))
      return *early;
    // If any operand is unknown, the result is unknown too.
    if (existsUnknown)
      return markUnknownAndReturn(std::move(resultValue));

    if (failed(evaluateTyped(cast<OpT>(op), readyOperands, resultValue, loc)))
      return ResolvedValue::failure();
    return resolveValueState(std::move(resultValue));
  }

protected:
  virtual LogicalResult
  evaluateTyped(OpT op, ArrayRef<evaluator::EvaluatorValuePtr> operands,
                evaluator::EvaluatorValuePtr resultValue,
                Location loc) const = 0;
};

/// Base class for single-result ops that can reuse the op's regular folder once
/// all operands are ready.
template <typename OpT>
class OpFolderPattern : public OpWhenOperandsReadyPattern<OpT> {
public:
  using OpWhenOperandsReadyPattern<OpT>::OpWhenOperandsReadyPattern;

protected:
  FailureOr<evaluator::EvaluatorValuePtr>
  createInitialValueFor(OpT op, Value value,
                     typename OperationPattern::GetPartialValueForTypeFn
                         getPartialValueForType,
                     typename OperationPattern::GetValueForFn getValueFor,
                     Location loc) const override {
    SmallVector<Attribute> operandAttrs(op->getNumOperands());
    SmallVector<OpFoldResult> foldResults;
    if (succeeded(op->fold(operandAttrs, foldResults)) &&
        foldResults.size() == 1) {
      if (auto foldedAttr = llvm::dyn_cast<Attribute>(foldResults.front()))
        return success(
            circt::om::evaluator::AttributeValue::get(foldedAttr, loc));
      if (auto foldedValue = llvm::dyn_cast<Value>(foldResults.front()))
        return getValueFor(foldedValue, loc);
    }
    return OpPattern<OpT>::createInitialValueFor(op, value, getPartialValueForType,
                                              getValueFor, loc);
  }

  LogicalResult evaluateTyped(OpT op,
                              ArrayRef<evaluator::EvaluatorValuePtr> operands,
                              evaluator::EvaluatorValuePtr resultValue,
                              Location loc) const override {
    return foldSingleResultOperation(op.getOperation(), operands,
                                     std::move(resultValue));
  }
};

class OperationPatternRegistry {
public:
  const OperationPattern *lookup(Operation *op) const {
    auto it = patternsByOpName.find(op->getName().getStringRef());
    return it == patternsByOpName.end() ? nullptr : it->second;
  }

  template <typename OpT, typename PatternT>
  void addPattern() {
    auto pattern = std::make_unique<PatternT>(OpT::getOperationName());
    const OperationPattern *patternPtr = pattern.get();
    patterns.push_back(std::move(pattern));
    patternsByOpName[OpT::getOperationName()] = patternPtr;
  }

private:
  SmallVector<std::unique_ptr<OperationPattern>> patterns;
  llvm::StringMap<const OperationPattern *> patternsByOpName;
};

void registerOperationPatterns(OperationPatternRegistry &registry);

} // namespace detail
} // namespace om
} // namespace circt

#endif // DIALECT_OM_EVALUATOR_EVALUATORPATTERNS_H
