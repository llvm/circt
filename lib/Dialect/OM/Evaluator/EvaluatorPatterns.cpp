//===- EvaluatorPatterns.cpp - OM evaluator concrete patterns -------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines the concrete OM evaluator operation patterns.
//
//===----------------------------------------------------------------------===//

#include "EvaluatorPatterns.h"
#include <memory>

using namespace mlir;
using namespace circt::om;

namespace circt {
namespace om {
namespace detail {

class ListCreatePattern final
    : public OpWhenOperandsReadyPattern<ListCreateOp> {
public:
  using OpWhenOperandsReadyPattern::OpWhenOperandsReadyPattern;

protected:
  LogicalResult evaluateTyped(ListCreateOp op,
                              ArrayRef<evaluator::EvaluatorValuePtr> operands,
                              evaluator::EvaluatorValuePtr resultValue,
                              Location loc) const override {
    SmallVector<evaluator::EvaluatorValuePtr> values;
    values.reserve(operands.size());
    for (auto operand : operands)
      values.push_back(std::move(operand));

    cast<evaluator::ListValue>(resultValue.get())
        ->setElements(std::move(values));
    return success();
  }
};

class ListConcatPattern final
    : public OpWhenOperandsReadyPattern<ListConcatOp> {
public:
  using OpWhenOperandsReadyPattern::OpWhenOperandsReadyPattern;

protected:
  LogicalResult evaluateTyped(ListConcatOp op,
                              ArrayRef<evaluator::EvaluatorValuePtr> operands,
                              evaluator::EvaluatorValuePtr resultValue,
                              Location loc) const override {
    SmallVector<evaluator::EvaluatorValuePtr> values;
    for (const auto &operand : operands) {
      auto *subListValue = getReadyAs<evaluator::ListValue>(operand);
      llvm::append_range(values, subListValue->getElements());
    }

    cast<evaluator::ListValue>(resultValue.get())
        ->setElements(std::move(values));
    return success();
  }
};

class FrozenBasePathCreatePattern final
    : public OpWhenOperandsReadyPattern<FrozenBasePathCreateOp> {
public:
  using OpWhenOperandsReadyPattern::OpWhenOperandsReadyPattern;

protected:
  FailureOr<evaluator::EvaluatorValuePtr>
  createInitialValueFor(FrozenBasePathCreateOp op, Value value,
                        GetPartialValueForTypeFn getPartialValueForType,
                        GetValueForFn getValueFor,
                        Location loc) const override {
    return success(
        std::make_shared<evaluator::BasePathValue>(op.getPathAttr(), loc));
  }

  LogicalResult evaluateTyped(FrozenBasePathCreateOp op,
                              ArrayRef<evaluator::EvaluatorValuePtr> operands,
                              evaluator::EvaluatorValuePtr resultValue,
                              Location loc) const override {
    assert(operands.size() == 1 &&
           "expected one operand for frozenbasepath_create");

    auto *basePathValue =
        getReadyAs<evaluator::BasePathValue>(operands.front());
    cast<evaluator::BasePathValue>(resultValue.get())
        ->setBasepath(*basePathValue);
    return success();
  }
};

class FrozenPathCreatePattern final
    : public OpWhenOperandsReadyPattern<FrozenPathCreateOp> {
public:
  using OpWhenOperandsReadyPattern::OpWhenOperandsReadyPattern;

protected:
  FailureOr<evaluator::EvaluatorValuePtr>
  createInitialValueFor(FrozenPathCreateOp pathOp, Value value,
                        GetPartialValueForTypeFn getPartialValueForType,
                        GetValueForFn getValueFor,
                        Location loc) const override {
    return success(std::make_shared<evaluator::PathValue>(
        pathOp.getTargetKindAttr(), pathOp.getPathAttr(),
        pathOp.getModuleAttr(), pathOp.getRefAttr(), pathOp.getFieldAttr(),
        loc));
  }

  LogicalResult evaluateTyped(FrozenPathCreateOp op,
                              ArrayRef<evaluator::EvaluatorValuePtr> operands,
                              evaluator::EvaluatorValuePtr resultValue,
                              Location loc) const override {
    assert(operands.size() == 1 &&
           "expected one operand for frozenpath_create");

    auto *basePathValue =
        getReadyAs<evaluator::BasePathValue>(operands.front());
    cast<evaluator::PathValue>(resultValue.get())->setBasepath(*basePathValue);
    return success();
  }
};

class AnyCastPattern final : public OpPattern<AnyCastOp> {
public:
  using OpPattern::OpPattern;

protected:
  FailureOr<evaluator::EvaluatorValuePtr>
  createInitialValueFor(AnyCastOp op, Value value,
                        GetPartialValueForTypeFn getPartialValueForType,
                        GetValueForFn getValueFor,
                        Location loc) const override {
    return getValueFor(op.getInput(), loc);
  }
};

class FrozenEmptyPathPattern final : public OpPattern<FrozenEmptyPathOp> {
public:
  using OpPattern::OpPattern;

protected:
  FailureOr<evaluator::EvaluatorValuePtr>
  createInitialValueFor(FrozenEmptyPathOp op, Value value,
                        GetPartialValueForTypeFn getPartialValueForType,
                        GetValueForFn getValueFor,
                        Location loc) const override {
    return success(std::make_shared<evaluator::PathValue>(
        evaluator::PathValue::getEmptyPath(loc)));
  }
};

void registerOperationPatterns(OperationPatternRegistry &registry) {
  registry.addPattern<ConstantOp, OpFolderPattern<ConstantOp>>();
  registry.addPattern<AnyCastOp, AnyCastPattern>();
  registry.addPattern<FrozenEmptyPathOp, FrozenEmptyPathPattern>();
  registry.addPattern<IntegerAddOp, OpFolderPattern<IntegerAddOp>>();
  registry.addPattern<IntegerMulOp, OpFolderPattern<IntegerMulOp>>();
  registry.addPattern<IntegerShrOp, OpFolderPattern<IntegerShrOp>>();
  registry.addPattern<IntegerShlOp, OpFolderPattern<IntegerShlOp>>();
  registry.addPattern<ListCreateOp, ListCreatePattern>();
  registry.addPattern<ListConcatOp, ListConcatPattern>();
  registry.addPattern<StringConcatOp, OpFolderPattern<StringConcatOp>>();
  registry.addPattern<PropEqOp, OpFolderPattern<PropEqOp>>();
  registry.addPattern<FrozenBasePathCreateOp, FrozenBasePathCreatePattern>();
  registry.addPattern<FrozenPathCreateOp, FrozenPathCreatePattern>();
}

} // namespace detail
} // namespace om
} // namespace circt
