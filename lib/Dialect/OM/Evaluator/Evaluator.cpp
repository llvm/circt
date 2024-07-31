//===- Evaluator.cpp - Object Model dialect evaluator ---------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements the Object Model dialect Evaluator.
//
//===----------------------------------------------------------------------===//

#include "circt/Dialect/OM/Evaluator/Evaluator.h"
#include "mlir/IR/BuiltinAttributeInterfaces.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/SymbolTable.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "om-evaluator"

using namespace mlir;
using namespace circt::om;

/// Construct an Evaluator with an IR module.
circt::om::Evaluator::Evaluator(ModuleOp mod) : symbolTable(mod) {}

/// Get the Module this Evaluator is built from.
ModuleOp circt::om::Evaluator::getModule() {
  return cast<ModuleOp>(symbolTable.getOp());
}

SmallVector<evaluator::EvaluatorValuePtr>
circt::om::getEvaluatorValuesFromAttributes(MLIRContext *context,
                                            ArrayRef<Attribute> attributes) {
  SmallVector<evaluator::EvaluatorValuePtr> values;
  values.reserve(attributes.size());
  for (auto attr : attributes)
    values.push_back(std::make_shared<evaluator::AttributeValue>(attr));
  return values;
}

LogicalResult circt::om::evaluator::EvaluatorValue::finalize() {
  using namespace evaluator;
  // Early return if already finalized.
  if (finalized)
    return success();
  // Enable the flag to avoid infinite recursions.
  finalized = true;
  assert(isFullyEvaluated());
  return llvm::TypeSwitch<EvaluatorValue *, LogicalResult>(this)
      .Case<AttributeValue, ObjectValue, ListValue, MapValue, ReferenceValue,
            TupleValue, BasePathValue, PathValue>(
          [](auto v) { return v->finalizeImpl(); });
}

Type circt::om::evaluator::EvaluatorValue::getType() const {
  return llvm::TypeSwitch<const EvaluatorValue *, Type>(this)
      .Case<AttributeValue>([](auto *attr) -> Type { return attr->getType(); })
      .Case<ObjectValue>([](auto *object) { return object->getObjectType(); })
      .Case<ListValue>([](auto *list) { return list->getListType(); })
      .Case<MapValue>([](auto *map) { return map->getMapType(); })
      .Case<ReferenceValue>([](auto *ref) { return ref->getValueType(); })
      .Case<TupleValue>([](auto *tuple) { return tuple->getTupleType(); })
      .Case<BasePathValue>(
          [this](auto *tuple) { return FrozenBasePathType::get(ctx); })
      .Case<PathValue>(
          [this](auto *tuple) { return FrozenPathType::get(ctx); });
}

FailureOr<evaluator::EvaluatorValuePtr>
circt::om::Evaluator::getPartiallyEvaluatedValue(Type type, Location loc) {
  using namespace circt::om::evaluator;

  return TypeSwitch<mlir::Type, FailureOr<evaluator::EvaluatorValuePtr>>(type)
      .Case([&](circt::om::MapType type) {
        evaluator::EvaluatorValuePtr result =
            std::make_shared<evaluator::MapValue>(type, loc);
        return success(result);
      })
      .Case([&](circt::om::ListType type) {
        evaluator::EvaluatorValuePtr result =
            std::make_shared<evaluator::ListValue>(type, loc);
        return success(result);
      })
      .Case([&](mlir::TupleType type) {
        evaluator::EvaluatorValuePtr result =
            std::make_shared<evaluator::TupleValue>(type, loc);
        return success(result);
      })

      .Case([&](circt::om::ClassType type)
                -> FailureOr<evaluator::EvaluatorValuePtr> {
        ClassOp cls =
            symbolTable.lookup<ClassOp>(type.getClassName().getValue());
        if (!cls)
          return symbolTable.getOp()->emitError("unknown class name ")
                 << type.getClassName();

        evaluator::EvaluatorValuePtr result =
            std::make_shared<evaluator::ObjectValue>(cls, loc);

        return success(result);
      })
      .Default([&](auto type) { return failure(); });
}

FailureOr<evaluator::EvaluatorValuePtr> circt::om::Evaluator::getOrCreateValue(
    Value value, ActualParameters actualParams, Location loc) {
  auto it = objects.find({value, actualParams});
  if (it != objects.end()) {
    auto evalVal = it->second;
    evalVal->setLocIfUnknown(loc);
    return evalVal;
  }

  FailureOr<evaluator::EvaluatorValuePtr> result =
      TypeSwitch<Value, FailureOr<evaluator::EvaluatorValuePtr>>(value)
          .Case([&](BlockArgument arg) {
            auto val = (*actualParams)[arg.getArgNumber()];
            val->setLoc(loc);
            return val;
          })
          .Case([&](OpResult result) {
            return TypeSwitch<Operation *,
                              FailureOr<evaluator::EvaluatorValuePtr>>(
                       result.getDefiningOp())
                .Case([&](ConstantOp op) {
                  return evaluateConstant(op, actualParams, loc);
                })
                .Case([&](IntegerBinaryArithmeticOp op) {
                  // Create a partially evaluated AttributeValue of
                  // om::IntegerType in case we need to delay evaluation.
                  evaluator::EvaluatorValuePtr result =
                      std::make_shared<evaluator::AttributeValue>(
                          op.getResult().getType(), loc);
                  return success(result);
                })
                .Case<ObjectFieldOp>([&](auto op) {
                  // Create a reference value since the value pointed by object
                  // field op is not created yet.
                  evaluator::EvaluatorValuePtr result =
                      std::make_shared<evaluator::ReferenceValue>(
                          value.getType(), loc);
                  return success(result);
                })
                .Case<AnyCastOp>([&](AnyCastOp op) {
                  return getOrCreateValue(op.getInput(), actualParams, loc);
                })
                .Case<FrozenBasePathCreateOp>([&](FrozenBasePathCreateOp op) {
                  evaluator::EvaluatorValuePtr result =
                      std::make_shared<evaluator::BasePathValue>(
                          op.getPathAttr(), loc);
                  return success(result);
                })
                .Case<FrozenPathCreateOp>([&](FrozenPathCreateOp op) {
                  evaluator::EvaluatorValuePtr result =
                      std::make_shared<evaluator::PathValue>(
                          op.getTargetKindAttr(), op.getPathAttr(),
                          op.getModuleAttr(), op.getRefAttr(),
                          op.getFieldAttr(), loc);
                  return success(result);
                })
                .Case<FrozenEmptyPathOp>([&](FrozenEmptyPathOp op) {
                  evaluator::EvaluatorValuePtr result =
                      std::make_shared<evaluator::PathValue>(
                          evaluator::PathValue::getEmptyPath(loc));
                  return success(result);
                })
                .Case<ListCreateOp, TupleCreateOp, MapCreateOp, ObjectFieldOp>(
                    [&](auto op) {
                      return getPartiallyEvaluatedValue(op.getType(), loc);
                    })
                .Case<ObjectOp>([&](auto op) {
                  return getPartiallyEvaluatedValue(op.getType(), op.getLoc());
                })
                .Default([&](Operation *op) {
                  auto error = op->emitError("unable to evaluate value");
                  error.attachNote() << "value: " << value;
                  return error;
                });
          });
  if (failed(result))
    return result;

  objects[{value, actualParams}] = result.value();
  return result;
}

FailureOr<evaluator::EvaluatorValuePtr>
circt::om::Evaluator::evaluateObjectInstance(StringAttr className,
                                             ActualParameters actualParams,
                                             Location loc,
                                             ObjectKey instanceKey) {
  ClassOp cls = symbolTable.lookup<ClassOp>(className);
  if (!cls)
    return symbolTable.getOp()->emitError("unknown class name ") << className;

  auto formalParamNames = cls.getFormalParamNames().getAsRange<StringAttr>();
  auto formalParamTypes = cls.getBodyBlock()->getArgumentTypes();

  // Verify the actual parameters are the right size and types for this class.
  if (actualParams->size() != formalParamTypes.size()) {
    auto error = cls.emitError("actual parameter list length (")
                 << actualParams->size() << ") does not match formal "
                 << "parameter list length (" << formalParamTypes.size() << ")";
    auto &diag = error.attachNote() << "actual parameters: ";
    // FIXME: `diag << actualParams` doesn't work for some reason.
    bool isFirst = true;
    for (const auto &param : *actualParams) {
      if (isFirst)
        isFirst = false;
      else
        diag << ", ";
      diag << param;
    }
    error.attachNote(cls.getLoc()) << "formal parameters: " << formalParamTypes;
    return error;
  }

  // Verify the actual parameter types match.
  for (auto [actualParam, formalParamName, formalParamType] :
       llvm::zip(*actualParams, formalParamNames, formalParamTypes)) {
    if (!actualParam || !actualParam.get())
      return cls.emitError("actual parameter for ")
             << formalParamName << " is null";

    // Subtyping: if formal param is any type, any actual param may be passed.
    if (isa<AnyType>(formalParamType))
      continue;

    Type actualParamType = actualParam->getType();

    assert(actualParamType && "actualParamType must be non-null!");

    if (actualParamType != formalParamType) {
      auto error = cls.emitError("actual parameter for ")
                   << formalParamName << " has invalid type";
      error.attachNote() << "actual parameter: " << *actualParam;
      error.attachNote() << "format parameter type: " << formalParamType;
      return error;
    }
  }

  // Instantiate the fields.
  evaluator::ObjectFields fields;

  auto *context = cls.getContext();
  for (auto &op : cls.getOps())
    for (auto result : op.getResults()) {
      // Allocate the value, with unknown loc. It will be later set when
      // evaluating the fields.
      if (failed(
              getOrCreateValue(result, actualParams, UnknownLoc::get(context))))
        return failure();
      // Add to the worklist.
      worklist.push({result, actualParams});
    }

  for (auto field : cls.getOps<ClassFieldOp>()) {
    StringAttr name = field.getNameAttr();
    Value value = field.getValue();
    FailureOr<evaluator::EvaluatorValuePtr> result =
        evaluateValue(value, actualParams, field.getLoc());
    if (failed(result))
      return result;

    fields[name] = result.value();
  }

  // If the there is an instance, we must update the object value.
  if (instanceKey.first) {
    auto result =
        getOrCreateValue(instanceKey.first, instanceKey.second, loc).value();
    auto *object = llvm::cast<evaluator::ObjectValue>(result.get());
    object->setFields(std::move(fields));
    return result;
  }

  // If it's external call, just allocate new ObjectValue.
  evaluator::EvaluatorValuePtr result =
      std::make_shared<evaluator::ObjectValue>(cls, fields, loc);
  return result;
}

/// Instantiate an Object with its class name and actual parameters.
FailureOr<std::shared_ptr<evaluator::EvaluatorValue>>
circt::om::Evaluator::instantiate(
    StringAttr className, ArrayRef<evaluator::EvaluatorValuePtr> actualParams) {
  ClassOp cls = symbolTable.lookup<ClassOp>(className);
  if (!cls)
    return symbolTable.getOp()->emitError("unknown class name ") << className;

  auto parameters =
      std::make_unique<SmallVector<std::shared_ptr<evaluator::EvaluatorValue>>>(
          actualParams);

  actualParametersBuffers.push_back(std::move(parameters));

  auto loc = cls.getLoc();
  auto result = evaluateObjectInstance(
      className, actualParametersBuffers.back().get(), loc);

  if (failed(result))
    return failure();

  // `evaluateObjectInstance` has populated the worklist. Continue evaluations
  // unless there is a partially evaluated value.
  while (!worklist.empty()) {
    auto [value, args] = worklist.front();
    worklist.pop();

    auto result = evaluateValue(value, args, loc);

    if (failed(result))
      return failure();

    // It's possible that the value is not fully evaluated.
    if (!result.value()->isFullyEvaluated())
      worklist.push({value, args});
  }

  auto &object = result.value();
  // Finalize the value. This will eliminate intermidiate ReferenceValue used as
  // a placeholder in the initialization.
  if (failed(object->finalize()))
    return cls.emitError() << "failed to finalize evaluation. Probably the "
                              "class contains a dataflow cycle";
  return object;
}

FailureOr<evaluator::EvaluatorValuePtr>
circt::om::Evaluator::evaluateValue(Value value, ActualParameters actualParams,
                                    Location loc) {
  auto evaluatorValue = getOrCreateValue(value, actualParams, loc).value();

  // Return if the value is already evaluated.
  if (evaluatorValue->isFullyEvaluated())
    return evaluatorValue;

  return llvm::TypeSwitch<Value, FailureOr<evaluator::EvaluatorValuePtr>>(value)
      .Case([&](BlockArgument arg) {
        return evaluateParameter(arg, actualParams, loc);
      })
      .Case([&](OpResult result) {
        return TypeSwitch<Operation *, FailureOr<evaluator::EvaluatorValuePtr>>(
                   result.getDefiningOp())
            .Case([&](ConstantOp op) {
              return evaluateConstant(op, actualParams, loc);
            })
            .Case([&](IntegerBinaryArithmeticOp op) {
              return evaluateIntegerBinaryArithmetic(op, actualParams, loc);
            })
            .Case([&](ObjectOp op) {
              return evaluateObjectInstance(op, actualParams);
            })
            .Case([&](ObjectFieldOp op) {
              return evaluateObjectField(op, actualParams, loc);
            })
            .Case([&](ListCreateOp op) {
              return evaluateListCreate(op, actualParams, loc);
            })
            .Case([&](TupleCreateOp op) {
              return evaluateTupleCreate(op, actualParams, loc);
            })
            .Case([&](TupleGetOp op) {
              return evaluateTupleGet(op, actualParams, loc);
            })
            .Case([&](AnyCastOp op) {
              return evaluateValue(op.getInput(), actualParams, loc);
            })
            .Case([&](MapCreateOp op) {
              return evaluateMapCreate(op, actualParams, loc);
            })
            .Case([&](FrozenBasePathCreateOp op) {
              return evaluateBasePathCreate(op, actualParams, loc);
            })
            .Case([&](FrozenPathCreateOp op) {
              return evaluatePathCreate(op, actualParams, loc);
            })
            .Case([&](FrozenEmptyPathOp op) {
              return evaluateEmptyPath(op, actualParams, loc);
            })
            .Default([&](Operation *op) {
              auto error = op->emitError("unable to evaluate value");
              error.attachNote() << "value: " << value;
              return error;
            });
      });
}

/// Evaluator dispatch function for parameters.
FailureOr<evaluator::EvaluatorValuePtr> circt::om::Evaluator::evaluateParameter(
    BlockArgument formalParam, ActualParameters actualParams, Location loc) {
  auto val = (*actualParams)[formalParam.getArgNumber()];
  val->setLoc(loc);
  return success(val);
}

/// Evaluator dispatch function for constants.
FailureOr<circt::om::evaluator::EvaluatorValuePtr>
circt::om::Evaluator::evaluateConstant(ConstantOp op,
                                       ActualParameters actualParams,
                                       Location loc) {
  return success(std::make_shared<circt::om::evaluator::AttributeValue>(
      op.getValue(), loc));
}

// Evaluator dispatch function for integer binary arithmetic.
FailureOr<EvaluatorValuePtr>
circt::om::Evaluator::evaluateIntegerBinaryArithmetic(
    IntegerBinaryArithmeticOp op, ActualParameters actualParams, Location loc) {
  // Get the op's EvaluatorValue handle, in case it hasn't been evaluated yet.
  auto handle = getOrCreateValue(op.getResult(), actualParams, loc);

  // If it's fully evaluated, we can return it.
  if (handle.value()->isFullyEvaluated())
    return handle;

  // Evaluate operands if necessary, and return the partially evaluated value if
  // they aren't ready.
  auto lhsResult = evaluateValue(op.getLhs(), actualParams, loc);
  if (failed(lhsResult))
    return lhsResult;
  if (!lhsResult.value()->isFullyEvaluated())
    return handle;

  auto rhsResult = evaluateValue(op.getRhs(), actualParams, loc);
  if (failed(rhsResult))
    return rhsResult;
  if (!rhsResult.value()->isFullyEvaluated())
    return handle;

  // Extract the integer attributes.
  auto extractAttr = [](evaluator::EvaluatorValue *value) {
    return std::move(
        llvm::TypeSwitch<evaluator::EvaluatorValue *, om::IntegerAttr>(value)
            .Case([](evaluator::AttributeValue *val) {
              return val->getAs<om::IntegerAttr>();
            })
            .Case([](evaluator::ReferenceValue *val) {
              return cast<evaluator::AttributeValue>(
                         val->getStrippedValue()->get())
                  ->getAs<om::IntegerAttr>();
            }));
  };

  om::IntegerAttr lhs = extractAttr(lhsResult.value().get());
  om::IntegerAttr rhs = extractAttr(rhsResult.value().get());
  assert(lhs && rhs &&
         "expected om::IntegerAttr for IntegerBinaryArithmeticOp operands");

  // Extend values if necessary to match bitwidth. Most interesting arithmetic
  // on APSInt asserts that both operands are the same bitwidth, but the
  // IntegerAttrs we are working with may have used the smallest necessary
  // bitwidth to represent the number they hold, and won't necessarily match.
  APSInt lhsVal = lhs.getValue().getAPSInt();
  APSInt rhsVal = rhs.getValue().getAPSInt();
  if (lhsVal.getBitWidth() > rhsVal.getBitWidth())
    rhsVal = rhsVal.extend(lhsVal.getBitWidth());
  else if (rhsVal.getBitWidth() > lhsVal.getBitWidth())
    lhsVal = lhsVal.extend(rhsVal.getBitWidth());

  // Perform arbitrary precision signed integer binary arithmetic.
  FailureOr<APSInt> result = op.evaluateIntegerOperation(lhsVal, rhsVal);

  if (failed(result))
    return op->emitError("failed to evaluate integer operation");

  // Package the result as a new om::IntegerAttr.
  MLIRContext *ctx = op->getContext();
  auto resultAttr =
      om::IntegerAttr::get(ctx, mlir::IntegerAttr::get(ctx, result.value()));

  // Finalize the op result value.
  auto *handleValue = cast<evaluator::AttributeValue>(handle.value().get());
  auto resultStatus = handleValue->setAttr(resultAttr);
  if (failed(resultStatus))
    return resultStatus;

  auto finalizeStatus = handleValue->finalize();
  if (failed(finalizeStatus))
    return finalizeStatus;

  return handle;
}

/// Evaluator dispatch function for Object instances.
FailureOr<circt::om::Evaluator::ActualParameters>
circt::om::Evaluator::createParametersFromOperands(
    ValueRange range, ActualParameters actualParams, Location loc) {
  // Create an unique storage to store parameters.
  auto parameters = std::make_unique<
      SmallVector<std::shared_ptr<evaluator::EvaluatorValue>>>();

  // Collect operands' evaluator values in the current instantiation context.
  for (auto input : range) {
    auto inputResult = getOrCreateValue(input, actualParams, loc);
    if (failed(inputResult))
      return failure();
    parameters->push_back(inputResult.value());
  }

  actualParametersBuffers.push_back(std::move(parameters));
  return actualParametersBuffers.back().get();
}

/// Evaluator dispatch function for Object instances.
FailureOr<evaluator::EvaluatorValuePtr>
circt::om::Evaluator::evaluateObjectInstance(ObjectOp op,
                                             ActualParameters actualParams) {
  auto loc = op.getLoc();
  if (isFullyEvaluated({op, actualParams}))
    return getOrCreateValue(op, actualParams, loc);

  auto params =
      createParametersFromOperands(op.getOperands(), actualParams, loc);
  if (failed(params))
    return failure();
  return evaluateObjectInstance(op.getClassNameAttr(), params.value(), loc,
                                {op, actualParams});
}

/// Evaluator dispatch function for Object fields.
FailureOr<evaluator::EvaluatorValuePtr>
circt::om::Evaluator::evaluateObjectField(ObjectFieldOp op,
                                          ActualParameters actualParams,
                                          Location loc) {
  // Evaluate the Object itself, in case it hasn't been evaluated yet.
  FailureOr<evaluator::EvaluatorValuePtr> currentObjectResult =
      evaluateValue(op.getObject(), actualParams, loc);
  if (failed(currentObjectResult))
    return currentObjectResult;

  auto *currentObject =
      llvm::cast<evaluator::ObjectValue>(currentObjectResult.value().get());

  auto objectFieldValue = getOrCreateValue(op, actualParams, loc).value();

  // Iteratively access nested fields through the path until we reach the final
  // field in the path.
  evaluator::EvaluatorValuePtr finalField;
  for (auto field : op.getFieldPath().getAsRange<FlatSymbolRefAttr>()) {
    // `currentObject` might no be fully evaluated.
    if (!currentObject->getFields().contains(field.getAttr()))
      return objectFieldValue;

    auto currentField = currentObject->getField(field.getAttr());
    finalField = currentField.value();
    if (auto *nextObject =
            llvm::dyn_cast<evaluator::ObjectValue>(finalField.get()))
      currentObject = nextObject;
  }

  // Update the reference.
  llvm::cast<evaluator::ReferenceValue>(objectFieldValue.get())
      ->setValue(finalField);

  // Return the field being accessed.
  return objectFieldValue;
}

/// Evaluator dispatch function for List creation.
FailureOr<evaluator::EvaluatorValuePtr>
circt::om::Evaluator::evaluateListCreate(ListCreateOp op,
                                         ActualParameters actualParams,
                                         Location loc) {
  // Evaluate the Object itself, in case it hasn't been evaluated yet.
  SmallVector<evaluator::EvaluatorValuePtr> values;
  auto list = getOrCreateValue(op, actualParams, loc);
  for (auto operand : op.getOperands()) {
    auto result = evaluateValue(operand, actualParams, loc);
    if (failed(result))
      return result;
    if (!result.value()->isFullyEvaluated())
      return list;
    values.push_back(result.value());
  }

  // Return the list.
  llvm::cast<evaluator::ListValue>(list.value().get())
      ->setElements(std::move(values));
  return list;
}

/// Evaluator dispatch function for Tuple creation.
FailureOr<evaluator::EvaluatorValuePtr>
circt::om::Evaluator::evaluateTupleCreate(TupleCreateOp op,
                                          ActualParameters actualParams,
                                          Location loc) {
  SmallVector<evaluator::EvaluatorValuePtr> values;
  for (auto operand : op.getOperands()) {
    auto result = evaluateValue(operand, actualParams, loc);
    if (failed(result))
      return result;
    values.push_back(result.value());
  }

  // Return the tuple.
  auto val = getOrCreateValue(op, actualParams, loc);
  llvm::cast<evaluator::TupleValue>(val.value().get())
      ->setElements(std::move(values));
  return val;
}

/// Evaluator dispatch function for List creation.
FailureOr<evaluator::EvaluatorValuePtr> circt::om::Evaluator::evaluateTupleGet(
    TupleGetOp op, ActualParameters actualParams, Location loc) {
  auto tuple = evaluateValue(op.getInput(), actualParams, loc);
  if (failed(tuple))
    return tuple;
  evaluator::EvaluatorValuePtr result =
      cast<evaluator::TupleValue>(tuple.value().get())
          ->getElements()[op.getIndex()];
  return result;
}

/// Evaluator dispatch function for Map creation.
FailureOr<evaluator::EvaluatorValuePtr> circt::om::Evaluator::evaluateMapCreate(
    MapCreateOp op, ActualParameters actualParams, Location loc) {
  // Evaluate the Object itself, in case it hasn't been evaluated yet.
  DenseMap<Attribute, evaluator::EvaluatorValuePtr> elements;
  auto valueResult = getOrCreateValue(op, actualParams, loc).value();
  for (auto operand : op.getOperands()) {
    auto result = evaluateValue(operand, actualParams, loc);
    if (failed(result))
      return result;
    // The result is a tuple.
    auto &value = result.value();
    if (!value->isFullyEvaluated())
      return valueResult;
    const auto &element =
        llvm::cast<evaluator::TupleValue>(value.get())->getElements();
    assert(element.size() == 2);
    auto attr =
        llvm::cast<evaluator::AttributeValue>(element[0].get())->getAttr();
    if (!elements.insert({attr, element[1]}).second)
      return op.emitError() << "map contains duplicated keys";
  }

  // Return the Map.
  llvm::cast<evaluator::MapValue>(valueResult.get())
      ->setElements(std::move(elements));
  return valueResult;
}

FailureOr<evaluator::EvaluatorValuePtr>
circt::om::Evaluator::evaluateBasePathCreate(FrozenBasePathCreateOp op,
                                             ActualParameters actualParams,
                                             Location loc) {
  // Evaluate the Object itself, in case it hasn't been evaluated yet.
  auto valueResult = getOrCreateValue(op, actualParams, loc).value();
  auto *path = llvm::cast<evaluator::BasePathValue>(valueResult.get());
  auto result = evaluateValue(op.getBasePath(), actualParams, loc);
  if (failed(result))
    return result;
  auto &value = result.value();
  if (!value->isFullyEvaluated())
    return valueResult;
  path->setBasepath(*llvm::cast<evaluator::BasePathValue>(value.get()));
  return valueResult;
}

FailureOr<evaluator::EvaluatorValuePtr>
circt::om::Evaluator::evaluatePathCreate(FrozenPathCreateOp op,
                                         ActualParameters actualParams,
                                         Location loc) {
  // Evaluate the Object itself, in case it hasn't been evaluated yet.
  auto valueResult = getOrCreateValue(op, actualParams, loc).value();
  auto *path = llvm::cast<evaluator::PathValue>(valueResult.get());
  auto result = evaluateValue(op.getBasePath(), actualParams, loc);
  if (failed(result))
    return result;
  auto &value = result.value();
  if (!value->isFullyEvaluated())
    return valueResult;
  path->setBasepath(*llvm::cast<evaluator::BasePathValue>(value.get()));
  return valueResult;
}

FailureOr<evaluator::EvaluatorValuePtr> circt::om::Evaluator::evaluateEmptyPath(
    FrozenEmptyPathOp op, ActualParameters actualParams, Location loc) {
  auto valueResult = getOrCreateValue(op, actualParams, loc).value();
  return valueResult;
}

//===----------------------------------------------------------------------===//
// ObjectValue
//===----------------------------------------------------------------------===//

/// Get a field of the Object by name.
FailureOr<EvaluatorValuePtr>
circt::om::evaluator::ObjectValue::getField(StringAttr name) {
  auto field = fields.find(name);
  if (field == fields.end())
    return cls.emitError("field ") << name << " does not exist";
  return success(fields[name]);
}

/// Get an ArrayAttr with the names of the fields in the Object. Sort the fields
/// so there is always a stable order.
ArrayAttr circt::om::Object::getFieldNames() {
  SmallVector<Attribute> fieldNames;
  for (auto &f : fields)
    fieldNames.push_back(f.first);

  llvm::sort(fieldNames, [](Attribute a, Attribute b) {
    return cast<StringAttr>(a).getValue() < cast<StringAttr>(b).getValue();
  });

  return ArrayAttr::get(cls.getContext(), fieldNames);
}

LogicalResult circt::om::evaluator::ObjectValue::finalizeImpl() {
  for (auto &&[e, value] : fields)
    if (failed(finalizeEvaluatorValue(value)))
      return failure();

  return success();
}

//===----------------------------------------------------------------------===//
// MapValue
//===----------------------------------------------------------------------===//

/// Return an array of keys in the ascending order.
ArrayAttr circt::om::evaluator::MapValue::getKeys() {
  SmallVector<Attribute> attrs;
  for (auto &[key, _] : elements)
    attrs.push_back(key);

  std::sort(attrs.begin(), attrs.end(), [](Attribute l, Attribute r) {
    if (auto lInt = dyn_cast<mlir::IntegerAttr>(l))
      if (auto rInt = dyn_cast<mlir::IntegerAttr>(r))
        return lInt.getValue().ult(rInt.getValue());

    assert(isa<StringAttr>(l) && isa<StringAttr>(r) &&
           "key type should be integer or string");
    return cast<StringAttr>(l).getValue() < cast<StringAttr>(r).getValue();
  });

  return ArrayAttr::get(type.getContext(), attrs);
}

LogicalResult circt::om::evaluator::MapValue::finalizeImpl() {
  for (auto &&[e, value] : elements)
    if (failed(finalizeEvaluatorValue(value)))
      return failure();
  return success();
}

//===----------------------------------------------------------------------===//
// ReferenceValue
//===----------------------------------------------------------------------===//

LogicalResult circt::om::evaluator::ReferenceValue::finalizeImpl() {
  auto result = getStrippedValue();
  if (failed(result))
    return result;
  value = std::move(result.value());
  return success();
}

//===----------------------------------------------------------------------===//
// ListValue
//===----------------------------------------------------------------------===//

LogicalResult circt::om::evaluator::ListValue::finalizeImpl() {
  for (auto &value : elements) {
    if (failed(finalizeEvaluatorValue(value)))
      return failure();
  }
  return success();
}

//===----------------------------------------------------------------------===//
// BasePathValue
//===----------------------------------------------------------------------===//

evaluator::BasePathValue::BasePathValue(MLIRContext *context)
    : EvaluatorValue(context, Kind::BasePath, UnknownLoc::get(context)),
      path(PathAttr::get(context, {})) {
  markFullyEvaluated();
}

evaluator::BasePathValue::BasePathValue(PathAttr path, Location loc)
    : EvaluatorValue(path.getContext(), Kind::BasePath, loc), path(path) {}

PathAttr evaluator::BasePathValue::getPath() const {
  assert(isFullyEvaluated());
  return path;
}

void evaluator::BasePathValue::setBasepath(const BasePathValue &basepath) {
  assert(!isFullyEvaluated());
  auto newPath = llvm::to_vector(basepath.path.getPath());
  auto oldPath = path.getPath();
  newPath.append(oldPath.begin(), oldPath.end());
  path = PathAttr::get(path.getContext(), newPath);
  markFullyEvaluated();
}

//===----------------------------------------------------------------------===//
// PathValue
//===----------------------------------------------------------------------===//

evaluator::PathValue::PathValue(TargetKindAttr targetKind, PathAttr path,
                                StringAttr module, StringAttr ref,
                                StringAttr field, Location loc)
    : EvaluatorValue(loc.getContext(), Kind::Path, loc), targetKind(targetKind),
      path(path), module(module), ref(ref), field(field) {}

evaluator::PathValue evaluator::PathValue::getEmptyPath(Location loc) {
  PathValue path(nullptr, nullptr, nullptr, nullptr, nullptr, loc);
  path.markFullyEvaluated();
  return path;
}

StringAttr evaluator::PathValue::getAsString() const {
  // If the module is null, then this is a path to a deleted object.
  if (!targetKind)
    return StringAttr::get(getContext(), "OMDeleted:");
  SmallString<64> result;
  switch (targetKind.getValue()) {
  case TargetKind::DontTouch:
    result += "OMDontTouchedReferenceTarget";
    break;
  case TargetKind::Instance:
    result += "OMInstanceTarget";
    break;
  case TargetKind::MemberInstance:
    result += "OMMemberInstanceTarget";
    break;
  case TargetKind::MemberReference:
    result += "OMMemberReferenceTarget";
    break;
  case TargetKind::Reference:
    result += "OMReferenceTarget";
    break;
  }
  result += ":~";
  if (!path.getPath().empty())
    result += path.getPath().front().module;
  else
    result += module.getValue();
  result += '|';
  for (const auto &elt : path) {
    result += elt.module.getValue();
    result += '/';
    result += elt.instance.getValue();
    result += ':';
  }
  if (!module.getValue().empty())
    result += module.getValue();
  if (!ref.getValue().empty()) {
    result += '>';
    result += ref.getValue();
  }
  if (!field.getValue().empty())
    result += field.getValue();
  return StringAttr::get(field.getContext(), result);
}

void evaluator::PathValue::setBasepath(const BasePathValue &basepath) {
  assert(!isFullyEvaluated());
  auto newPath = llvm::to_vector(basepath.getPath().getPath());
  auto oldPath = path.getPath();
  newPath.append(oldPath.begin(), oldPath.end());
  path = PathAttr::get(path.getContext(), newPath);
  markFullyEvaluated();
}

//===----------------------------------------------------------------------===//
// AttributeValue
//===----------------------------------------------------------------------===//

LogicalResult circt::om::evaluator::AttributeValue::setAttr(Attribute attr) {
  if (cast<TypedAttr>(attr).getType() != this->type)
    return mlir::emitError(getLoc(), "cannot set AttributeValue of type ")
           << this->type << " to Attribute " << attr;
  if (isFullyEvaluated())
    return mlir::emitError(
        getLoc(),
        "cannot set AttributeValue that has already been fully evaluated");
  this->attr = attr;
  markFullyEvaluated();
  return success();
}

LogicalResult circt::om::evaluator::AttributeValue::finalizeImpl() {
  if (!isFullyEvaluated())
    return mlir::emitError(
        getLoc(), "cannot finalize AttributeValue that is not fully evaluated");
  return success();
}
