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
            TupleValue>([](auto v) { return v->finalizeImpl(); });
}

Type circt::om::evaluator::EvaluatorValue::getType() const {
  return llvm::TypeSwitch<const EvaluatorValue *, Type>(this)
      .Case<AttributeValue>([](auto *attr) -> Type {
        return cast<TypedAttr>(attr->getAttr()).getType();
      })
      .Case<ObjectValue>([](auto *object) { return object->getObjectType(); })
      .Case<ListValue>([](auto *list) { return list->getListType(); })
      .Case<MapValue>([](auto *map) { return map->getMapType(); })
      .Case<ReferenceValue>([](auto *ref) { return ref->getValueType(); })
      .Case<TupleValue>([](auto *tuple) { return tuple->getTupleType(); });
}

FailureOr<evaluator::EvaluatorValuePtr>
circt::om::Evaluator::getPartiallyEvaluatedValue(Type type) {
  using namespace circt::om::evaluator;

  return TypeSwitch<mlir::Type, FailureOr<evaluator::EvaluatorValuePtr>>(type)
      .Case([&](circt::om::MapType type) {
        evaluator::EvaluatorValuePtr result =
            std::make_shared<evaluator::MapValue>(type);
        return success(result);
      })
      .Case([&](circt::om::ListType type) {
        evaluator::EvaluatorValuePtr result =
            std::make_shared<evaluator::ListValue>(type);
        return success(result);
      })
      .Case([&](mlir::TupleType type) {
        evaluator::EvaluatorValuePtr result =
            std::make_shared<evaluator::TupleValue>(type);
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
            std::make_shared<evaluator::ObjectValue>(cls);

        return success(result);
      })
      .Default([&](auto type) { return failure(); });
}

FailureOr<evaluator::EvaluatorValuePtr>
circt::om::Evaluator::getOrCreateValue(Value value,
                                       ActualParameters actualParams) {
  auto it = objects.find({value, actualParams});
  if (it != objects.end())
    return it->second;

  FailureOr<evaluator::EvaluatorValuePtr> result =
      TypeSwitch<Value, FailureOr<evaluator::EvaluatorValuePtr>>(value)
          .Case([&](BlockArgument arg) {
            return (*actualParams)[arg.getArgNumber()];
          })
          .Case([&](OpResult result) {
            return TypeSwitch<Operation *,
                              FailureOr<evaluator::EvaluatorValuePtr>>(
                       result.getDefiningOp())
                .Case([&](ConstantOp op) {
                  return evaluateConstant(op, actualParams);
                })
                .Case<ObjectFieldOp>([&](auto op) {
                  // Create a reference value since the value pointed by object
                  // field op is not created yet.
                  evaluator::EvaluatorValuePtr result =
                      std::make_shared<evaluator::ReferenceValue>(
                          value.getType());
                  return success(result);
                })
                .Case<AnyCastOp>([&](AnyCastOp op) {
                  return getOrCreateValue(op.getInput(), actualParams);
                })
                .Case<ListCreateOp, TupleCreateOp, MapCreateOp, ObjectFieldOp,
                      ObjectOp>([&](auto op) {
                  return getPartiallyEvaluatedValue(op.getType());
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

  for (auto &op : cls.getOps())
    for (auto result : op.getResults()) {
      // Allocate the value.
      if (failed(getOrCreateValue(result, actualParams)))
        return failure();
      // Add to the worklist.
      worklist.push({result, actualParams});
    }

  for (auto field : cls.getOps<ClassFieldOp>()) {
    StringAttr name = field.getSymNameAttr();
    Value value = field.getValue();
    FailureOr<evaluator::EvaluatorValuePtr> result =
        evaluateValue(value, actualParams);
    if (failed(result))
      return result;

    fields[name] = result.value();
  }

  // If the there is an instance, we must update the object value.
  if (instanceKey.first) {
    auto result =
        getOrCreateValue(instanceKey.first, instanceKey.second).value();
    auto *object = llvm::cast<evaluator::ObjectValue>(result.get());
    object->setFields(std::move(fields));
    return result;
  }

  // If it's external call, just allocate new ObjectValue.
  evaluator::EvaluatorValuePtr result =
      std::make_shared<evaluator::ObjectValue>(cls, fields);
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

  auto result =
      evaluateObjectInstance(className, actualParametersBuffers.back().get());

  if (failed(result))
    return failure();

  // `evaluateObjectInstance` has populated the worklist. Continue evaluations
  // unless there is a partially evaluated value.
  while (!worklist.empty()) {
    auto [value, args] = worklist.front();
    worklist.pop();

    auto result = evaluateValue(value, args);

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
circt::om::Evaluator::evaluateValue(Value value,
                                    ActualParameters actualParams) {
  auto evaluatorValue = getOrCreateValue(value, actualParams).value();

  // Return if the value is already evaluated.
  if (evaluatorValue->isFullyEvaluated())
    return evaluatorValue;

  return llvm::TypeSwitch<Value, FailureOr<evaluator::EvaluatorValuePtr>>(value)
      .Case([&](BlockArgument arg) {
        return evaluateParameter(arg, actualParams);
      })
      .Case([&](OpResult result) {
        return TypeSwitch<Operation *, FailureOr<evaluator::EvaluatorValuePtr>>(
                   result.getDefiningOp())
            .Case([&](ConstantOp op) {
              return evaluateConstant(op, actualParams);
            })
            .Case([&](ObjectOp op) {
              return evaluateObjectInstance(op, actualParams);
            })
            .Case([&](ObjectFieldOp op) {
              return evaluateObjectField(op, actualParams);
            })
            .Case([&](ListCreateOp op) {
              return evaluateListCreate(op, actualParams);
            })
            .Case([&](TupleCreateOp op) {
              return evaluateTupleCreate(op, actualParams);
            })
            .Case([&](TupleGetOp op) {
              return evaluateTupleGet(op, actualParams);
            })
            .Case([&](AnyCastOp op) {
              return evaluateValue(op.getInput(), actualParams);
            })
            .Case([&](MapCreateOp op) {
              return evaluateMapCreate(op, actualParams);
            })
            .Default([&](Operation *op) {
              auto error = op->emitError("unable to evaluate value");
              error.attachNote() << "value: " << value;
              return error;
            });
      });
}

/// Evaluator dispatch function for parameters.
FailureOr<evaluator::EvaluatorValuePtr>
circt::om::Evaluator::evaluateParameter(BlockArgument formalParam,
                                        ActualParameters actualParams) {
  return success((*actualParams)[formalParam.getArgNumber()]);
}

/// Evaluator dispatch function for constants.
FailureOr<circt::om::evaluator::EvaluatorValuePtr>
circt::om::Evaluator::evaluateConstant(ConstantOp op,
                                       ActualParameters actualParams) {
  return success(
      std::make_shared<circt::om::evaluator::AttributeValue>(op.getValue()));
}

/// Evaluator dispatch function for Object instances.
FailureOr<circt::om::Evaluator::ActualParameters>
circt::om::Evaluator::createParametersFromOperands(
    ValueRange range, ActualParameters actualParams) {
  // Create an unique storage to store parameters.
  auto parameters = std::make_unique<
      SmallVector<std::shared_ptr<evaluator::EvaluatorValue>>>();

  // Collect operands' evaluator values in the current instantiation context.
  for (auto input : range) {
    auto inputResult = getOrCreateValue(input, actualParams);
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
  if (isFullyEvaluated({op, actualParams}))
    return getOrCreateValue(op, actualParams);

  auto params = createParametersFromOperands(op.getOperands(), actualParams);
  if (failed(params))
    return failure();
  return evaluateObjectInstance(op.getClassNameAttr(), params.value(),
                                {op, actualParams});
}

/// Evaluator dispatch function for Object fields.
FailureOr<evaluator::EvaluatorValuePtr>
circt::om::Evaluator::evaluateObjectField(ObjectFieldOp op,
                                          ActualParameters actualParams) {
  // Evaluate the Object itself, in case it hasn't been evaluated yet.
  FailureOr<evaluator::EvaluatorValuePtr> currentObjectResult =
      evaluateValue(op.getObject(), actualParams);
  if (failed(currentObjectResult))
    return currentObjectResult;

  auto *currentObject =
      llvm::cast<evaluator::ObjectValue>(currentObjectResult.value().get());

  auto objectFieldValue = getOrCreateValue(op, actualParams).value();

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
                                         ActualParameters actualParams) {
  // Evaluate the Object itself, in case it hasn't been evaluated yet.
  SmallVector<evaluator::EvaluatorValuePtr> values;
  auto list = getOrCreateValue(op, actualParams);
  for (auto operand : op.getOperands()) {
    auto result = evaluateValue(operand, actualParams);
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
                                          ActualParameters actualParams) {
  SmallVector<evaluator::EvaluatorValuePtr> values;
  for (auto operand : op.getOperands()) {
    auto result = evaluateValue(operand, actualParams);
    if (failed(result))
      return result;
    values.push_back(result.value());
  }

  // Return the tuple.
  auto val = getOrCreateValue(op, actualParams);
  llvm::cast<evaluator::TupleValue>(val.value().get())
      ->setElements(std::move(values));
  return val;
}

/// Evaluator dispatch function for List creation.
FailureOr<evaluator::EvaluatorValuePtr>
circt::om::Evaluator::evaluateTupleGet(TupleGetOp op,
                                       ActualParameters actualParams) {
  auto tuple = evaluateValue(op.getInput(), actualParams);
  if (failed(tuple))
    return tuple;
  evaluator::EvaluatorValuePtr result =
      cast<evaluator::TupleValue>(tuple.value().get())
          ->getElements()[op.getIndex()];
  return result;
}

/// Evaluator dispatch function for Map creation.
FailureOr<evaluator::EvaluatorValuePtr>
circt::om::Evaluator::evaluateMapCreate(MapCreateOp op,
                                        ActualParameters actualParams) {
  // Evaluate the Object itself, in case it hasn't been evaluated yet.
  DenseMap<Attribute, evaluator::EvaluatorValuePtr> elements;
  auto valueResult = getOrCreateValue(op, actualParams).value();
  for (auto operand : op.getOperands()) {
    auto result = evaluateValue(operand, actualParams);
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
    if (auto lInt = dyn_cast<IntegerAttr>(l))
      if (auto rInt = dyn_cast<IntegerAttr>(r))
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
