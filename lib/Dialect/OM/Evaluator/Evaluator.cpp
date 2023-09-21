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

/// Instantiate an Object with its class name and actual parameters.
FailureOr<std::shared_ptr<evaluator::ObjectValue>>
circt::om::Evaluator::instantiate(
    StringAttr className, ArrayRef<evaluator::EvaluatorValuePtr> actualParams) {
  ClassOp cls = symbolTable.lookup<ClassOp>(className);
  if (!cls)
    return symbolTable.getOp()->emitError("unknown class name ") << className;

  auto formalParamNames = cls.getFormalParamNames().getAsRange<StringAttr>();
  auto formalParamTypes = cls.getBodyBlock()->getArgumentTypes();

  // Verify the actual parameters are the right size and types for this class.
  if (actualParams.size() != formalParamTypes.size()) {
    auto error = cls.emitError("actual parameter list length (")
                 << actualParams.size() << ") does not match formal "
                 << "parameter list length (" << formalParamTypes.size() << ")";
    auto &diag = error.attachNote() << "actual parameters: ";
    // FIXME: `diag << actualParams` doesn't work for some reason.
    bool isFirst = true;
    for (const auto &param : actualParams) {
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
       llvm::zip(actualParams, formalParamNames, formalParamTypes)) {
    if (!actualParam || !actualParam.get())
      return cls.emitError("actual parameter for ")
             << formalParamName << " is null";

    // Subtyping: if formal param is any type, any actual param may be passed.
    if (isa<AnyType>(formalParamType))
      continue;

    Type actualParamType;
    if (auto *attr = dyn_cast<evaluator::AttributeValue>(actualParam.get())) {
      if (auto typedActualParam = attr->getAttr().dyn_cast_or_null<TypedAttr>())
        actualParamType = typedActualParam.getType();
    } else if (auto *object =
                   dyn_cast<evaluator::ObjectValue>(actualParam.get()))
      actualParamType = object->getType();
    else if (auto *list = dyn_cast<evaluator::ListValue>(actualParam.get()))
      actualParamType = list->getType();
    else if (auto *tuple = dyn_cast<evaluator::TupleValue>(actualParam.get()))
      actualParamType = tuple->getType();

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
  for (auto field : cls.getOps<ClassFieldOp>()) {
    StringAttr name = field.getSymNameAttr();
    Value value = field.getValue();

    FailureOr<evaluator::EvaluatorValuePtr> result =
        evaluateValue(value, actualParams);
    if (failed(result))
      return failure();

    fields[name] = result.value();
  }

  // Allocate the Object. Further refinement is expected.
  auto *object = new evaluator::ObjectValue(cls, fields);

  return success(std::shared_ptr<evaluator::ObjectValue>(object));
}

/// Evaluate a Value in a Class body according to the semantics of the IR. The
/// actual parameters are the values supplied at the current instantiation of
/// the Class being evaluated.
FailureOr<evaluator::EvaluatorValuePtr> circt::om::Evaluator::evaluateValue(
    Value value, ArrayRef<evaluator::EvaluatorValuePtr> actualParams) {
  return TypeSwitch<Value, FailureOr<evaluator::EvaluatorValuePtr>>(value)
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
            .Case([&](MapCreateOp op) {
              return evaluateMapCreate(op, actualParams);
            })
            .Case([&](AnyCastOp op) {
              return evaluateValue(op.getInput(), actualParams);
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
    BlockArgument formalParam,
    ArrayRef<evaluator::EvaluatorValuePtr> actualParams) {
  return success(actualParams[formalParam.getArgNumber()]);
}

/// Evaluator dispatch function for constants.
FailureOr<circt::om::evaluator::EvaluatorValuePtr>
circt::om::Evaluator::evaluateConstant(
    ConstantOp op,
    ArrayRef<circt::om::evaluator::EvaluatorValuePtr> actualParams) {
  return success(
      std::make_shared<circt::om::evaluator::AttributeValue>(op.getValue()));
}

/// Evaluator dispatch function for Object instances.
FailureOr<evaluator::EvaluatorValuePtr>
circt::om::Evaluator::evaluateObjectInstance(
    ObjectOp op, ArrayRef<evaluator::EvaluatorValuePtr> actualParams) {
  // First, check if we have already evaluated this object, and return it if so.
  auto existingInstance = objects.find(op);
  if (existingInstance != objects.end())
    return success(existingInstance->second);

  // If we need to instantiate a new object, evaluate values for all of its
  // actual parameters. Note that this is eager evaluation, which precludes
  // creating cycles in the object model. Further refinement is expected.
  SmallVector<evaluator::EvaluatorValuePtr> objectParams;
  for (auto param : op.getActualParams()) {
    FailureOr<evaluator::EvaluatorValuePtr> result =
        evaluateValue(param, actualParams);
    if (failed(result))
      return result;
    objectParams.push_back(result.value());
  }

  // Instantiate and return the new Object, saving the instance for later.
  auto newInstance = instantiate(op.getClassNameAttr(), objectParams);
  if (succeeded(newInstance))
    objects[op.getResult()] = newInstance.value();
  return newInstance;
}

/// Evaluator dispatch function for Object fields.
FailureOr<evaluator::EvaluatorValuePtr>
circt::om::Evaluator::evaluateObjectField(
    ObjectFieldOp op, ArrayRef<evaluator::EvaluatorValuePtr> actualParams) {
  // Evaluate the Object itself, in case it hasn't been evaluated yet.
  FailureOr<evaluator::EvaluatorValuePtr> currentObjectResult =
      evaluateValue(op.getObject(), actualParams);
  if (failed(currentObjectResult))
    return currentObjectResult;

  auto *currentObject =
      llvm::cast<evaluator::ObjectValue>(currentObjectResult.value().get());

  // Iteratively access nested fields through the path until we reach the final
  // field in the path.
  evaluator::EvaluatorValuePtr finalField;
  for (auto field : op.getFieldPath().getAsRange<FlatSymbolRefAttr>()) {
    auto currentField = currentObject->getField(field.getAttr());
    finalField = currentField.value();
    if (auto *nextObject =
            llvm::dyn_cast<evaluator::ObjectValue>(finalField.get()))
      currentObject = nextObject;
  }

  // Return the field being accessed.
  return finalField;
}

/// Evaluator dispatch function for List creation.
FailureOr<evaluator::EvaluatorValuePtr>
circt::om::Evaluator::evaluateListCreate(
    ListCreateOp op, ArrayRef<evaluator::EvaluatorValuePtr> actualParams) {
  // Evaluate the Object itself, in case it hasn't been evaluated yet.
  SmallVector<evaluator::EvaluatorValuePtr> values;
  for (auto operand : op.getOperands()) {
    auto result = evaluateValue(operand, actualParams);
    if (failed(result))
      return result;
    values.push_back(result.value());
  }

  // Return the list.
  evaluator::EvaluatorValuePtr result =
      std::make_shared<evaluator::ListValue>(op.getType(), std::move(values));
  return result;
}

/// Evaluator dispatch function for Tuple creation.
FailureOr<evaluator::EvaluatorValuePtr>
circt::om::Evaluator::evaluateTupleCreate(
    TupleCreateOp op, ArrayRef<evaluator::EvaluatorValuePtr> actualParams) {
  SmallVector<evaluator::EvaluatorValuePtr> values;
  for (auto operand : op.getOperands()) {
    auto result = evaluateValue(operand, actualParams);
    if (failed(result))
      return result;
    values.push_back(result.value());
  }

  // Return the tuple.
  evaluator::EvaluatorValuePtr result = std::make_shared<evaluator::TupleValue>(
      op.getType().cast<mlir::TupleType>(), std::move(values));

  return result;
}

/// Evaluator dispatch function for List creation.
FailureOr<evaluator::EvaluatorValuePtr> circt::om::Evaluator::evaluateTupleGet(
    TupleGetOp op, ArrayRef<evaluator::EvaluatorValuePtr> actualParams) {
  auto tuple = evaluateValue(op.getInput(), actualParams);
  if (failed(tuple))
    return tuple;
  evaluator::EvaluatorValuePtr result =
      cast<evaluator::TupleValue>(tuple.value().get())
          ->getElements()[op.getIndex()];
  return result;
}

/// Evaluator dispatch function for Map creation.
FailureOr<evaluator::EvaluatorValuePtr> circt::om::Evaluator::evaluateMapCreate(
    MapCreateOp op, ArrayRef<evaluator::EvaluatorValuePtr> actualParams) {
  // Evaluate the Object itself, in case it hasn't been evaluated yet.
  DenseMap<Attribute, evaluator::EvaluatorValuePtr> elements;
  for (auto operand : op.getOperands()) {
    auto result = evaluateValue(operand, actualParams);
    if (failed(result))
      return result;
    // The result is a tuple.
    auto &value = result.value();
    const auto &element =
        llvm::cast<evaluator::TupleValue>(value.get())->getElements();
    assert(element.size() == 2);
    auto attr =
        llvm::cast<evaluator::AttributeValue>(element[0].get())->getAttr();
    if (!elements.insert({attr, element[1]}).second)
      return op.emitError() << "map contains duplicated keys";
  }

  // Return the Map.
  evaluator::EvaluatorValuePtr result =
      std::make_shared<evaluator::MapValue>(op.getType(), std::move(elements));
  return result;
}

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
