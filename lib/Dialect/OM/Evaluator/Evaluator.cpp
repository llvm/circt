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
#include "circt/Dialect/OM/OMPasses.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributeInterfaces.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Pass/PassManager.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/ScopeExit.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/ADT/iterator_range.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "om-evaluator"

using namespace mlir;
using namespace circt::om;

namespace {

LogicalResult verifyActualParameters(ClassLike classLike,
                                     ArrayRef<EvaluatorValuePtr> actualParams) {
  auto formalParamNames =
      classLike.getFormalParamNames().getAsRange<StringAttr>();
  auto formalParamTypes = classLike.getBodyBlock()->getArgumentTypes();

  if (actualParams.size() != formalParamTypes.size()) {
    auto error = classLike.emitError("actual parameter list length (")
                 << actualParams.size() << ") does not match formal "
                 << "parameter list length (" << formalParamTypes.size() << ")";
    auto &diag = error.attachNote() << "actual parameters: ";
    bool isFirst = true;
    for (const auto &param : actualParams) {
      if (isFirst)
        isFirst = false;
      else
        diag << ", ";
      diag << param;
    }
    error.attachNote(classLike.getLoc())
        << "formal parameters: " << formalParamTypes;
    return failure();
  }

  for (auto [actualParam, formalParamName, formalParamType] :
       llvm::zip(actualParams, formalParamNames, formalParamTypes)) {
    if (!actualParam || !actualParam.get())
      return classLike.emitError("actual parameter for ")
             << formalParamName << " is null";

    // Subtyping: if formal param is any type, any actual param may be passed.
    if (isa<AnyType>(formalParamType))
      continue;

    Type actualParamType = actualParam->getType();
    assert(actualParamType && "actualParamType must be non-null!");

    if (actualParamType != formalParamType) {
      auto error = classLike.emitError("actual parameter for ")
                   << formalParamName << " has invalid type";
      error.attachNote() << "actual parameter: " << *actualParam;
      error.attachNote() << "format parameter type: " << formalParamType;
      return failure();
    }
  }
  return success();
}

bool requiresCompleteEvaluation(const evaluator::EvaluatorValuePtr &value) {
  return !value->isFullyEvaluated() &&
         !isa<evaluator::ObjectValue>(value.get());
}

/// A helper class that builds the scratch IR for evaluating an object. This is
/// used to convert from the evaluator's API (which uses opaque pointers to
/// evaluator values) into actual MLIR IR.
class ScratchIRBuilder {
public:
  struct InstantiationInfo {
    StringAttr className;
    SmallVector<EvaluatorValuePtr> actualParams;
  };

  ScratchIRBuilder(ModuleOp module, SymbolTable &symbolTable,
                   ClassLike rootClass)
      : module(module), symbolTable(symbolTable), rootClass(rootClass),
        wrapperClass(createWrapperClass(rootClass)) {}

  FailureOr<InstantiationInfo> run(ArrayRef<EvaluatorValuePtr> actualParams);

private:
  /// Create the temporary class that owns all scratch IR.
  ClassOp createWrapperClass(ClassLike rootClass);

  /// Convert an API input value into scratch IR, preserving opaque any-typed
  /// inputs and rejecting object cycles.
  FailureOr<Value> materializeInput(const EvaluatorValuePtr &value,
                                    Location loc, Type expectedType);
  /// Convert a fully evaluated list value into scratch IR.
  FailureOr<Value> materializeListInput(evaluator::ListValue *listValue,
                                        Location loc);
  /// Convert a fully evaluated object value into scratch IR.
  FailureOr<Value> materializeObjectInput(evaluator::ObjectValue *objectValue,
                                          Location loc);
  /// Add a wrapper class parameter for an input that must stay opaque.
  FailureOr<Value> createWrapperArgument(EvaluatorValuePtr value, Location loc,
                                         Type argType);

  ModuleOp module;
  SymbolTable &symbolTable;
  ClassLike rootClass;
  ClassOp wrapperClass;
  // A mapping from evaluator input values to their corresponding imported IR
  // values.
  DenseMap<evaluator::EvaluatorValue *, Value> importedValues;

  // A set of object values that have been imported into the scratch IR, used to
  // detect mutual references in the inputs.
  SmallPtrSet<evaluator::ObjectValue *, 8> activeObjectImports;

  SmallVector<Attribute> wrapperArgNames;
  SmallVector<EvaluatorValuePtr> wrapperActualParams;
};

FailureOr<ScratchIRBuilder::InstantiationInfo>
ScratchIRBuilder::run(ArrayRef<EvaluatorValuePtr> actualParams) {
  auto *ctx = module.getContext();
  assert(rootClass && "root class must be resolved before building scratch IR");
  auto rootLoc = rootClass.getLoc();
  auto rootClassName = rootClass.getSymNameAttr();

  OpBuilder builder(wrapperClass.getFieldsOp());
  builder.setInsertionPoint(wrapperClass.getFieldsOp());
  SmallVector<Value> importedActualValues;
  importedActualValues.reserve(actualParams.size());
  auto formalTypes = rootClass.getBodyBlock()->getArgumentTypes();
  for (auto [actual, expectedType] : llvm::zip(actualParams, formalTypes)) {
    auto imported = materializeInput(actual, rootLoc, expectedType);
    if (failed(imported))
      return failure();
    importedActualValues.push_back(*imported);
  }

  // Update wrapper class after materializing actual parameters.
  wrapperClass->setAttr(wrapperClass.getFormalParamNamesAttrName(),
                        builder.getArrayAttr(wrapperArgNames));

  wrapperClass.updateFields(
      {rootLoc},
      {ObjectOp::create(
           builder, rootLoc,
           ClassType::get(ctx, FlatSymbolRefAttr::get(rootClassName)),
           rootClassName, importedActualValues)
           .getResult()},
      {builder.getStringAttr("root")});

  if (failed(verify(module)))
    return failure();

  PassManager pm(ctx);
  ElaborateObjectOptions options;
  auto wrapperName = wrapperClass.getSymNameAttr();
  options.targetClass = wrapperName.getValue().str();
  pm.addPass(createElaborateObject(std::move(options)));
  if (failed(pm.run(module)))
    return failure();

  return InstantiationInfo{wrapperName, std::move(wrapperActualParams)};
}

ClassOp ScratchIRBuilder::createWrapperClass(ClassLike rootClass) {
  OpBuilder builder(module.getBody(), module.getBody()->end());
  builder.setInsertionPointToEnd(module.getBody());

  auto wrapper = ClassOp::create(builder, rootClass.getLoc(),
                                 Twine("__om_evaluator_wrapper_") +
                                     rootClass.getSymName());
  (void)symbolTable.insert(wrapper);
  Block *body = &wrapper.getBody().emplaceBlock();
  builder.setInsertionPointToEnd(body);
  ClassFieldsOp::create(builder, rootClass.getLoc(), ValueRange(), ArrayAttr{});
  return wrapper;
}

FailureOr<Value>
ScratchIRBuilder::materializeInput(const EvaluatorValuePtr &value, Location loc,
                                   Type expectedType) {
  if (!value)
    return emitError(loc, "cannot materialize null OM evaluator value");

  loc = value->getLoc();
  if (!expectedType)
    return emitError(loc, "cannot import OM evaluator value without an "
                          "expected type");

  // Keep any-typed values opaque at the wrapper boundary.
  if (isa<AnyType>(expectedType))
    return createWrapperArgument(value, loc, expectedType);

  if (auto it = importedValues.find(value.get()); it != importedValues.end())
    return it->second;

  if (value->isUnknown()) {
    OpBuilder builder(wrapperClass.getFieldsOp());
    auto result = UnknownValueOp::create(builder, loc, expectedType);
    importedValues[value.get()] = result.getResult();
    return result.getResult();
  }

  return llvm::TypeSwitch<evaluator::EvaluatorValue *, FailureOr<Value>>(
             value.get())
      .Case([&](evaluator::AttributeValue *attrValue) -> FailureOr<Value> {
        auto attr = attrValue->getAttr();
        if (!attr)
          return emitError(loc, "cannot import OM attribute value without an "
                                "attribute");

        OpBuilder builder(wrapperClass.getFieldsOp());
        auto result = ConstantOp::create(builder, loc, cast<TypedAttr>(attr));
        importedValues[value.get()] = result.getResult();
        return result.getResult();
      })
      .Case([&](evaluator::ListValue *listValue) {
        return materializeListInput(listValue, loc);
      })
      .Case([&](evaluator::ObjectValue *objectValue) {
        return materializeObjectInput(objectValue, loc);
      })
      .Default([&](evaluator::EvaluatorValue *) -> FailureOr<Value> {
        auto result = createWrapperArgument(value, loc, expectedType);
        if (succeeded(result))
          importedValues[value.get()] = *result;
        return result;
      });
}

FailureOr<Value>
ScratchIRBuilder::materializeListInput(evaluator::ListValue *listValue,
                                       Location loc) {
  if (!listValue->isFullyEvaluated())
    return emitError(loc, "cannot import partially evaluated OM list value");

  auto listType = listValue->getListType();
  SmallVector<Value> elementValues;
  elementValues.reserve(listValue->getElements().size());
  for (const auto &elementValue : listValue->getElements()) {
    auto materializedElement =
        materializeInput(elementValue, loc, listType.getElementType());
    if (failed(materializedElement))
      return failure();
    elementValues.push_back(*materializedElement);
  }

  OpBuilder builder(wrapperClass.getFieldsOp());
  auto result = ListCreateOp::create(builder, loc, listType, elementValues);
  importedValues[listValue] = result.getResult();
  return result.getResult();
}

FailureOr<Value>
ScratchIRBuilder::materializeObjectInput(evaluator::ObjectValue *objectValue,
                                         Location loc) {
  // TODO: Currently we only support importing object values that don't have
  // mutual references with other object values in the inputs for the
  // simplicity. We could construct mutually referencing object values with a
  // backedge builder but currently we don't have a use case for that.
  if (!activeObjectImports.insert(objectValue).second)
    return emitError(loc, "cannot import mutually referential OM objects");

  llvm::scope_exit popActiveObjectImport(
      [&] { activeObjectImports.erase(objectValue); });

  auto classLike = objectValue->getClassOp();
  SmallVector<Value> fieldValues;
  auto fieldNames = classLike.getFieldNames();
  fieldValues.reserve(fieldNames.size());
  for (auto fieldName : fieldNames) {
    auto fieldNameAttr = cast<StringAttr>(fieldName);
    auto field = objectValue->getField(fieldNameAttr);
    if (failed(field))
      return failure();
    auto materializedField = materializeInput(
        field.value(), loc, classLike.getFieldType(fieldNameAttr).value());
    if (failed(materializedField))
      return failure();
    fieldValues.push_back(*materializedField);
  }

  OpBuilder builder(wrapperClass.getFieldsOp());
  auto result =
      ElaboratedObjectOp::create(builder, loc, classLike, fieldValues);
  importedValues[objectValue] = result.getResult();
  return result.getResult();
}

FailureOr<Value>
ScratchIRBuilder::createWrapperArgument(EvaluatorValuePtr value, Location loc,
                                        Type argType) {
  Builder builder(module.getContext());
  wrapperArgNames.push_back(
      builder.getStringAttr(Twine("arg") + Twine(wrapperArgNames.size())));
  wrapperActualParams.push_back(value);
  return wrapperClass.getBodyBlock()->addArgument(argType, loc);
}

} // namespace

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
    values.push_back(evaluator::AttributeValue::get(cast<TypedAttr>(attr)));
  return values;
}

Type circt::om::evaluator::EvaluatorValue::getType() const {
  return llvm::TypeSwitch<const EvaluatorValue *, Type>(this)
      .Case<AttributeValue>([](auto *attr) -> Type { return attr->getType(); })
      .Case<ObjectValue>([](auto *object) { return object->getObjectType(); })
      .Case<ListValue>([](auto *list) { return list->getListType(); })
      .Case<BasePathValue>(
          [this](auto *tuple) { return FrozenBasePathType::get(ctx); })
      .Case<PathValue>(
          [this](auto *tuple) { return FrozenPathType::get(ctx); });
}

FailureOr<evaluator::EvaluatorValuePtr>
circt::om::Evaluator::getPartiallyEvaluatedValue(Type type, Location loc) {
  using namespace circt::om::evaluator;

  auto result =
      TypeSwitch<mlir::Type, FailureOr<evaluator::EvaluatorValuePtr>>(type)
          .Case([&](circt::om::ListType type) {
            evaluator::EvaluatorValuePtr result =
                std::make_shared<evaluator::ListValue>(type, loc);
            return success(result);
          })
          .Case([&](circt::om::ClassType type)
                    -> FailureOr<evaluator::EvaluatorValuePtr> {
            auto classDef =
                symbolTable.lookup<ClassLike>(type.getClassName().getValue());
            if (!classDef)
              return symbolTable.getOp()->emitError("unknown class name ")
                     << type.getClassName();

            // Create an ObjectValue for both ClassOp and ClassExternOp
            evaluator::EvaluatorValuePtr result =
                std::make_shared<evaluator::ObjectValue>(classDef, loc);

            return success(result);
          })
          .Case([&](circt::om::StringType type) {
            evaluator::EvaluatorValuePtr result =
                evaluator::AttributeValue::get(type, loc);
            return success(result);
          })
          .Default([&](auto type) { return failure(); });

  return result;
}

FailureOr<evaluator::EvaluatorValuePtr>
circt::om::Evaluator::getOrCreateValue(Value value, Location loc) {
  LLVM_DEBUG(dbgs() << "- get: " << value << "\n");

  auto it = objects.find(value);
  if (it != objects.end()) {
    auto evalVal = it->second;
    evalVal->setLocIfUnknown(loc);
    return evalVal;
  }

  FailureOr<evaluator::EvaluatorValuePtr> result =
      TypeSwitch<Value, FailureOr<evaluator::EvaluatorValuePtr>>(value)
          .Case([&](BlockArgument arg) {
            auto error = arg.getOwner()->getParentOp()->emitError(
                "unable to evaluate unbound parameter");
            error.attachNote() << "value: " << value;
            return error;
          })
          .Case([&](OpResult result) {
            return TypeSwitch<Operation *,
                              FailureOr<evaluator::EvaluatorValuePtr>>(
                       result.getDefiningOp())
                .Case<ConstantOp, UnknownValueOp>(
                    [&](auto op) { return evaluateOp(op, loc); })
                .Case<AnyCastOp>([&](AnyCastOp op) {
                  return getOrCreateValue(op.getInput(), loc);
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
                .Case<ListCreateOp, ListConcatOp>([&](auto op) {
                  return getPartiallyEvaluatedValue(op.getType(), loc);
                })
                .Case<ElaboratedObjectOp>([&](auto op) {
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

  objects[value] = result.value();
  return result;
}

FailureOr<evaluator::EvaluatorValuePtr>
circt::om::Evaluator::evaluateObjectInstance(
    StringAttr className, ArrayRef<EvaluatorValuePtr> actualParams,
    Location loc) {
#ifndef NDEBUG
  DebugNesting nestOne(debugNesting);
#endif
  LLVM_DEBUG(dbgs() << "object:\n");
#ifndef NDEBUG
  DebugNesting nestTwo(debugNesting);
#endif
  LLVM_DEBUG(dbgs() << "name: " << className << "\n");

  auto classDef = symbolTable.lookup<ClassLike>(className);
  if (!classDef)
    return symbolTable.getOp()->emitError("unknown class name ") << className;

  // If this is an external class, create an ObjectValue and mark it unknown
  if (isa<ClassExternOp>(classDef)) {
    evaluator::EvaluatorValuePtr result =
        std::make_shared<evaluator::ObjectValue>(classDef, loc);
    result->markUnknown();
    LLVM_DEBUG(dbgs(1) << "extern: <unknown-value>\n");
    return result;
  }

  // Otherwise, it's a regular class, proceed normally
  ClassOp cls = cast<ClassOp>(classDef);

  if (failed(verifyActualParameters(cls, actualParams)))
    return failure();

  for (auto [arg, actual] :
       llvm::zip(cls.getBodyBlock()->getArguments(), actualParams)) {
    actual->setLoc(arg.getLoc());
    objects.try_emplace(arg, actual);
  }

  evaluator::ObjectFields fields;
  auto *context = cls.getContext();
  {
    LLVM_DEBUG(dbgs() << "ops:\n");
#ifndef NDEBUG
    DebugNesting nestOne(debugNesting);
#endif
    // Allocate placeholders for all class-body results before evaluating any
    // fields.
    for (auto &op : cls.getOps())
      for (auto result : op.getResults())
        if (failed(getOrCreateValue(result, UnknownLoc::get(context))))
          return failure();

    // A later field evaluation can then connect object valued cycles by
    // pointing at these placeholders, and the placeholders are filled
    // below.
    for (auto &op : cls.getOps()) {
      for (auto result : op.getResults()) {
        auto evaluated = evaluateValue(result, op.getLoc());
        if (failed(evaluated))
          return failure();
        if (requiresCompleteEvaluation(evaluated.value()))
          return op.emitError("failed to evaluate value");
      }
    }
  }

  LLVM_DEBUG(dbgs() << "fields:\n");
  auto fieldNames = cls.getFieldNames();
  auto operands = cls.getFieldsOp()->getOperands();
  for (size_t i = 0; i < fieldNames.size(); ++i) {
    auto name = fieldNames[i];
    auto value = operands[i];
    auto fieldLoc = cls.getFieldLocByIndex(i);
    LLVM_DEBUG(dbgs() << "- name: " << name << "\n"
                      << indent(1) << "evaluate:\n");
#ifndef NDEBUG
    DebugNesting nestOne(debugNesting);
#endif
    FailureOr<evaluator::EvaluatorValuePtr> result =
        evaluateValue(value, fieldLoc);
    if (failed(result))
      return result;
    if (requiresCompleteEvaluation(result.value()))
      return emitError(fieldLoc, "failed to evaluate field ") << name;

    LLVM_DEBUG(dbgs() << "value: " << result.value() << "\n");
    fields[cast<StringAttr>(name)] = result.value();
  }

  evaluator::EvaluatorValuePtr result =
      std::make_shared<evaluator::ObjectValue>(cls, fields, loc);
  assert(result->isFullyEvaluated() &&
         "object with fields should be fully evaluated");
  return result;
}

/// Instantiate an Object with its class name and actual parameters.
FailureOr<std::shared_ptr<evaluator::EvaluatorValue>>
circt::om::Evaluator::instantiate(
    StringAttr className, ArrayRef<evaluator::EvaluatorValuePtr> actualParams) {
  LLVM_DEBUG(dbgs() << "instantiate:\n");
#ifndef NDEBUG
  DebugNesting nest(debugNesting);
#endif
  LLVM_DEBUG({
    dbgs() << "class: " << className << "\n" << indent() << "params:\n";
    for (auto &param : actualParams)
      dbgs() << "- " << param << "\n";
  });

  auto rootClass = symbolTable.lookup<ClassLike>(className);
  if (!rootClass)
    return symbolTable.getOp()->emitError("unknown class name ") << className;
  if (failed(verifyActualParameters(rootClass, actualParams)))
    return failure();

  ScratchIRBuilder scratchBuilder(getModule(), symbolTable, rootClass);
  auto transformedInstantiation = scratchBuilder.run(actualParams);
  if (failed(transformedInstantiation))
    return failure();

  auto wrapper = instantiateImpl(transformedInstantiation->className,
                                 transformedInstantiation->actualParams);
  if (failed(wrapper))
    return failure();

  auto root =
      cast<evaluator::ObjectValue>(wrapper.value().get())->getField("root");
  if (failed(root))
    return failure();
  return root.value();
}

FailureOr<std::shared_ptr<evaluator::EvaluatorValue>>
circt::om::Evaluator::instantiateImpl(
    StringAttr className, ArrayRef<evaluator::EvaluatorValuePtr> actualParams) {
  auto classDef = symbolTable.lookup<ClassLike>(className);
  if (!classDef)
    return symbolTable.getOp()->emitError("unknown class name ") << className;

  // If this is an external class, create an ObjectValue and mark it unknown
  if (isa<ClassExternOp>(classDef)) {
    evaluator::EvaluatorValuePtr result =
        std::make_shared<evaluator::ObjectValue>(
            classDef, UnknownLoc::get(classDef.getContext()));
    result->markUnknown();
    LLVM_DEBUG(dbgs(1) << "result: <unknown extern>\n");
    return result;
  }

  // Otherwise, it's a regular class, proceed normally
  ClassOp cls = cast<ClassOp>(classDef);

  objects.clear();

  auto loc = cls.getLoc();
  LLVM_DEBUG(dbgs() << "evaluate object:\n");
  auto result = evaluateObjectInstance(className, actualParams, loc);

  if (failed(result))
    return failure();

  LLVM_DEBUG(dbgs() << "result: " << result.value() << "\n");
  return result;
}

FailureOr<evaluator::EvaluatorValuePtr>
circt::om::Evaluator::evaluateValue(Value value, Location loc) {
  auto evaluatorValue = getOrCreateValue(value, loc);
  if (failed(evaluatorValue))
    return failure();

  LLVM_DEBUG(dbgs() << "- eval: " << value << "\n");

  // Return if the value is already evaluated.
  if (evaluatorValue.value()->isFullyEvaluated()) {
    LLVM_DEBUG(dbgs(1) << "fully evaluated: " << evaluatorValue.value()
                       << "\n");
    return evaluatorValue;
  }

  return llvm::TypeSwitch<Value, FailureOr<evaluator::EvaluatorValuePtr>>(value)
      .Case([&](BlockArgument arg) { return evaluatorValue; })
      .Case([&](OpResult result) {
        return TypeSwitch<Operation *, FailureOr<evaluator::EvaluatorValuePtr>>(
                   result.getDefiningOp())
            .Case<ConstantOp, ElaboratedObjectOp, ListCreateOp, ListConcatOp,
                  FrozenBasePathCreateOp, FrozenPathCreateOp, FrozenEmptyPathOp,
                  UnknownValueOp>([&](auto op) { return evaluateOp(op, loc); })
            .Case(
                [&](AnyCastOp op) { return evaluateValue(op.getInput(), loc); })
            .Default([&](Operation *op) {
              auto error = op->emitError("unable to evaluate value");
              error.attachNote() << "value: " << value;
              return error;
            });
      });
}

/// Evaluator dispatch function for constants.
FailureOr<circt::om::evaluator::EvaluatorValuePtr>
circt::om::Evaluator::evaluateOp(ConstantOp op, Location loc) {
  // For list constants, create ListValue.
  return success(om::evaluator::AttributeValue::get(op.getValue(), loc));
}

FailureOr<evaluator::EvaluatorValuePtr>
circt::om::Evaluator::evaluateOp(ElaboratedObjectOp op, Location loc) {
  auto objectValue = getOrCreateValue(op, loc);
  if (failed(objectValue))
    return failure();
  auto object = cast<evaluator::ObjectValue>(objectValue.value().get());
  if (object->isFullyEvaluated())
    return objectValue;

  auto classLike =
      symbolTable.lookup<ClassLike>(op.getClassNameAttr().getAttr());
  if (!classLike)
    return symbolTable.getOp()->emitError("unknown class name ")
           << op.getClassNameAttr();

  auto fieldNames = classLike.getFieldNames();
  auto fieldValues = op.getFieldValues();
  if (fieldNames.size() != fieldValues.size())
    return op.emitError("field value list doesn't match class field list, "
                        "expected ")
           << fieldNames.size() << " values but got " << fieldValues.size();

  evaluator::ObjectFields fields;
  auto classOp = dyn_cast<ClassOp>(classLike.getOperation());
  for (auto [index, fieldNameAndValue] :
       llvm::enumerate(llvm::zip(fieldNames, fieldValues))) {
    auto [fieldName, fieldValue] = fieldNameAndValue;
    auto fieldLoc = classOp ? classOp.getFieldLocByIndex(index) : loc;
    auto fieldResult = getOrCreateValue(fieldValue, fieldLoc);
    if (failed(fieldResult))
      return failure();
    if (!isa<evaluator::ObjectValue>(fieldResult.value().get()))
      fieldResult = evaluateValue(fieldValue, fieldLoc);
    if (failed(fieldResult))
      return failure();

    if (requiresCompleteEvaluation(fieldResult.value()))
      return emitError(fieldLoc, "failed to evaluate field ") << fieldName;

    fields[cast<StringAttr>(fieldName)] = fieldResult.value();
  }

  object->setFields(std::move(fields));
  return objectValue;
}

/// Evaluator dispatch function for List creation.
FailureOr<evaluator::EvaluatorValuePtr>
circt::om::Evaluator::evaluateOp(ListCreateOp op, Location loc) {
  // Evaluate the Object itself, in case it hasn't been evaluated yet.
  SmallVector<evaluator::EvaluatorValuePtr> values;
  auto list = getOrCreateValue(op, loc);
  bool hasUnknown = false;
  for (auto operand : op.getOperands()) {
    auto result = evaluateValue(operand, loc);
    if (failed(result))
      return result;
    if (requiresCompleteEvaluation(result.value()))
      return op.emitError()
             << "failed to evaluate list element operand: " << operand;
    // Check if any operand is unknown.
    if (result.value()->isUnknown())
      hasUnknown = true;
    values.push_back(result.value());
  }

  // Set the list elements (this also marks the list as fully evaluated).
  llvm::cast<evaluator::ListValue>(list.value().get())
      ->setElements(std::move(values));

  // If any operand is unknown, mark the list as unknown.
  // markUnknown() checks if already fully evaluated before calling
  // markFullyEvaluated().
  if (hasUnknown)
    list.value()->markUnknown();

  return list;
}

/// Evaluator dispatch function for List concatenation.
FailureOr<evaluator::EvaluatorValuePtr>
circt::om::Evaluator::evaluateOp(ListConcatOp op, Location loc) {
  // Evaluate the List concat op itself, in case it hasn't been evaluated yet.
  SmallVector<evaluator::EvaluatorValuePtr> values;
  auto list = getOrCreateValue(op, loc);

  // Extract the ListValue.
  auto extractList = [](evaluator::EvaluatorValue *value) {
    return std::move(
        llvm::TypeSwitch<evaluator::EvaluatorValue *, evaluator::ListValue *>(
            value)
            .Case([](evaluator::ListValue *val) { return val; }));
  };

  bool hasUnknown = false;
  for (auto operand : op.getOperands()) {
    auto result = evaluateValue(operand, loc);
    if (failed(result))
      return result;
    if (!result.value()->isFullyEvaluated())
      return list;
    // Check if any operand is unknown.
    if (result.value()->isUnknown())
      hasUnknown = true;

    // Extract this sublist and ensure it's done evaluating.
    evaluator::ListValue *subList = extractList(result.value().get());
    if (!subList->isFullyEvaluated())
      return list;

    // Append each EvaluatorValue from the sublist.
    for (const auto &subValue : subList->getElements())
      values.push_back(subValue);
  }

  // Return the concatenated list.
  llvm::cast<evaluator::ListValue>(list.value().get())
      ->setElements(std::move(values));

  // If any operand is unknown, mark the result as unknown.
  // markUnknown() checks if already fully evaluated before calling
  // markFullyEvaluated().
  if (hasUnknown)
    list.value()->markUnknown();

  return list;
}

FailureOr<evaluator::EvaluatorValuePtr>
circt::om::Evaluator::evaluateOp(FrozenBasePathCreateOp op, Location loc) {
  // Evaluate the Object itself, in case it hasn't been evaluated yet.
  auto valueResult = getOrCreateValue(op, loc).value();
  auto *path = llvm::cast<evaluator::BasePathValue>(valueResult.get());
  auto result = evaluateValue(op.getBasePath(), loc);
  if (failed(result))
    return result;
  auto &value = result.value();
  if (!value->isFullyEvaluated())
    return valueResult;

  // If the base path is unknown, mark the result as unknown.
  if (result.value()->isUnknown()) {
    valueResult->markUnknown();
    return valueResult;
  }

  path->setBasepath(*llvm::cast<evaluator::BasePathValue>(value.get()));
  return valueResult;
}

FailureOr<evaluator::EvaluatorValuePtr>
circt::om::Evaluator::evaluateOp(FrozenPathCreateOp op, Location loc) {
  // Evaluate the Object itself, in case it hasn't been evaluated yet.
  auto valueResult = getOrCreateValue(op, loc).value();
  auto *path = llvm::cast<evaluator::PathValue>(valueResult.get());
  auto result = evaluateValue(op.getBasePath(), loc);
  if (failed(result))
    return result;
  auto &value = result.value();
  if (!value->isFullyEvaluated())
    return valueResult;

  // If the base path is unknown, mark the result as unknown.
  if (result.value()->isUnknown()) {
    valueResult->markUnknown();
    return valueResult;
  }

  path->setBasepath(*llvm::cast<evaluator::BasePathValue>(value.get()));
  return valueResult;
}

FailureOr<evaluator::EvaluatorValuePtr>
circt::om::Evaluator::evaluateOp(FrozenEmptyPathOp op, Location loc) {
  auto valueResult = getOrCreateValue(op, loc).value();
  return valueResult;
}

/// Create an unknown value of the specified type
FailureOr<evaluator::EvaluatorValuePtr>
circt::om::Evaluator::createUnknownValue(Type type, Location loc) {
  using namespace circt::om::evaluator;

  // Create an unknown value of the appropriate type by switching on the type
  auto result =
      TypeSwitch<Type, FailureOr<EvaluatorValuePtr>>(type)
          .Case([&](ListType type) -> FailureOr<EvaluatorValuePtr> {
            // Create an empty list
            return success(std::make_shared<ListValue>(type, loc));
          })
          .Case([&](ClassType type) -> FailureOr<EvaluatorValuePtr> {
            // Look up the class definition
            auto classDef =
                symbolTable.lookup<ClassLike>(type.getClassName().getValue());
            if (!classDef)
              return symbolTable.getOp()->emitError("unknown class name ")
                     << type.getClassName();

            // Create an ObjectValue for both ClassOp and ClassExternOp
            return success(std::make_shared<ObjectValue>(classDef, loc));
          })
          .Case([&](FrozenBasePathType type) -> FailureOr<EvaluatorValuePtr> {
            // Create an empty basepath
            return success(std::make_shared<BasePathValue>(type.getContext()));
          })
          .Case([&](FrozenPathType type) -> FailureOr<EvaluatorValuePtr> {
            // Create an empty path
            return success(
                std::make_shared<PathValue>(PathValue::getEmptyPath(loc)));
          })
          .Default([&](Type type) -> FailureOr<EvaluatorValuePtr> {
            // For all other types (primitives like integer, string,
            // etc.), create an AttributeValue
            return success(AttributeValue::get(type, LocationAttr(loc)));
          });

  // Mark the result as unknown if successful.
  if (succeeded(result))
    result->get()->markUnknown();

  return result;
}

/// Evaluate an unknown value
FailureOr<evaluator::EvaluatorValuePtr>
circt::om::Evaluator::evaluateOp(UnknownValueOp op, Location loc) {
  return createUnknownValue(op.getType(), loc);
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

std::shared_ptr<evaluator::EvaluatorValue>
circt::om::evaluator::AttributeValue::get(Attribute attr, LocationAttr loc) {
  auto type = cast<TypedAttr>(attr).getType();
  auto *context = type.getContext();
  if (!loc)
    loc = UnknownLoc::get(context);

  // Special handling for ListType to create proper ListValue objects instead of
  // AttributeValue objects.
  if (auto listType = dyn_cast<circt::om::ListType>(type)) {
    SmallVector<EvaluatorValuePtr> elements;
    auto listAttr = cast<om::ListAttr>(attr);
    auto values = getEvaluatorValuesFromAttributes(
        listAttr.getContext(), listAttr.getElements().getValue());
    elements.append(values.begin(), values.end());
    auto list = std::make_shared<evaluator::ListValue>(listType, elements, loc);
    return list;
  }

  return std::shared_ptr<AttributeValue>(
      new AttributeValue(PrivateTag{}, attr, loc));
}

std::shared_ptr<evaluator::EvaluatorValue>
circt::om::evaluator::AttributeValue::get(Type type, LocationAttr loc) {
  auto *context = type.getContext();
  if (!loc)
    loc = UnknownLoc::get(context);

  // Special handling for ListType to create proper ListValue objects instead of
  // AttributeValue objects.
  if (auto listType = dyn_cast<circt::om::ListType>(type))
    return std::make_shared<evaluator::ListValue>(listType, loc);
  // Create the AttributeValue with the private tag
  return std::shared_ptr<AttributeValue>(
      new AttributeValue(PrivateTag{}, type, loc));
}
