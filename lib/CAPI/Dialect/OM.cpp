//===- OM.cpp - C interface for the OM dialect ----------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "circt-c/Dialect/OM.h"
#include "circt/Dialect/OM/Evaluator/Evaluator.h"
#include "circt/Dialect/OM/OMAttributes.h"
#include "circt/Dialect/OM/OMDialect.h"
#include "mlir/CAPI/Registration.h"
#include "mlir/CAPI/Wrap.h"
#include "mlir/IR/Location.h"
#include "llvm/ADT/Hashing.h"
#include "llvm/Support/Casting.h"

using namespace mlir;
using namespace circt::om;

//===----------------------------------------------------------------------===//
// Dialect API.
//===----------------------------------------------------------------------===//

MLIR_DEFINE_CAPI_DIALECT_REGISTRATION(OM, om, OMDialect)

//===----------------------------------------------------------------------===//
// Type API.
//===----------------------------------------------------------------------===//

/// Is the Type an AnyType.
bool omTypeIsAAnyType(MlirType type) { return isa<AnyType>(unwrap(type)); }

/// Get the TypeID for an AnyType.
MlirTypeID omAnyTypeGetTypeID(void) { return wrap(AnyType::getTypeID()); }

/// Is the Type a ClassType.
bool omTypeIsAClassType(MlirType type) { return isa<ClassType>(unwrap(type)); }

/// Get the TypeID for a ClassType.
MlirTypeID omClassTypeGetTypeID() { return wrap(ClassType::getTypeID()); }

/// Get the name for a ClassType.
MlirIdentifier omClassTypeGetName(MlirType type) {
  return wrap(cast<ClassType>(unwrap(type)).getClassName().getAttr());
}

/// Is the Type a FrozenBasePathType.
bool omTypeIsAFrozenBasePathType(MlirType type) {
  return isa<FrozenBasePathType>(unwrap(type));
}

/// Get the TypeID for a FrozenBasePathType.
MlirTypeID omFrozenBasePathTypeGetTypeID(void) {
  return wrap(FrozenBasePathType::getTypeID());
}

/// Is the Type a FrozenPathType.
bool omTypeIsAFrozenPathType(MlirType type) {
  return isa<FrozenPathType>(unwrap(type));
}

/// Get the TypeID for a FrozenPathType.
MlirTypeID omFrozenPathTypeGetTypeID(void) {
  return wrap(FrozenPathType::getTypeID());
}

/// Is the Type a ListType.
bool omTypeIsAListType(MlirType type) { return isa<ListType>(unwrap(type)); }

/// Get the TypeID for a ListType.
MlirTypeID omListTypeGetTypeID(void) { return wrap(ListType::getTypeID()); }

// Return a element type of a ListType.
MlirType omListTypeGetElementType(MlirType type) {
  return wrap(cast<ListType>(unwrap(type)).getElementType());
}

/// Is the Type a StringType.
bool omTypeIsAStringType(MlirType type) {
  return isa<StringType>(unwrap(type));
}

/// Get a StringType.
MlirType omStringTypeGet(MlirContext ctx) {
  return wrap(StringType::get(unwrap(ctx)));
}

/// Get the TypeID for a StringType.
MlirTypeID omStringTypeGetTypeID(void) { return wrap(StringType::getTypeID()); }

/// Is the Type a MapType.
bool omTypeIsAMapType(MlirType type) { return isa<MapType>(unwrap(type)); }

/// Return a key type of a map.
MlirType omMapTypeGetKeyType(MlirType type) {
  return wrap(cast<MapType>(unwrap(type)).getKeyType());
}

//===----------------------------------------------------------------------===//
// Evaluator data structures.
//===----------------------------------------------------------------------===//

DEFINE_C_API_PTR_METHODS(OMEvaluator, circt::om::Evaluator)

/// Define our own wrap and unwrap instead of using the usual macro. This is To
/// handle the std::shared_ptr reference counts appropriately. We want to always
/// create *new* shared pointers to the EvaluatorValue when we wrap it for C, to
/// increment the reference count. We want to use the shared_from_this
/// functionality to ensure it is unwrapped into C++ with the correct reference
/// count.

static inline OMEvaluatorValue wrap(EvaluatorValuePtr object) {
  return OMEvaluatorValue{
      static_cast<void *>((new EvaluatorValuePtr(std::move(object)))->get())};
}

static inline EvaluatorValuePtr unwrap(OMEvaluatorValue c) {
  return static_cast<evaluator::EvaluatorValue *>(c.ptr)->shared_from_this();
}

//===----------------------------------------------------------------------===//
// Evaluator API.
//===----------------------------------------------------------------------===//

/// Construct an Evaluator with an IR module.
OMEvaluator omEvaluatorNew(MlirModule mod) {
  // Just allocate and wrap the Evaluator.
  return wrap(new Evaluator(unwrap(mod)));
}

/// Use the Evaluator to Instantiate an Object from its class name and actual
/// parameters.
OMEvaluatorValue omEvaluatorInstantiate(OMEvaluator evaluator,
                                        MlirAttribute className,
                                        intptr_t nActualParams,
                                        OMEvaluatorValue *actualParams) {
  // Unwrap the Evaluator.
  Evaluator *cppEvaluator = unwrap(evaluator);

  // Unwrap the className, which the client must supply as a StringAttr.
  StringAttr cppClassName = cast<StringAttr>(unwrap(className));

  // Unwrap the actual parameters.
  SmallVector<std::shared_ptr<evaluator::EvaluatorValue>> cppActualParams;
  for (unsigned i = 0; i < nActualParams; i++)
    cppActualParams.push_back(unwrap(actualParams[i]));

  // Invoke the Evaluator to instantiate the Object.
  auto result = cppEvaluator->instantiate(cppClassName, cppActualParams);

  // If instantiation failed, return a null Object. A Diagnostic will be emitted
  // in this case.
  if (failed(result))
    return OMEvaluatorValue();

  // Wrap and return the Object.
  return wrap(result.value());
}

/// Get the Module the Evaluator is built from.
MlirModule omEvaluatorGetModule(OMEvaluator evaluator) {
  // Just unwrap the Evaluator, get the Module, and wrap it.
  return wrap(unwrap(evaluator)->getModule());
}

//===----------------------------------------------------------------------===//
// Object API.
//===----------------------------------------------------------------------===//

/// Query if the Object is null.
bool omEvaluatorObjectIsNull(OMEvaluatorValue object) {
  // Just check if the Object shared pointer is null.
  return !object.ptr;
}

/// Get the Type from an Object, which will be a ClassType.
MlirType omEvaluatorObjectGetType(OMEvaluatorValue object) {
  return wrap(llvm::cast<Object>(unwrap(object).get())->getType());
}

/// Get the hash for the object.
unsigned omEvaluatorObjectGetHash(OMEvaluatorValue object) {
  return llvm::hash_value(llvm::cast<Object>(unwrap(object).get()));
}

/// Check if two objects are same.
bool omEvaluatorObjectIsEq(OMEvaluatorValue object, OMEvaluatorValue other) {
  return llvm::cast<Object>(unwrap(object).get()) ==
         llvm::cast<Object>(unwrap(other).get());
}

/// Get an ArrayAttr with the names of the fields in an Object.
MlirAttribute omEvaluatorObjectGetFieldNames(OMEvaluatorValue object) {
  return wrap(llvm::cast<Object>(unwrap(object).get())->getFieldNames());
}

MlirType omEvaluatorMapGetType(OMEvaluatorValue value) {
  return wrap(llvm::cast<evaluator::MapValue>(unwrap(value).get())->getType());
}

/// Get an ArrayAttr with the keys in a Map.
MlirAttribute omEvaluatorMapGetKeys(OMEvaluatorValue object) {
  return wrap(llvm::cast<evaluator::MapValue>(unwrap(object).get())->getKeys());
}

/// Get a field from an Object, which must contain a field of that name.
OMEvaluatorValue omEvaluatorObjectGetField(OMEvaluatorValue object,
                                           MlirAttribute name) {
  // Unwrap the Object and get the field of the name, which the client must
  // supply as a StringAttr.
  FailureOr<EvaluatorValuePtr> result =
      llvm::cast<Object>(unwrap(object).get())
          ->getField(cast<StringAttr>(unwrap(name)));

  // If getField failed, return a null EvaluatorValue. A Diagnostic will be
  // emitted in this case.
  if (failed(result))
    return OMEvaluatorValue();

  return OMEvaluatorValue{wrap(result.value())};
}

//===----------------------------------------------------------------------===//
// EvaluatorValue API.
//===----------------------------------------------------------------------===//

// Get a context from an EvaluatorValue.
MlirContext omEvaluatorValueGetContext(OMEvaluatorValue evaluatorValue) {
  return wrap(unwrap(evaluatorValue)->getContext());
}

// Get location from an EvaluatorValue.
MlirLocation omEvaluatorValueGetLoc(OMEvaluatorValue evaluatorValue) {
  return wrap(unwrap(evaluatorValue)->getLoc());
}

// Query if the EvaluatorValue is null.
bool omEvaluatorValueIsNull(OMEvaluatorValue evaluatorValue) {
  // Check if the pointer is null.
  return !evaluatorValue.ptr;
}

/// Query if the EvaluatorValue is an Object.
bool omEvaluatorValueIsAObject(OMEvaluatorValue evaluatorValue) {
  // Check if the Object is non-null.
  return isa<evaluator::ObjectValue>(unwrap(evaluatorValue).get());
}

/// Query if the EvaluatorValue is a Primitive.
bool omEvaluatorValueIsAPrimitive(OMEvaluatorValue evaluatorValue) {
  // Check if the Attribute is non-null.
  return isa<evaluator::AttributeValue>(unwrap(evaluatorValue).get());
}

/// Get the Primitive from an EvaluatorValue, which must contain a Primitive.
MlirAttribute omEvaluatorValueGetPrimitive(OMEvaluatorValue evaluatorValue) {
  // Assert the Attribute is non-null, and return it.
  assert(omEvaluatorValueIsAPrimitive(evaluatorValue));
  return wrap(
      llvm::cast<evaluator::AttributeValue>(unwrap(evaluatorValue).get())
          ->getAttr());
}

/// Get the Primitive from an EvaluatorValue, which must contain a Primitive.
OMEvaluatorValue omEvaluatorValueFromPrimitive(MlirAttribute primitive) {
  // Assert the Attribute is non-null, and return it.
  return wrap(std::make_shared<evaluator::AttributeValue>(unwrap(primitive)));
}

/// Query if the EvaluatorValue is a List.
bool omEvaluatorValueIsAList(OMEvaluatorValue evaluatorValue) {
  return isa<evaluator::ListValue>(unwrap(evaluatorValue).get());
}

/// Get the List from an EvaluatorValue, which must contain a List.
/// TODO: This can be removed.
OMEvaluatorValue omEvaluatorValueGetList(OMEvaluatorValue evaluatorValue) {
  // Assert the List is non-null, and return it.
  assert(omEvaluatorValueIsAList(evaluatorValue));
  return evaluatorValue;
}

/// Get the length of the List.
intptr_t omEvaluatorListGetNumElements(OMEvaluatorValue evaluatorValue) {
  return cast<evaluator::ListValue>(unwrap(evaluatorValue).get())
      ->getElements()
      .size();
}

/// Get an element of the List.
OMEvaluatorValue omEvaluatorListGetElement(OMEvaluatorValue evaluatorValue,
                                           intptr_t pos) {
  return wrap(cast<evaluator::ListValue>(unwrap(evaluatorValue).get())
                  ->getElements()[pos]);
}

/// Query if the EvaluatorValue is a Tuple.
bool omEvaluatorValueIsATuple(OMEvaluatorValue evaluatorValue) {
  return isa<evaluator::TupleValue>(unwrap(evaluatorValue).get());
}

/// Get the length of the Tuple.
intptr_t omEvaluatorTupleGetNumElements(OMEvaluatorValue evaluatorValue) {
  return cast<evaluator::TupleValue>(unwrap(evaluatorValue).get())
      ->getElements()
      .size();
}

/// Get an element of the Tuple.
OMEvaluatorValue omEvaluatorTupleGetElement(OMEvaluatorValue evaluatorValue,
                                            intptr_t pos) {
  return wrap(cast<evaluator::TupleValue>(unwrap(evaluatorValue).get())
                  ->getElements()[pos]);
}

/// Get an element of the Map.
OMEvaluatorValue omEvaluatorMapGetElement(OMEvaluatorValue evaluatorValue,
                                          MlirAttribute attr) {
  const auto &elements =
      cast<evaluator::MapValue>(unwrap(evaluatorValue).get())->getElements();
  const auto &it = elements.find(unwrap(attr));
  if (it != elements.end())
    return wrap(it->second);
  return OMEvaluatorValue{nullptr};
}

/// Query if the EvaluatorValue is a map.
bool omEvaluatorValueIsAMap(OMEvaluatorValue evaluatorValue) {
  return isa<evaluator::MapValue>(unwrap(evaluatorValue).get());
}

bool omEvaluatorValueIsABasePath(OMEvaluatorValue evaluatorValue) {
  return isa<evaluator::BasePathValue>(unwrap(evaluatorValue).get());
}

OMEvaluatorValue omEvaluatorBasePathGetEmpty(MlirContext context) {
  return wrap(std::make_shared<evaluator::BasePathValue>(unwrap(context)));
}

bool omEvaluatorValueIsAPath(OMEvaluatorValue evaluatorValue) {
  return isa<evaluator::PathValue>(unwrap(evaluatorValue).get());
}

MlirAttribute omEvaluatorPathGetAsString(OMEvaluatorValue evaluatorValue) {
  const auto *path = cast<evaluator::PathValue>(unwrap(evaluatorValue).get());
  return wrap((Attribute)path->getAsString());
}

/// Query if the EvaluatorValue is a Reference.
bool omEvaluatorValueIsAReference(OMEvaluatorValue evaluatorValue) {
  return isa<evaluator::ReferenceValue>(unwrap(evaluatorValue).get());
}

/// Dereference a Reference EvaluatorValue. Emits an error and returns null if
/// the Reference cannot be dereferenced.
OMEvaluatorValue
omEvaluatorValueGetReferenceValue(OMEvaluatorValue evaluatorValue) {
  // Assert the EvaluatorValue is a Reference.
  assert(omEvaluatorValueIsAReference(evaluatorValue));

  // Attempt to get the final EvaluatorValue from the Reference.
  auto result =
      llvm::cast<evaluator::ReferenceValue>(unwrap(evaluatorValue).get())
          ->getStrippedValue();

  // If this failed, an error diagnostic has been emitted, and we return null.
  if (failed(result))
    return {};

  // If this succeeded, wrap the EvaluatorValue and return it.
  return wrap(result.value());
}

//===----------------------------------------------------------------------===//
// ReferenceAttr API.
//===----------------------------------------------------------------------===//

bool omAttrIsAReferenceAttr(MlirAttribute attr) {
  return isa<ReferenceAttr>(unwrap(attr));
}

MlirAttribute omReferenceAttrGetInnerRef(MlirAttribute referenceAttr) {
  return wrap(
      (Attribute)cast<ReferenceAttr>(unwrap(referenceAttr)).getInnerRef());
}

//===----------------------------------------------------------------------===//
// IntegerAttr API.
//===----------------------------------------------------------------------===//

bool omAttrIsAIntegerAttr(MlirAttribute attr) {
  return isa<circt::om::IntegerAttr>(unwrap(attr));
}

MlirAttribute omIntegerAttrGetInt(MlirAttribute attr) {
  return wrap(cast<circt::om::IntegerAttr>(unwrap(attr)).getValue());
}

MlirAttribute omIntegerAttrGet(MlirAttribute attr) {
  auto integerAttr = cast<mlir::IntegerAttr>(unwrap(attr));
  return wrap(
      circt::om::IntegerAttr::get(integerAttr.getContext(), integerAttr));
}

/// Get a string representation of an om::IntegerAttr.
MlirStringRef omIntegerAttrToString(MlirAttribute attr) {
  mlir::IntegerAttr integerAttr =
      cast<circt::om::IntegerAttr>(unwrap(attr)).getValue();
  SmallVector<char> str;
  integerAttr.getValue().toString(
      str, /*Radix=*/10, /*Signed=*/integerAttr.getType().isSignedInteger());
  return wrap(StringAttr::get(integerAttr.getContext(), str).getValue());
}

//===----------------------------------------------------------------------===//
// ListAttr API.
//===----------------------------------------------------------------------===//

bool omAttrIsAListAttr(MlirAttribute attr) {
  return isa<ListAttr>(unwrap(attr));
}

intptr_t omListAttrGetNumElements(MlirAttribute attr) {
  auto listAttr = llvm::cast<ListAttr>(unwrap(attr));
  return static_cast<intptr_t>(listAttr.getElements().size());
}

MlirAttribute omListAttrGetElement(MlirAttribute attr, intptr_t pos) {
  auto listAttr = llvm::cast<ListAttr>(unwrap(attr));
  return wrap(listAttr.getElements()[pos]);
}

MlirAttribute omListAttrGet(MlirType elementType, intptr_t numElements,
                            const MlirAttribute *elements) {
  SmallVector<Attribute, 8> attrs;
  (void)unwrapList(static_cast<size_t>(numElements), elements, attrs);
  auto type = unwrap(elementType);
  auto *ctx = type.getContext();
  return wrap(ListAttr::get(ctx, type, ArrayAttr::get(ctx, attrs)));
}

//===----------------------------------------------------------------------===//
// MapAttr API.
//===----------------------------------------------------------------------===//

bool omAttrIsAMapAttr(MlirAttribute attr) { return isa<MapAttr>(unwrap(attr)); }

intptr_t omMapAttrGetNumElements(MlirAttribute attr) {
  auto mapAttr = llvm::cast<MapAttr>(unwrap(attr));
  return static_cast<intptr_t>(mapAttr.getElements().size());
}

MlirIdentifier omMapAttrGetElementKey(MlirAttribute attr, intptr_t pos) {
  auto mapAttr = llvm::cast<MapAttr>(unwrap(attr));
  return wrap(mapAttr.getElements().getValue()[pos].getName());
}

MlirAttribute omMapAttrGetElementValue(MlirAttribute attr, intptr_t pos) {
  auto mapAttr = llvm::cast<MapAttr>(unwrap(attr));
  return wrap(mapAttr.getElements().getValue()[pos].getValue());
}
