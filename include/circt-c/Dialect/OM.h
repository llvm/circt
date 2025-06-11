//===- OM.h - C interface for the OM dialect ----------------------*- C -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_C_DIALECT_OM_H
#define CIRCT_C_DIALECT_OM_H

#include "mlir-c/IR.h"

#ifdef __cplusplus
extern "C" {
#endif

//===----------------------------------------------------------------------===//
// Dialect API.
//===----------------------------------------------------------------------===//

MLIR_DECLARE_CAPI_DIALECT_REGISTRATION(OM, om);

//===----------------------------------------------------------------------===//
// Type API.
//===----------------------------------------------------------------------===//

/// Is the Type an AnyType.
MLIR_CAPI_EXPORTED bool omTypeIsAAnyType(MlirType type);

/// Get the TypeID for an AnyType.
MLIR_CAPI_EXPORTED MlirTypeID omAnyTypeGetTypeID(void);

/// Is the Type a ClassType.
MLIR_CAPI_EXPORTED bool omTypeIsAClassType(MlirType type);

/// Get the TypeID for a ClassType.
MLIR_CAPI_EXPORTED MlirTypeID omClassTypeGetTypeID(void);

/// Get the name for a ClassType.
MLIR_CAPI_EXPORTED MlirIdentifier omClassTypeGetName(MlirType type);

/// Is the Type a FrozenBasePathType.
MLIR_CAPI_EXPORTED bool omTypeIsAFrozenBasePathType(MlirType type);

/// Get the TypeID for a FrozenBasePathType.
MLIR_CAPI_EXPORTED MlirTypeID omFrozenBasePathTypeGetTypeID(void);

/// Is the Type a FrozenPathType.
MLIR_CAPI_EXPORTED bool omTypeIsAFrozenPathType(MlirType type);

/// Get the TypeID for a FrozenPathType.
MLIR_CAPI_EXPORTED MlirTypeID omFrozenPathTypeGetTypeID(void);

/// Is the Type a ListType.
MLIR_CAPI_EXPORTED bool omTypeIsAListType(MlirType type);

/// Get the TypeID for a ListType.
MLIR_CAPI_EXPORTED MlirTypeID omListTypeGetTypeID(void);

// Return a element type of a ListType.
MLIR_CAPI_EXPORTED MlirType omListTypeGetElementType(MlirType type);

/// Is the Type a MapType.
MLIR_CAPI_EXPORTED bool omTypeIsAMapType(MlirType type);

// Return a key type of a MapType.
MLIR_CAPI_EXPORTED MlirType omMapTypeGetKeyType(MlirType type);

/// Is the Type a StringType.
MLIR_CAPI_EXPORTED bool omTypeIsAStringType(MlirType type);

/// Get the TypeID for a StringType.
MLIR_CAPI_EXPORTED MlirTypeID omStringTypeGetTypeID(void);

/// Get a StringType.
MLIR_CAPI_EXPORTED MlirType omStringTypeGet(MlirContext ctx);

//===----------------------------------------------------------------------===//
// Evaluator data structures.
//===----------------------------------------------------------------------===//

/// A value type for use in C APIs that just wraps a pointer to an Evaluator.
/// This is in line with the usual MLIR DEFINE_C_API_STRUCT.
struct OMEvaluator {
  void *ptr;
};

// clang-tidy doesn't respect extern "C".
// see https://github.com/llvm/llvm-project/issues/35272.
// NOLINTNEXTLINE(modernize-use-using)
typedef struct OMEvaluator OMEvaluator;

/// A value type for use in C APIs that just wraps a pointer to an Object.
/// This is in line with the usual MLIR DEFINE_C_API_STRUCT.
struct OMEvaluatorValue {
  void *ptr;
};

// clang-tidy doesn't respect extern "C".
// see https://github.com/llvm/llvm-project/issues/35272.
// NOLINTNEXTLINE(modernize-use-using)
typedef struct OMEvaluatorValue OMEvaluatorValue;

//===----------------------------------------------------------------------===//
// Evaluator API.
//===----------------------------------------------------------------------===//

/// Construct an Evaluator with an IR module.
MLIR_CAPI_EXPORTED OMEvaluator omEvaluatorNew(MlirModule mod);

/// Use the Evaluator to Instantiate an Object from its class name and actual
/// parameters.
MLIR_CAPI_EXPORTED OMEvaluatorValue
omEvaluatorInstantiate(OMEvaluator evaluator, MlirAttribute className,
                       intptr_t nActualParams, OMEvaluatorValue *actualParams);

/// Get the Module the Evaluator is built from.
MLIR_CAPI_EXPORTED MlirModule omEvaluatorGetModule(OMEvaluator evaluator);

//===----------------------------------------------------------------------===//
// Object API.
//===----------------------------------------------------------------------===//

/// Query if the Object is null.
MLIR_CAPI_EXPORTED bool omEvaluatorObjectIsNull(OMEvaluatorValue object);

/// Get the Type from an Object, which will be a ClassType.
MLIR_CAPI_EXPORTED MlirType omEvaluatorObjectGetType(OMEvaluatorValue object);

/// Get a field from an Object, which must contain a field of that name.
MLIR_CAPI_EXPORTED OMEvaluatorValue
omEvaluatorObjectGetField(OMEvaluatorValue object, MlirAttribute name);

/// Get the object hash.
MLIR_CAPI_EXPORTED unsigned omEvaluatorObjectGetHash(OMEvaluatorValue object);

/// Check equality of two objects.
MLIR_CAPI_EXPORTED bool omEvaluatorObjectIsEq(OMEvaluatorValue object,
                                              OMEvaluatorValue other);

/// Get all the field names from an Object, can be empty if object has no
/// fields.
MLIR_CAPI_EXPORTED MlirAttribute
omEvaluatorObjectGetFieldNames(OMEvaluatorValue object);

//===----------------------------------------------------------------------===//
// EvaluatorValue API.
//===----------------------------------------------------------------------===//

// Get a context from an EvaluatorValue.
MLIR_CAPI_EXPORTED MlirContext
omEvaluatorValueGetContext(OMEvaluatorValue evaluatorValue);

// Get Location from an EvaluatorValue.
MLIR_CAPI_EXPORTED MlirLocation
omEvaluatorValueGetLoc(OMEvaluatorValue evaluatorValue);

// Query if the EvaluatorValue is null.
MLIR_CAPI_EXPORTED bool omEvaluatorValueIsNull(OMEvaluatorValue evaluatorValue);

/// Query if the EvaluatorValue is an Object.
MLIR_CAPI_EXPORTED bool
omEvaluatorValueIsAObject(OMEvaluatorValue evaluatorValue);

/// Query if the EvaluatorValue is a Primitive.
MLIR_CAPI_EXPORTED bool
omEvaluatorValueIsAPrimitive(OMEvaluatorValue evaluatorValue);

/// Get the Primitive from an EvaluatorValue, which must contain a Primitive.
MLIR_CAPI_EXPORTED MlirAttribute
omEvaluatorValueGetPrimitive(OMEvaluatorValue evaluatorValue);

/// Get the EvaluatorValue from a Primitive value.
MLIR_CAPI_EXPORTED OMEvaluatorValue
omEvaluatorValueFromPrimitive(MlirAttribute primitive);

/// Query if the EvaluatorValue is an Object.
MLIR_CAPI_EXPORTED bool
omEvaluatorValueIsAList(OMEvaluatorValue evaluatorValue);

/// Get the length of the list.
MLIR_CAPI_EXPORTED intptr_t
omEvaluatorListGetNumElements(OMEvaluatorValue evaluatorValue);

/// Get an element of the list.
MLIR_CAPI_EXPORTED OMEvaluatorValue
omEvaluatorListGetElement(OMEvaluatorValue evaluatorValue, intptr_t pos);

/// Query if the EvaluatorValue is a Tuple.
MLIR_CAPI_EXPORTED bool
omEvaluatorValueIsATuple(OMEvaluatorValue evaluatorValue);

/// Get the size of the tuple.
MLIR_CAPI_EXPORTED intptr_t
omEvaluatorTupleGetNumElements(OMEvaluatorValue evaluatorValue);

/// Get an element of the tuple.
MLIR_CAPI_EXPORTED OMEvaluatorValue
omEvaluatorTupleGetElement(OMEvaluatorValue evaluatorValue, intptr_t pos);

/// Get an element of the map.
MLIR_CAPI_EXPORTED OMEvaluatorValue
omEvaluatorMapGetElement(OMEvaluatorValue evaluatorValue, MlirAttribute attr);

MLIR_CAPI_EXPORTED MlirAttribute omEvaluatorMapGetKeys(OMEvaluatorValue object);

/// Query if the EvaluatorValue is a Map.
MLIR_CAPI_EXPORTED bool omEvaluatorValueIsAMap(OMEvaluatorValue evaluatorValue);

/// Get the Type from a Map, which will be a MapType.
MLIR_CAPI_EXPORTED MlirType
omEvaluatorMapGetType(OMEvaluatorValue evaluatorValue);

/// Query if the EvaluatorValue is a BasePath.
MLIR_CAPI_EXPORTED bool
omEvaluatorValueIsABasePath(OMEvaluatorValue evaluatorValue);

/// Create an empty BasePath.
MLIR_CAPI_EXPORTED OMEvaluatorValue
omEvaluatorBasePathGetEmpty(MlirContext context);

/// Query if the EvaluatorValue is a Path.
MLIR_CAPI_EXPORTED bool
omEvaluatorValueIsAPath(OMEvaluatorValue evaluatorValue);

/// Get a string representation of a Path.
MLIR_CAPI_EXPORTED MlirAttribute
omEvaluatorPathGetAsString(OMEvaluatorValue evaluatorValue);

/// Query if the EvaluatorValue is a Reference.
MLIR_CAPI_EXPORTED bool
omEvaluatorValueIsAReference(OMEvaluatorValue evaluatorValue);

/// Dereference a Reference EvaluatorValue. Emits an error and returns null if
/// the Reference cannot be dereferenced.
MLIR_CAPI_EXPORTED OMEvaluatorValue
omEvaluatorValueGetReferenceValue(OMEvaluatorValue evaluatorValue);

//===----------------------------------------------------------------------===//
// ReferenceAttr API
//===----------------------------------------------------------------------===//

MLIR_CAPI_EXPORTED bool omAttrIsAReferenceAttr(MlirAttribute attr);

MLIR_CAPI_EXPORTED MlirAttribute omReferenceAttrGetInnerRef(MlirAttribute attr);

//===----------------------------------------------------------------------===//
// IntegerAttr API
//===----------------------------------------------------------------------===//

MLIR_CAPI_EXPORTED bool omAttrIsAIntegerAttr(MlirAttribute attr);

/// Given an om::IntegerAttr, return the mlir::IntegerAttr.
MLIR_CAPI_EXPORTED MlirAttribute omIntegerAttrGetInt(MlirAttribute attr);

/// Get an om::IntegerAttr from mlir::IntegerAttr.
MLIR_CAPI_EXPORTED MlirAttribute omIntegerAttrGet(MlirAttribute attr);

/// Get a string representation of an om::IntegerAttr.
MLIR_CAPI_EXPORTED MlirStringRef omIntegerAttrToString(MlirAttribute attr);

//===----------------------------------------------------------------------===//
// ListAttr API
//===----------------------------------------------------------------------===//

MLIR_CAPI_EXPORTED bool omAttrIsAListAttr(MlirAttribute attr);

MLIR_CAPI_EXPORTED intptr_t omListAttrGetNumElements(MlirAttribute attr);

MLIR_CAPI_EXPORTED MlirAttribute omListAttrGetElement(MlirAttribute attr,
                                                      intptr_t pos);

MLIR_CAPI_EXPORTED MlirAttribute omListAttrGet(MlirType elementType,
                                               intptr_t numElements,
                                               const MlirAttribute *elements);

//===----------------------------------------------------------------------===//
// MapAttr API
//===----------------------------------------------------------------------===//

MLIR_CAPI_EXPORTED bool omAttrIsAMapAttr(MlirAttribute attr);

MLIR_CAPI_EXPORTED intptr_t omMapAttrGetNumElements(MlirAttribute attr);

MLIR_CAPI_EXPORTED MlirIdentifier omMapAttrGetElementKey(MlirAttribute attr,
                                                         intptr_t pos);

MLIR_CAPI_EXPORTED MlirAttribute omMapAttrGetElementValue(MlirAttribute attr,
                                                          intptr_t pos);

#ifdef __cplusplus
}
#endif

#endif // CIRCT_C_DIALECT_OM_H
