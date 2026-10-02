//===- SVTypes.cpp - Implement the SV types -------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implement the SV dialect type system.
//
//===----------------------------------------------------------------------===//

#include "circt/Dialect/SV/SVTypes.h"
#include "circt/Dialect/HW/HWTypes.h"
#include "circt/Dialect/SV/SVDialect.h"
#include "circt/Support/LLVM.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/DialectImplementation.h"
#include "llvm/ADT/TypeSwitch.h"

using namespace circt;
using namespace circt::sv;

/// Return the element type of an ArrayType or UnpackedArrayType, or null if the
/// operand isn't an array.
Type circt::sv::getAnyHWArrayElementType(Type type) {
  if (!type)
    return {};
  if (auto array = hw::type_dyn_cast<hw::ArrayType>(type))
    return array.getElementType();
  if (auto array = hw::type_dyn_cast<hw::UnpackedArrayType>(type))
    return array.getElementType();

  return {};
}

/// Return true if the specified type has a packed layout, i.e. it
/// represents a contiguous, bit-sliceable vector: basic bit-vectors, or
/// packed aggregates (hw.array, hw.struct, hw.union) built up entirely out of
/// packed element types.
static bool isPackedType(Type type) {

  if (isa<hw::IntType, IntegerType, hw::EnumType>(type))
    return true;

  if (auto array = dyn_cast<hw::ArrayType>(type))
    return isPackedType(array.getElementType());

  if (auto t = dyn_cast<hw::StructType>(type))
    return llvm::all_of(t.getElements(),
                        [](auto f) { return isPackedType(f.type); });

  if (auto t = dyn_cast<hw::UnionType>(type))
    return llvm::all_of(t.getElements(),
                        [](auto m) { return isPackedType(m.type); });

  return false;
}

//===----------------------------------------------------------------------===//
// InOut type logic.
//===----------------------------------------------------------------------===//

/// Return the element type of an InOutType or null if the operand isn't an
/// InOut type.
mlir::Type circt::sv::getInOutElementType(mlir::Type type) {
  if (auto inout = dyn_cast_or_null<InOutType>(type))
    return inout.getElementType();
  return {};
}

//===----------------------------------------------------------------------===//
// NetType type logic.
//===----------------------------------------------------------------------===//

/// Return the element type of a NetType or null if the operand isn't a Net
/// type.
mlir::Type circt::sv::getNetElementType(mlir::Type type) {
  if (auto net = dyn_cast_or_null<NetType>(type))
    return net.getElementType();
  return {};
}

/// Return the innermost non-unpacked element type.
static Type stripUnpackedDimensions(Type type) {
  type = hw::getCanonicalType(type);
  while (auto uarray = dyn_cast<hw::UnpackedArrayType>(type))
    type = uarray.getElementType();
  return type;
}

/// Return whether a type is valid as the element type of a NetType.
bool circt::sv::isValidNetElementType(Type type) {
  return isPackedType(stripUnpackedDimensions(type));
}

LogicalResult NetType::verify(function_ref<InFlightDiagnostic()> emitError,
                              Type elementType) {
  if (isa_and_present<NetType, VarType>(elementType))
    return emitError() << "sv.net element type may not be itself an sv.net or "
                          "sv.var handle";

  Type packedElementType = stripUnpackedDimensions(elementType);
  if (!isPackedType(packedElementType))
    return emitError()
           << "sv.net element type must have a packed base type, but got "
           << packedElementType;

  return success();
}

//===----------------------------------------------------------------------===//
// VarType type logic.
//===----------------------------------------------------------------------===//

/// Return the element type of a VarType or null if the operand isn't a Var
/// type.
mlir::Type circt::sv::getVarElementType(mlir::Type type) {
  if (auto var = dyn_cast_or_null<VarType>(type))
    return var.getElementType();
  return {};
}

/// Return true if the type can be stored in a variable, including
// simulation-only and unpacked data types that are not valid net elements.
bool circt::sv::isValidVarElementType(Type type) {
  type = hw::getCanonicalType(type);

  if (isPackedType(type) || isa<hw::StringType>(type))
    return true;

  if (auto uarray = dyn_cast<hw::UnpackedArrayType>(type))
    return isValidVarElementType(uarray.getElementType());

  return false;
}

LogicalResult VarType::verify(function_ref<InFlightDiagnostic()> emitError,
                              Type elementType) {
  if (isa_and_present<VarType, NetType>(elementType))
    return emitError() << "sv.var element type may not be itself an sv.var or "
                          "sv.net handle";
  if (!isValidVarElementType(elementType))
    return emitError() << "sv.var element type must be a valid value type, "
                          "but got "
                       << elementType;
  return success();
}

//===----------------------------------------------------------------------===//
// TableGen generated logic.
//===----------------------------------------------------------------------===//

// Provide the autogenerated implementation guts for the Op classes.
#define GET_TYPEDEF_CLASSES
#include "circt/Dialect/SV/SVTypes.cpp.inc"

void SVDialect::registerTypes() {
  addTypes<
#define GET_TYPEDEF_LIST
#include "circt/Dialect/SV/SVTypes.cpp.inc"
      >();
}
