//===- ProbeTypes.cpp - Probe dialect types -------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "circt/Dialect/Probe/ProbeTypes.h"
#include "circt/Dialect/HW/HWTypes.h"
#include "circt/Dialect/Probe/ProbeDialect.h"
#include "circt/Dialect/Seq/SeqTypes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/DialectImplementation.h"

using namespace circt;
using namespace probe;
using namespace mlir;

//===----------------------------------------------------------------------===//
// TableGen
//===----------------------------------------------------------------------===//

#define GET_TYPEDEF_CLASSES
#include "circt/Dialect/Probe/ProbeTypes.cpp.inc"

bool probe::isProbeElementType(Type type) {
  if (!type || hw::hasHWInOutType(type))
    return false;

  if (isa<seq::ClockType>(type))
    return true;

  if (auto alias = dyn_cast<hw::TypeAliasType>(type))
    return isProbeElementType(alias.getCanonicalType());

  if (auto array = dyn_cast<hw::ArrayType>(type))
    return isProbeElementType(array.getElementType());

  if (auto array = dyn_cast<hw::UnpackedArrayType>(type))
    return isProbeElementType(array.getElementType());

  if (auto structType = dyn_cast<hw::StructType>(type))
    return llvm::all_of(structType.getElements(), [](auto field) {
      return isProbeElementType(field.type);
    });

  if (auto unionType = dyn_cast<hw::UnionType>(type))
    return llvm::all_of(unionType.getElements(), [](auto field) {
      return isProbeElementType(field.type);
    });

  return hw::isHWValueType(type);
}

LogicalResult RefType::verify(function_ref<InFlightDiagnostic()> emitError,
                              Type elementType) {
  if (!isProbeElementType(elementType))
    return emitError()
           << "probe element type must be a non-inout type containing only HW "
              "value types or seq.clock leaves, got "
           << elementType;
  return success();
}

void ProbeDialect::registerTypes() {
  addTypes<
#define GET_TYPEDEF_LIST
#include "circt/Dialect/Probe/ProbeTypes.cpp.inc"
      >();
}
