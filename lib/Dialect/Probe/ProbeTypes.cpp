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

LogicalResult RefType::verify(function_ref<InFlightDiagnostic()> emitError,
                              Type elementType) {
  if (!hw::isHWValueType(elementType) || hw::hasHWInOutType(elementType))
    return emitError()
           << "probe element type must be a non-inout HW value type, got "
           << elementType;
  return success();
}

void ProbeDialect::registerTypes() {
  addTypes<
#define GET_TYPEDEF_LIST
#include "circt/Dialect/Probe/ProbeTypes.cpp.inc"
      >();
}
