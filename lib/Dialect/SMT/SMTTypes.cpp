//===- SMTTypes.cpp -------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "circt/Dialect/SMT/SMTTypes.h"
#include "circt/Dialect/SMT/SMTDialect.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/DialectImplementation.h"
#include "llvm/ADT/TypeSwitch.h"

using namespace circt;
using namespace smt;
using namespace mlir;

#define GET_TYPEDEF_CLASSES
#include "circt/Dialect/SMT/SMTTypes.cpp.inc"

void SMTDialect::registerTypes() {
  addTypes<
#define GET_TYPEDEF_LIST
#include "circt/Dialect/SMT/SMTTypes.cpp.inc"
      >();
}

bool smt::isAnySMTValueType(Type type) {
  return isa<BoolType, BitVectorType, ArrayType>(type);
}

//===----------------------------------------------------------------------===//
// BitVectorType
//===----------------------------------------------------------------------===//

LogicalResult
BitVectorType::verify(function_ref<InFlightDiagnostic()> emitError,
                      unsigned width) {
  if (width <= 0)
    return emitError() << "bit-vector must have at least a width of one";
  return success();
}

//===----------------------------------------------------------------------===//
// ArrayType
//===----------------------------------------------------------------------===//

LogicalResult ArrayType::verify(function_ref<InFlightDiagnostic()> emitError,
                                Type domainType, Type rangeType) {
  if (!isAnySMTValueType(domainType))
    return emitError() << "domain must be any SMT value type";
  if (!isAnySMTValueType(rangeType))
    return emitError() << "range must be any SMT value type";

  return success();
}
