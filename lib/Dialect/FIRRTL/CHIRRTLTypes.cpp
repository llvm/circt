//===- CHIRRTLTypes.cpp - Implement the CHIRRTL dialect -----------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements the CHIRRTL dialect type system.
//
//===----------------------------------------------------------------------===//

#include "circt/Dialect/FIRRTL/CHIRRTLDialect.h"
#include "circt/Dialect/FIRRTL/FIRRTLAttributes.h"
#include "circt/Dialect/FIRRTL/FIRRTLDialect.h"
#include "circt/Dialect/FIRRTL/FIRRTLTypes.h"
#include "circt/Dialect/HW/HWAttributes.h"
#include "circt/Support/LLVM.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/OpDefinition.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/TypeSwitch.h"

using namespace mlir;
using namespace circt;
using namespace chirrtl;
using namespace firrtl;

#define GET_TYPEDEF_CLASSES
#include "circt/Dialect/FIRRTL/CHIRRTLTypes.cpp.inc"

//===----------------------------------------------------------------------===//
// CMemory Type
//===----------------------------------------------------------------------===//

void CMemoryType::print(AsmPrinter &printer) const {
  printer << "<";
  // Don't print element types with "!firrtl.".
  firrtl::printNestedType(getElementType(), printer);
  printer << ", " << getNumElements() << ">";
}

Type CMemoryType::parse(AsmParser &parser) {
  FIRRTLBaseType elementType;
  uint64_t numElements;
  if (parser.parseLess() || firrtl::parseNestedBaseType(elementType, parser) ||
      parser.parseComma() || parser.parseInteger(numElements) ||
      parser.parseGreater())
    return {};
  return parser.getChecked<CMemoryType>(elementType, numElements);
}

LogicalResult CMemoryType::verify(function_ref<InFlightDiagnostic()> emitError,
                                  FIRRTLBaseType elementType,
                                  uint64_t numElements) {
  if (!elementType.isPassive()) {
    return emitError() << "behavioral memory element type must be passive";
  }
  return success();
}

//===----------------------------------------------------------------------===//
// CHIRRTLDialect
//===----------------------------------------------------------------------===//

void CHIRRTLDialect::registerTypes() {
  addTypes<
#define GET_TYPEDEF_LIST
#include "circt/Dialect/FIRRTL/CHIRRTLTypes.cpp.inc"
      >();
}
