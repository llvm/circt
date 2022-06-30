//===- OFIRDialect.cpp - Implement the OFIR dialect -----------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements the OFIR dialect.
//
//===----------------------------------------------------------------------===//

#include "circt/Dialect/OFIR/OFIRDialect.h"
#include "circt/Dialect/OFIR/OFIRAttributes.h"
#include "circt/Dialect/OFIR/OFIROps.h"
//#include "circt/Support/FieldRef.h"
#include "mlir/IR/DialectImplementation.h"
#include "llvm/ADT/APSInt.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/TypeSwitch.h"

using namespace circt;
using namespace ofir;

//===----------------------------------------------------------------------===//
// Dialect specification.
//===----------------------------------------------------------------------===//

// Pull in the dialect definition.
#include "circt/Dialect/OFIR/OFIRDialect.cpp.inc"

void OFIRDialect::initialize() {
  // Register types and attributes.
  //registerTypes();
  //registerAttributes();

  // Register operations.
  addOperations<
#define GET_OP_LIST
#include "circt/Dialect/OFIR/OFIR.cpp.inc"
      >();
}

/// Registered hook to materialize a single constant operation from a given
/// attribute value with the desired resultant type. This method should use
/// the provided builder to create the operation without changing the
/// insertion position. The generated operation is expected to be constant
/// like, i.e. single result, zero operands, non side-effecting, etc. On
/// success, this hook should return the value generated to represent the
/// constant value. Otherwise, it should return null on failure.
Operation *OFIRDialect::materializeConstant(OpBuilder &builder,
                                              Attribute value, Type type,
                                              Location loc) {
  return nullptr;
}

void OFIRDialect::printAttribute(Attribute attr, DialectAsmPrinter &p) const {
//  if (succeeded(generatedAttributePrinter(attr, p)))
//    return;
  llvm_unreachable("Unexpected attribute");
}

Attribute OFIRDialect::parseAttribute(DialectAsmParser &p, Type type) const {
  p.emitError(p.getNameLoc(), "Unexpected hw attribute");
  return {};
}


/// Print a type registered to this dialect.
void OFIRDialect::printType(Type type, DialectAsmPrinter &os) const {
  abort();
}

/// Parse a type registered to this dialect.
Type OFIRDialect::parseType(DialectAsmParser &parser) const {
  return Type();
}


// Provide implementations for the enums we use.
//#include "circt/Dialect/OFIR/OFIREnums.cpp.inc"
