//===- CHIRRTLDialect.cpp - Implement the CHIRRTL dialect -----------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements the CHIRRTL dialect.
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

//===----------------------------------------------------------------------===//
// CHIRRTL Dialect
//===----------------------------------------------------------------------===//

// This is used to give custom SSA names which match the "name" attribute of the
// memory operation, which allows us to elide the name attribute.
namespace {
struct CHIRRTLOpAsmDialectInterface : public OpAsmDialectInterface {
  using OpAsmDialectInterface::OpAsmDialectInterface;
  void getAsmResultNames(Operation *op, OpAsmSetValueNameFn setNameFn) const {
    // Many CHIRRTL dialect operations have an optional 'name' attribute.  If
    // present, use it.
    if (op->getNumResults() == 1)
      if (auto nameAttr = op->getAttrOfType<StringAttr>("name"))
        setNameFn(op->getResult(0), nameAttr.getValue());
  }
};
} // namespace

void CHIRRTLDialect::initialize() {
  // Register operations.
  addOperations<
#define GET_OP_LIST
#include "circt/Dialect/FIRRTL/CHIRRTL.cpp.inc"
      >();

  // Register types.
  registerTypes();

  // Register interface implementations.
  addInterfaces<CHIRRTLOpAsmDialectInterface>();
}

#include "circt/Dialect/FIRRTL/CHIRRTLDialect.cpp.inc"
