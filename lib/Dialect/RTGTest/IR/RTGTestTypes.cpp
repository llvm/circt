//===- RTGTestTypes.cpp ---------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "circt/Dialect/RTGTest/IR/RTGTestTypes.h"
#include "circt/Dialect/RTGTest/IR/RTGTestDialect.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/DialectImplementation.h"
#include "llvm/ADT/TypeSwitch.h"

using namespace circt;
using namespace rtgtest;

//===----------------------------------------------------------------------===//
// TableGen generated logic.
//===----------------------------------------------------------------------===//

#define GET_TYPEDEF_CLASSES
#include "circt/Dialect/RTGTest/IR/RTGTestTypes.cpp.inc"

//===----------------------------------------------------------------------===//
// Type registration
//===----------------------------------------------------------------------===//

void circt::rtgtest::RTGTestDialect::registerTypes() {
  addTypes<
#define GET_TYPEDEF_LIST
#include "circt/Dialect/RTGTest/IR/RTGTestTypes.cpp.inc"
      >();
}
