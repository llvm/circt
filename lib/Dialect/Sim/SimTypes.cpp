//===- SimTypes.cpp -------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "circt/Dialect/Sim/SimTypes.h"
#include "circt/Dialect/Sim/SimDialect.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/DialectImplementation.h"
#include "llvm/ADT/TypeSwitch.h"

using namespace circt;
using namespace sim;
using namespace mlir;

#define GET_TYPEDEF_CLASSES
#include "circt/Dialect/Sim/SimTypes.cpp.inc"

void SimDialect::registerTypes() {
  addTypes<
#define GET_TYPEDEF_LIST
#include "circt/Dialect/Sim/SimTypes.cpp.inc"
      >();
}
