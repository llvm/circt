//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "circt/Dialect/Probe/ProbeTypes.h"
#include "circt/Dialect/Probe/ProbeDialect.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/DialectImplementation.h"

using namespace circt;
using namespace probe;
using namespace mlir;

#define GET_TYPEDEF_CLASSES
#include "circt/Dialect/Probe/ProbeTypes.cpp.inc"

void ProbeDialect::registerTypes() {
  addTypes<
#define GET_TYPEDEF_LIST
#include "circt/Dialect/Probe/ProbeTypes.cpp.inc"
      >();
}
