//===- ArcTypes.cpp -------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "circt/Dialect/Arc/ArcTypes.h"
#include "circt/Dialect/Arc/ArcDialect.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/DialectImplementation.h"
#include "llvm/ADT/TypeSwitch.h"

using namespace circt;
using namespace arc;
using namespace mlir;

#define GET_TYPEDEF_CLASSES
#include "circt/Dialect/Arc/ArcTypes.cpp.inc"

void ArcDialect::registerTypes() {
  addTypes<
#define GET_TYPEDEF_LIST
#include "circt/Dialect/Arc/ArcTypes.cpp.inc"
      >();
}
