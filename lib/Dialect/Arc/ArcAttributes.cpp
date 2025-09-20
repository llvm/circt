//===- ArcAttributes.cpp - Implement Arc attributes -----------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "circt/Dialect/Arc/ArcAttributes.h"
#include "circt/Dialect/Arc/ArcDialect.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/DialectImplementation.h"
#include "llvm/ADT/TypeSwitch.h"

using namespace circt;
using namespace arc;

#define GET_ATTRDEF_CLASSES
#include "circt/Dialect/Arc/ArcAttributes.cpp.inc"

void ArcDialect::registerAttributes() {
  addAttributes<
#define GET_ATTRDEF_LIST
#include "circt/Dialect/Arc/ArcAttributes.cpp.inc"
      >();
}
