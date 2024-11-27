//===- RTGTypes.cpp -------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "circt/Dialect/RTG/IR/RTGTypes.h"
#include "circt/Dialect/RTG/IR/RTGDialect.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/DialectImplementation.h"
#include "llvm/ADT/StringSet.h"
#include "llvm/ADT/TypeSwitch.h"

using namespace circt;
using namespace rtg;

#define GET_TYPEDEF_CLASSES
#include "circt/Dialect/RTG/IR/RTGTypes.cpp.inc"

void circt::rtg::RTGDialect::registerTypes() {
  addTypes<
#define GET_TYPEDEF_LIST
#include "circt/Dialect/RTG/IR/RTGTypes.cpp.inc"
      >();
}
