//===- RTGTestAttributes.cpp ----------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "circt/Dialect/RTGTest/IR/RTGTestAttributes.h"
#include "circt/Dialect/RTGTest/IR/RTGTestDialect.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/DialectImplementation.h"
#include "llvm/ADT/TypeSwitch.h"

using namespace circt;
using namespace rtgtest;

//===----------------------------------------------------------------------===//
// CPUAttr
//===----------------------------------------------------------------------===//

Type CPUAttr::getType() const { return rtgtest::CPUType::get(getContext()); }

//===----------------------------------------------------------------------===//
// TableGen generated logic.
//===----------------------------------------------------------------------===//

void RTGTestDialect::registerAttributes() {
  addAttributes<
#define GET_ATTRDEF_LIST
#include "circt/Dialect/RTGTest/IR/RTGTestAttributes.cpp.inc"
      >();
}

#define GET_ATTRDEF_CLASSES
#include "circt/Dialect/RTGTest/IR/RTGTestAttributes.cpp.inc"
