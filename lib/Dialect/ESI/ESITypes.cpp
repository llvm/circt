//===- ESITypes.cpp - ESI types code defs -----------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Definitions for esi data types. Anything which doesn't have to be public
// should go in here.
//
//===----------------------------------------------------------------------===//

#include "circt/Dialect/ESI/ESITypes.h"
#include "circt/Dialect/HW/HWTypes.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/DialectImplementation.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/TypeSwitch.h"

using namespace circt;
using namespace circt::esi;

Type ChannelPort::parse(AsmParser &p) {
  Type inner;
  if (p.parseLess() || p.parseType(inner) || p.parseGreater())
    return Type();
  return get(p.getContext(), inner);
}

void ChannelPort::print(AsmPrinter &p) const {
  p << "<";
  p.printType(getInner());
  p << ">";
}

#define GET_TYPEDEF_CLASSES
#include "circt/Dialect/ESI/ESITypes.cpp.inc"

void ESIDialect::registerTypes() {
  addTypes<
#define GET_TYPEDEF_LIST
#include "circt/Dialect/ESI/ESITypes.cpp.inc"
      >();
}
