//===- MooreAttributes.cpp - Implement the Moore attributes ---------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements the Moore dialect attributes.
//
//===----------------------------------------------------------------------===//

#include "circt/Dialect/Moore/MooreAttributes.h"
#include "circt/Dialect/Moore/MooreDialect.h"
#include "circt/Dialect/Moore/MooreTypes.h"
#include "mlir/IR/DialectImplementation.h"
#include "llvm/ADT/TypeSwitch.h"

using namespace circt;
using namespace circt::moore;
using mlir::AsmParser;
using mlir::AsmPrinter;

//===----------------------------------------------------------------------===//
// FVIntegerAttr
//===----------------------------------------------------------------------===//

Attribute FVIntegerAttr::parse(AsmParser &p, Type) {
  // Parse the value and width specifier.
  FVInt value;
  unsigned width;
  llvm::SMLoc widthLoc;
  if (p.parseLess() || parseFVInt(p, value) || p.parseColon() ||
      p.getCurrentLocation(&widthLoc) || p.parseInteger(width) ||
      p.parseGreater())
    return {};

  // Make sure the integer fits into the requested number of bits.
  unsigned neededBits =
      value.isNegative() ? value.getSignificantBits() : value.getActiveBits();
  if (width < neededBits) {
    p.emitError(widthLoc) << "integer literal requires at least " << neededBits
                          << " bits, but attribute specifies only " << width;
    return {};
  }

  return FVIntegerAttr::get(p.getContext(), value.sextOrTrunc(width));
}

void FVIntegerAttr::print(AsmPrinter &p) const {
  p << "<";
  printFVInt(p, getValue());
  p << " : " << getValue().getBitWidth() << ">";
}

//===----------------------------------------------------------------------===//
// Generated logic
//===----------------------------------------------------------------------===//

#define GET_ATTRDEF_CLASSES
#include "circt/Dialect/Moore/MooreAttributes.cpp.inc"

void MooreDialect::registerAttributes() {
  addAttributes<
#define GET_ATTRDEF_LIST
#include "circt/Dialect/Moore/MooreAttributes.cpp.inc"
      >();
}
