//===- RTGAttributes.cpp --------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "circt/Dialect/RTG/IR/RTGAttributes.h"
#include "circt/Dialect/RTG/IR/RTGDialect.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/DialectImplementation.h"
#include "llvm/ADT/TypeSwitch.h"

using namespace circt;
using namespace rtg;

//===----------------------------------------------------------------------===//
// ImmediateAttr
//===----------------------------------------------------------------------===//

Type ImmediateAttr::getType() const {
  return ImmediateType::get(getContext(), getValue().getBitWidth());
}

Attribute ImmediateAttr::parse(AsmParser &odsParser, Type odsType) {
  llvm::SMLoc loc = odsParser.getCurrentLocation();

  APInt val;
  uint32_t width; // NOTE: this integer type should match the 'width' parameter
                  // type in immediate type.
  if (odsParser.parseLess() || odsParser.parseInteger(width) ||
      odsParser.parseComma() || odsParser.parseInteger(val) ||
      odsParser.parseGreater())
    return {};

  // If the attribute type is explicitly given, check that the bit-widths match.
  if (auto immTy = llvm::dyn_cast_or_null<ImmediateType>(odsType)) {
    if (immTy.getWidth() != width) {
      odsParser.emitError(loc) << "explicit immediate type bit-width does not "
                                  "match attribute bit-width, "
                               << immTy.getWidth() << " vs " << width;
      return {};
    }
  }

  if (width > val.getBitWidth()) {
    // sext is always safe here, even for unsigned values, because the
    // parseOptionalInteger method will return something with a zero in the
    // top bits if it is a positive number.
    val = val.sext(width);
  } else if (width < val.getBitWidth()) {
    // The parser can return an unnecessarily wide result.
    // This isn't a problem, but truncating off bits is bad.
    unsigned neededBits =
        val.isNegative() ? val.getSignificantBits() : val.getActiveBits();
    if (width < neededBits) {
      odsParser.emitError(loc)
          << "integer value out-of-range for bit-width " << width;
      return {};
    }
    val = val.trunc(width);
  }

  return ImmediateAttr::get(odsParser.getContext(), val);
}

void ImmediateAttr::print(AsmPrinter &odsPrinter) const {
  odsPrinter << "<" << getValue().getBitWidth() << ", " << getValue() << ">";
}

//===----------------------------------------------------------------------===//
// TableGen generated logic.
//===----------------------------------------------------------------------===//

void RTGDialect::registerAttributes() {
  addAttributes<
#define GET_ATTRDEF_LIST
#include "circt/Dialect/RTG/IR/RTGAttributes.cpp.inc"
      >();
}

#define GET_ATTRDEF_CLASSES
#include "circt/Dialect/RTG/IR/RTGAttributes.cpp.inc"
