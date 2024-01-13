//===- SMTAttributes.cpp - Implement SMT attributes -----------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "circt/Dialect/SMT/SMTAttributes.h"
#include "circt/Dialect/SMT/SMTDialect.h"
#include "circt/Dialect/SMT/SMTTypes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/DialectImplementation.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/Format.h"

using namespace circt;
using namespace circt::smt;

//===----------------------------------------------------------------------===//
// BitVectorAttr
//===----------------------------------------------------------------------===//

LogicalResult BitVectorAttr::verify(
    function_ref<InFlightDiagnostic()> emitError,
    APInt value) { // NOLINT(performance-unnecessary-value-param)
  if (value.getBitWidth() < 1)
    return emitError() << "bit-width must be at least 1, but got "
                       << value.getBitWidth();
  return success();
}

std::string BitVectorAttr::getValueAsString(bool prefix) const {
  unsigned width = getValue().getBitWidth();
  SmallVector<char> toPrint;
  StringRef pref = prefix ? "#" : "";
  if (width % 4 == 0) {
    getValue().toString(toPrint, 16, false, false, false);
    // APInt's 'toString' omits leading zeros. However, those are critical here
    // because they determine the bit-width of the bit-vector.
    SmallVector<char> leadingZeros(width / 4 - toPrint.size(), '0');
    return (pref + "x" + Twine(leadingZeros) + toPrint).str();
  }

  getValue().toString(toPrint, 2, false, false, false);
  // APInt's 'toString' omits leading zeros
  SmallVector<char> leadingZeros(width - toPrint.size(), '0');
  return (pref + "b" + Twine(leadingZeros) + toPrint).str();
}

/// Parse an SMT-LIB formatted bit-vector string.
static FailureOr<APInt>
parseBitVectorString(function_ref<InFlightDiagnostic()> emitError,
                     StringRef value) {
  if (value[0] != '#')
    return emitError() << "expected '#'";

  if (value.size() < 3)
    return emitError() << "expected at least one digit";

  if (value[1] == 'b')
    return APInt(value.size() - 2, std::string(value.begin() + 2, value.end()),
                 2);

  if (value[1] == 'x')
    return APInt((value.size() - 2) * 4,
                 std::string(value.begin() + 2, value.end()), 16);

  return emitError() << "expected either 'b' or 'x'";
}

BitVectorAttr BitVectorAttr::get(MLIRContext *context, StringRef value) {
  InFlightDiagnostic diag;
  auto maybeValue =
      parseBitVectorString([&]() { return std::move(diag); }, value);
  diag.abandon();

  assert(succeeded(maybeValue) && "string must have SMT-LIB format");
  return Base::get(context, *maybeValue);
}

BitVectorAttr
BitVectorAttr::getChecked(function_ref<InFlightDiagnostic()> emitError,
                          MLIRContext *context, StringRef value) {
  auto maybeValue = parseBitVectorString(emitError, value);
  if (failed(maybeValue))
    return {};

  return Base::getChecked(emitError, context, *maybeValue);
}

BitVectorAttr BitVectorAttr::get(MLIRContext *context, unsigned value,
                                 unsigned width) {
  return Base::get(context, APInt(width, value));
}

BitVectorAttr
BitVectorAttr::getChecked(function_ref<InFlightDiagnostic()> emitError,
                          MLIRContext *context, unsigned value,
                          unsigned width) {
  return Base::getChecked(emitError, context, APInt(width, value));
}

Attribute BitVectorAttr::parse(AsmParser &odsParser, Type odsType) {
  llvm::SMLoc loc = odsParser.getCurrentLocation();

  std::string val;
  if (odsParser.parseLess() || odsParser.parseString(&val))
    return {};

  auto maybeVal =
      parseBitVectorString([&]() { return odsParser.emitError(loc); }, val);
  if (failed(maybeVal))
    return {};

  if (odsParser.parseGreater())
    return {};

  // We implement the TypedAttr interface, i.e., the AsmParser allows an
  // optional explicit type as suffix which is provided here as 'odsType'. We
  // can always build the type from the constant itself, therefore, we can
  // ignore this explicit type. However, to ensure consistency we should verify
  // that the correct type is provided.
  auto bv = BitVectorAttr::get(odsParser.getContext(), *maybeVal);
  if (odsType && bv.getType() != odsType) {
    odsParser.emitError(loc, "expected type for constant does not match "
                             "explicitly provided attribute type, got ")
        << odsType << ", expected " << bv.getType();
    return {};
  }

  return bv;
}

void BitVectorAttr::print(AsmPrinter &odsPrinter) const {
  odsPrinter << "<\"" << getValueAsString() << "\">";
}

Type BitVectorAttr::getType() const {
  return BitVectorType::get(getContext(), getValue().getBitWidth());
}

//===----------------------------------------------------------------------===//
// ODS Boilerplate
//===----------------------------------------------------------------------===//

#define GET_ATTRDEF_CLASSES
#include "circt/Dialect/SMT/SMTAttributes.cpp.inc"

void SMTDialect::registerAttributes() {
  addAttributes<
#define GET_ATTRDEF_LIST
#include "circt/Dialect/SMT/SMTAttributes.cpp.inc"
      >();
}
