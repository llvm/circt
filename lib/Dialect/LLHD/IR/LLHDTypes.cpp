//===- LLHDTypes.cpp - LLHD types and attributes code defs ----------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Implementation for LLHD data types.
//
//===----------------------------------------------------------------------===//

#include "circt/Dialect/LLHD/IR/LLHDTypes.h"
#include "circt/Dialect/LLHD/IR/LLHDDialect.h"
#include "mlir/IR/DialectImplementation.h"
#include "llvm/ADT/TypeSwitch.h"

using namespace circt;
using namespace circt::llhd;

#define GET_TYPEDEF_CLASSES
#include "circt/Dialect/LLHD/IR/LLHDTypes.cpp.inc"

#define GET_ATTRDEF_CLASSES
#include "circt/Dialect/LLHD/IR/LLHDAttributes.cpp.inc"

//===----------------------------------------------------------------------===//
// Helpers
//===----------------------------------------------------------------------===//

/// Parse a nested type, enclosed in angle brackts (`<...>`).
static Type parseNestedType(AsmParser &parser) {
  Type underlyingType;
  if (parser.parseLess())
    return Type();

  llvm::SMLoc loc = parser.getCurrentLocation();
  if (parser.parseType(underlyingType)) {
    parser.emitError(loc, "No signal type found. Signal needs an underlying "
                          "type.");
    return nullptr;
  }

  if (parser.parseGreater())
    return Type();

  return underlyingType;
}

//===----------------------------------------------------------------------===//
// Signal Type
//===----------------------------------------------------------------------===//

/// Parse a signal type.
/// Syntax: sig ::= !llhd.sig<type>
Type SigType::parse(AsmParser &p) {
  auto loc = p.getEncodedSourceLoc(p.getCurrentLocation());
  return getChecked(mlir::detail::getDefaultDiagnosticEmitFn(loc),
                    p.getContext(), parseNestedType(p));
}

void SigType::print(AsmPrinter &p) const {
  p << "<" << getUnderlyingType() << '>';
}

//===----------------------------------------------------------------------===//
// Pointer Type
//===----------------------------------------------------------------------===//

/// Parse a pointer type.
/// Syntax: ptr ::= !llhd.ptr<type>
Type PtrType::parse(AsmParser &p) {
  auto loc = p.getEncodedSourceLoc(p.getCurrentLocation());
  return getChecked(mlir::detail::getDefaultDiagnosticEmitFn(loc),
                    p.getContext(), parseNestedType(p));
}

void PtrType::print(AsmPrinter &p) const {
  p << "<" << getUnderlyingType() << '>';
}

//===----------------------------------------------------------------------===//
// Time Attribute
//===----------------------------------------------------------------------===//

/// Parse a time attribute.
/// Syntax: timeattr ::= #llhd.time<[time][timeUnit], [delta]d, [epsilon]e>
Attribute TimeAttr::parse(AsmParser &p, Type type) {
  llvm::StringRef timeUnit;
  unsigned time = 0;
  unsigned delta = 0;
  unsigned eps = 0;

  // parse the time value
  if (p.parseLess() || p.parseInteger(time) || p.parseKeyword(&timeUnit))
    return {};

  // parse the delta step value
  if (p.parseComma() || p.parseInteger(delta) || p.parseKeyword("d"))
    return {};

  // parse the epsilon value
  if (p.parseComma() || p.parseInteger(eps) || p.parseKeyword("e") ||
      p.parseGreater())
    return {};

  // return a new instance of time attribute
  auto loc = p.getEncodedSourceLoc(p.getCurrentLocation());
  return getChecked(mlir::detail::getDefaultDiagnosticEmitFn(loc),
                    p.getContext(), time, timeUnit, delta, eps);
}

void TimeAttr::print(AsmPrinter &p) const {
  p << "<" << getTime() << getTimeUnit() << ", " << getDelta()
    << "d, " << getEpsilon() << "e>";
}

LogicalResult TimeAttr::verify(function_ref<InFlightDiagnostic()> emitError,
                               TimeType type, unsigned time,
                               llvm::StringRef timeUnit, unsigned delta,
                               unsigned epsilon) {
  // Check the time unit is a legal SI unit
  std::vector<std::string> legalUnits{"ys", "zs", "as", "fs", "ps",
                                      "ns", "us", "ms", "s"};
  if (std::find(legalUnits.begin(), legalUnits.end(), timeUnit) ==
      legalUnits.end())
    return emitError() << "Illegal time unit.";

  return success();
}

//===----------------------------------------------------------------------===//
// Register attributes and types to the LLHD Dialect
//===----------------------------------------------------------------------===//

/// Parses a type registered to this dialect. Parse out the mnemonic then invoke
/// the tblgen'd type parser dispatcher.
Type LLHDDialect::parseType(DialectAsmParser &parser) const {
  llvm::StringRef mnemonic;
  if (parser.parseKeyword(&mnemonic))
    return {};

  Type type;
  if (generatedTypeParser(parser, mnemonic, type).hasValue())
    return type;

  emitError(parser.getEncodedSourceLoc(parser.getCurrentLocation()),
            "Invalid LLHD type!");
  return {};
}

/// Print a type registered to this dialect. Try the tblgen'd type printer
/// dispatcher then fail since all LLHD types are defined via ODS.
void LLHDDialect::printType(Type type, DialectAsmPrinter &printer) const {
  if (succeeded(generatedTypePrinter(type, printer)))
    return;
  llvm_unreachable("unexpected 'llhd' type");
}

/// Parses an attribute registered to this dialect. Parse out the mnemonic then
/// invoke the tblgen'd attribute parser dispatcher.
Attribute LLHDDialect::parseAttribute(DialectAsmParser &parser,
                                      Type type) const {
  llvm::StringRef mnemonic;
  if (parser.parseKeyword(&mnemonic))
    return {};

  Attribute value;
  if (generatedAttributeParser(parser, mnemonic, type, value).hasValue())
    return value;

  emitError(parser.getEncodedSourceLoc(parser.getCurrentLocation()),
            "Invalid LLHD attribute!");
  return {};
}

/// Print an attribute registered to this dialect. Try the tblgen'd attribute
/// printer dispatcher then fail since all LLHD attributes are defined via ODS.
void LLHDDialect::printAttribute(Attribute attr,
                                 DialectAsmPrinter &printer) const {
  if (succeeded(generatedAttributePrinter(attr, printer)))
    return;
  llvm_unreachable("unexpected 'llhd' type");
}

void LLHDDialect::registerTypes() {
  addTypes<
#define GET_TYPEDEF_LIST
#include "circt/Dialect/LLHD/IR/LLHDTypes.cpp.inc"
      >();
}

void LLHDDialect::registerAttributes() {
  addAttributes<
#define GET_ATTRDEF_LIST
#include "circt/Dialect/LLHD/IR/LLHDAttributes.cpp.inc"
      >();
}
