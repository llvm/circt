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

Type ChannelPort::parse(MLIRContext *ctxt, DialectAsmParser &p) {
  Type inner;
  if (p.parseLess() || p.parseType(inner) || p.parseGreater())
    return Type();
  return get(ctxt, inner);
}

void ChannelPort::print(DialectAsmPrinter &p) const {
  p << "channel<";
  p.printType(getInner());
  p << ">";
}

Type StructType::parse(MLIRContext *ctxt, DialectAsmParser &p) {
  if (p.parseLess())
    return Type();
  StringRef structName;
  if (p.parseKeyword(&structName) || p.parseComma())
    return Type();

  llvm::SmallVector<hw::StructType::FieldInfo, 4> fields;
  StringRef fieldName;
  while (mlir::succeeded(p.parseOptionalKeyword(&fieldName))) {
    Type type;
    if (p.parseColon() || p.parseType(type))
      return Type();
    fields.push_back(hw::StructType::FieldInfo{fieldName, type});
    if (p.parseOptionalComma())
      break;
  }
  if (p.parseGreater())
    return Type();

  auto inner = hw::StructType::get(ctxt, fields);
  return get(ctxt, structName, inner);
}

void StructType::print(DialectAsmPrinter &p) const {
  p << "struct<";
  p << getName() << ", ";
  llvm::interleaveComma(getInner().getElements(), p,
                        [&](const hw::StructType::FieldInfo &field) {
                          p << field.name << ": " << field.type;
                        });
  p << ">";
}

#define GET_TYPEDEF_CLASSES
#include "circt/Dialect/ESI/ESITypes.cpp.inc"

/// Parses a type registered to this dialect
Type ESIDialect::parseType(DialectAsmParser &parser) const {
  llvm::StringRef mnemonic;
  if (parser.parseKeyword(&mnemonic))
    return Type();
  Type genType;
  auto parseResult =
      generatedTypeParser(getContext(), parser, mnemonic, genType);
  if (parseResult.hasValue())
    return genType;
  parser.emitError(parser.getCurrentLocation(), "Could not parse esi.")
      << mnemonic << "!\n";
  return Type();
}

/// Print a type registered to this dialect
void ESIDialect::printType(Type type, DialectAsmPrinter &printer) const {
  if (succeeded(generatedTypePrinter(type, printer)))
    return;
  llvm_unreachable("unexpected 'esi' type kind");
}

void ESIDialect::registerTypes() {
  addTypes<
#define GET_TYPEDEF_LIST
#include "circt/Dialect/ESI/ESITypes.cpp.inc"
      >();
}
