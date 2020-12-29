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
#include "mlir/IR/Attributes.h"
#include "mlir/IR/DialectImplementation.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/Hashing.h"
#include "llvm/ADT/TypeSwitch.h"

using namespace mlir;
using namespace circt::esi;

// We compute a deterministic hash based on the type. Since llvm::hash_value
// changes from execution to execution, we don't use it. This assumes a closed
// type system, which is reasonable since we only support some types in the
// Capnp schema generation anyway.
uint64_t circt::esi::getCapnpTypeID(Type t) {
  // We can hash up to 64 bytes with a single function call.
  char buffer[64];
  memset(buffer, 0, sizeof(buffer));

  // The first byte is for the outer type.
  auto chanPort = t.dyn_cast<ChannelPort>();
  assert(chanPort && "Type not supported as top level");
  buffer[0] = 1; // Constant for the ChannelPort type.

  TypeSwitch<Type>(chanPort.getInner())
      .Case([&buffer](IntegerType t) {
        // The second byte is for the inner type.
        buffer[1] = 1;
        // The rest can be defined arbitrarily.
        buffer[2] = (char)t.getSignedness();
        *(int64_t *)&buffer[4] = t.getWidth();
      })
      .Default([](Type) { assert(false && "Type not yet supported"); });

  return llvm::hashing::detail::hash_short(buffer, 12, esiCosimSchemaVersion);
}

Type ChannelPort::parse(mlir::MLIRContext *ctxt, mlir::DialectAsmParser &p) {
  Type inner;
  if (p.parseLess() || p.parseType(inner) || p.parseGreater())
    return Type();
  return get(ctxt, inner);
}

void ChannelPort::print(mlir::DialectAsmPrinter &p) const {
  p << "channel<";
  p.printType(getInner());
  p << ">";
}

#define GET_TYPEDEF_CLASSES
#include "circt/Dialect/ESI/ESITypes.cpp.inc"

/// Parses a type registered to this dialect
Type ESIDialect::parseType(DialectAsmParser &parser) const {
  llvm::StringRef mnemonic;
  if (parser.parseKeyword(&mnemonic))
    return Type();
  auto genType = generatedTypeParser(getContext(), parser, mnemonic);
  if (genType != Type())
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
