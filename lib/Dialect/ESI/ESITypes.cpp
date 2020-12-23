//===- ESITypes.cpp - ESI types code defs -----------------------*- C++ -*-===//
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

static uint64_t getCapnpTypeIDImpl(Type t) {
  auto chanPort = t.dyn_cast<ChannelPort>();
  if (chanPort) {
    uint64_t innerHash = getCapnpTypeIDImpl(chanPort.getInner());
    return llvm::hash_combine(chanPort.getTypeID(), innerHash);
  }

  // This is temporary until I figure a way to access a deterministic hash.
  // TODO: replace me!
  IntegerType i = t.dyn_cast<IntegerType>();
  if (i)
    return llvm::hash_combine(i.getWidth(), i.getSignedness());
  return 0;
}

uint64_t circt::esi::getCapnpTypeID(Type t) {
  return llvm::hash_combine(esiCosimSchemaVersion, getCapnpTypeIDImpl(t));
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
