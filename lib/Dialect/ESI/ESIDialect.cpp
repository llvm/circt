//===- ESIDialect.cpp - ESI dialect code defs -------------------*- C++ -*-===//
//
// Dialect definitions. Should be relatively standard boilerplate.
//
//===----------------------------------------------------------------------===//

#include "circt/Dialect/ESI/ESIDialect.h"
#include "circt/Dialect/ESI/ESIOps.h"
#include "circt/Dialect/ESI/ESITypes.h"

#include "mlir/IR/DialectImplementation.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/FormatVariadic.h"

using namespace mlir;

namespace circt {
namespace esi {

ESIDialect::ESIDialect(MLIRContext *context)
    : Dialect("esi", context, TypeID::get<ESIDialect>()) {
  addTypes<
#define GET_TYPEDEF_LIST
#include "circt/Dialect/ESI/ESITypes.cpp.inc"
      >();

  addOperations<
#define GET_OP_LIST
#include "circt/Dialect/ESI/ESI.cpp.inc"
      >();
}

/// Parses a type registered to this dialect
Type ESIDialect::parseType(DialectAsmParser &parser) const {
  llvm::StringRef mnemonic;
  if (parser.parseKeyword(&mnemonic))
    return Type();
  auto genType = generatedTypeParser(getContext(), parser, mnemonic);
  if (genType != Type())
    return genType;
  parser.emitError(parser.getCurrentLocation(),
                   llvm::formatv("Could not parse esi.{0}!\n", mnemonic));
  return Type();
}

/// Print a type registered to this dialect
void ESIDialect::printType(Type type, DialectAsmPrinter &printer) const {
  if (succeeded(generatedTypePrinter(type, printer)))
    return;
  llvm_unreachable("unexpected 'esi' type kind");
}

} // namespace esi
} // namespace circt

#include "circt/Dialect/ESI/ESIAttrs.cpp.inc"
