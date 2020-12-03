//===- Dialect.cpp - Implement the SV dialect -----------------------------===//
//
//===----------------------------------------------------------------------===//

#include "circt/Dialect/SV/Dialect.h"
#include "circt/Dialect/SV/Ops.h"
#include "circt/Dialect/SV/Types.h"

#include "mlir/IR/DialectImplementation.h"

using namespace circt::sv;
using namespace mlir;

//===----------------------------------------------------------------------===//
// Dialect specification.
//===----------------------------------------------------------------------===//

SVDialect::SVDialect(MLIRContext *context)
    : Dialect(getDialectNamespace(), context,
    ::mlir::TypeID::get<SVDialect>()) {

  // Register types.
  addTypes<
#define GET_TYPEDEF_LIST
#include "circt/Dialect/SV/SVTypes.cpp.inc"
      >();

  // Register operations.
  addOperations<
#define GET_OP_LIST
#include "circt/Dialect/SV/SV.cpp.inc"
      >();
}

SVDialect::~SVDialect() {}

/// Parses a type registered to this dialect
Type SVDialect::parseType(DialectAsmParser &parser) const {
  llvm::StringRef mnemonic;
  auto loc = parser.getCurrentLocation();
  if (parser.parseKeyword(&mnemonic))
    return Type();
  if (auto type = generatedTypeParser(getContext(), parser, mnemonic))
    return type;
  parser.emitError(loc, "Failed to parse type sv.") << mnemonic << "\n";
  return Type();
}

/// Print a type registered to this dialect
void SVDialect::printType(Type type, DialectAsmPrinter &printer) const {
  if (succeeded(generatedTypePrinter(type, printer)))
    return;
  llvm_unreachable("unexpected 'rtl' type kind");
}
