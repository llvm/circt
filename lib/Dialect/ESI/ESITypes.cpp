//===- ESITypes.cpp - ESI types code defs -----------------------*- C++ -*-===//
//
// Definitions for esi data types. Anything which doesn't have to be public
// should go in here.
//
//===----------------------------------------------------------------------===//

#include "circt/Dialect/ESI/ESITypes.h"
#include "circt/AutoGenParser.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/Hashing.h"
#include "llvm/ADT/TypeSwitch.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/DialectImplementation.h"

using namespace mlir;

namespace circt {

namespace autogen {

// Parser for StringEncoding. Is a specialization of the standard type parsers
// in AutoGenParser.h.
template <>
struct Parse<circt::esi::StringEncoding> {
  mlir::ParseResult go(MLIRContext *ctxt, DialectAsmParser &parser,
                       StringRef parameterName,
                       circt::esi::StringEncoding &result) {
    StringRef encStr;
    if (parser.parseKeyword(&encStr))
      return mlir::failure();
    auto maybeEnc =
        circt::esi::symbolizeEnum<circt::esi::StringEncoding>(encStr.upper());
    if (!maybeEnc.hasValue())
      return mlir::failure();
    result = maybeEnc.getValue();
    return mlir::success();
  }
};

// Printer for StringEncoding. Is a specialization of the standard type printers
// in AutoGenParser.h.
template <>
struct Print<circt::esi::StringEncoding> {
  static void go(DialectAsmPrinter &printer,
                 const circt::esi::StringEncoding &enc) {
    printer << circt::esi::stringifyEnum(enc).lower();
  }
};

} // namespace autogen
} // namespace circt

#define GET_TYPEDEF_CLASSES
#include "circt/Dialect/ESI/ESITypes.cpp.inc"
