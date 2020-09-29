//===- ESITypes.cpp - ESI types code defs -----------------------*- C++ -*-===//
//
//===----------------------------------------------------------------------===//

#include "circt/Dialect/ESI/ESITypes.hpp"
#include "circt/AutoGenParser.h"
#include <llvm/ADT/ArrayRef.h>
#include <llvm/ADT/Hashing.h>
#include <llvm/ADT/TypeSwitch.h>
#include <mlir/IR/Attributes.h>
#include <mlir/IR/DialectImplementation.h>

using namespace mlir;

// Parser for StringEncoding
// template <>
// struct Parse<circt::esi::StringEncoding> {
//   mlir::ParseResult go(MLIRContext *ctxt, DialectAsmParser &parser,
//                        StringRef parameterName,
//                        circt::esi::StringEncoding &result) {
//     StringRef encStr;
//     if (parser.parseKeyword(&encStr))
//       return mlir::failure();
//     if (encStr.compare_lower("ascii") == 0)
//       result = circt::esi::StringEncoding::ASCII;
//     else if (encStr.compare_lower("utf8") == 0)
//       result = circt::esi::StringEncoding::UTF8;
//     else if (encStr.compare_lower("utf16") == 0)
//       result = circt::esi::StringEncoding::UTF16;
//     else if (encStr.compare_lower("utf32") == 0)
//       result = circt::esi::StringEncoding::UTF32;
//     else {
//       llvm::errs() << "Not a valid string encoding: " << encStr << ".\n";
//       return mlir::failure();
//     }
//     return mlir::success();
//   }
// };

// // Printer for StringEncoding
// template <>
// struct Print<circt::esi::StringEncoding> {
//   static void go(DialectAsmPrinter &printer,
//                  const circt::esi::StringEncoding &enc) {
//     switch (enc) {
//     case circt::esi::StringEncoding::UTF8:
//       printer << "utf8";
//       break;
//     case circt::esi::StringEncoding::UTF16:
//       printer << "utf16";
//       break;
//     case circt::esi::StringEncoding::UTF32:
//       printer << "utf32";
//       break;
//     case circt::esi::StringEncoding::ASCII:
//       printer << "ascii";
//       break;
//     }
//   }
// };

namespace circt {
namespace esi {

/// Equality operator for FieldInfo. Required for data types in TypeStorage
static bool operator==(const FieldInfo &a, const FieldInfo &b) {
  return a.name == b.name && a.type == b.type;
}

/// Compute a hash for FieldInfo. Required for data types in TypeStorage
static llvm::hash_code hash_value(const FieldInfo &fi) {
  return llvm::hash_combine(fi.name, fi.type);
}

} // namespace esi
} // namespace circt

#define GET_TYPEDEF_CLASSES
#include "circt/Dialect/ESI/ESITypes.cpp.inc"
