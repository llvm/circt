//===- Types.cpp - RTL types code defs --------------------------*- C++ -*-===//
//
// Definitions for RTL data types. Anything which doesn't have to be public
// should go in here.
//
//===----------------------------------------------------------------------===//

#include "circt/Dialect/RTL/Types.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/DialectImplementation.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/Hashing.h"
#include "llvm/ADT/TypeSwitch.h"

using namespace mlir;
using namespace circt::rtl;

namespace circt {
namespace rtl {

/// Equality operator for FieldInfo. Required for data types in TypeStorage
static bool operator==(const FieldInfo &a, const FieldInfo &b) {
  return a.name == b.name && a.type == b.type;
}

/// Compute a hash for FieldInfo. Required for data types in TypeStorage
static llvm::hash_code hash_value(const FieldInfo &fi) { // NOLINT
  return llvm::hash_combine(fi.name, fi.type);
}

} // namespace rtl
} // namespace circt

#define GET_TYPEDEF_CLASSES
#include "circt/Dialect/RTL/RTLTypes.cpp.inc"

Type EnumType::parse(MLIRContext *ctxt, DialectAsmParser &parser) {
  SmallVector<StringRef, 16> enumFields;
  if (parser.parseLess())
    return Type();
  StringRef entry;
  while (!parser.parseKeyword(&entry)) {
    enumFields.push_back(entry);
    if (parser.parseOptionalComma())
      break;
  }
  if (parser.parseGreater())
    return Type();
  return get(ctxt, enumFields);
}

void EnumType::print(::mlir::DialectAsmPrinter &printer) const {
  ArrayRef<StringRef> fields = getImpl()->fields;
  printer << "enum"
          << "<";
  for (auto it = fields.begin(), e = fields.end(); it != e; ++it) {
    printer << *it;
    if (it < e - 1)
      printer << ",";
  }
  printer << ">";
}
