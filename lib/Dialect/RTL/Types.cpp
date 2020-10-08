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

/// The parser for both the struct and union types.
template <typename FT>
static Type parseFieldInfoType(MLIRContext *ctxt, DialectAsmParser &parser) {
  SmallVector<FieldInfo, 4> parameters;
  if (parser.parseLess())
    return ::mlir::Type();
  while (mlir::succeeded(parser.parseOptionalLBrace())) {
    StringRef name;
    if (parser.parseKeyword(&name))
      return ::mlir::Type();
    if (parser.parseComma())
      return ::mlir::Type();
    ::mlir::Type type;
    if (parser.parseType(type))
      return ::mlir::Type();
    if (parser.parseRBrace())
      return ::mlir::Type();
    parameters.push_back(FieldInfo(name, type));
    if (parser.parseOptionalComma())
      break;
  }
  if (parser.parseGreater())
    return ::mlir::Type();
  return FT::get(ctxt, parameters);
}

/// The printer for both the struct and union types.
template <typename FT>
static void printFieldInfoType(DialectAsmPrinter &printer,
                               const FT &typeInstance) {
  printer << FT::getMnemonic() << "<";

  for (size_t i = 0, e = typeInstance.getFields().size(); i < e; i++) {
    const auto &field = typeInstance.getFields()[i];
    printer << "{" << field.name << "," << field.type << "}";
    if (i < e - 1)
      printer << ",";
  }
  printer << ">";
}

#define GET_TYPEDEF_CLASSES
#include "circt/Dialect/RTL/RTLTypes.cpp.inc"

/// The parser for the enum type.
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
