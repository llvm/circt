//===- OMTypes.cpp - Object Model type definitions ------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file contains the Object Model type definitions.
//
//===----------------------------------------------------------------------===//

#include "circt/Dialect/OM/OMTypes.h"
#include "circt/Dialect/OM/OMDialect.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/DialectImplementation.h"
#include "llvm/ADT/TypeSwitch.h"

using namespace circt;
using namespace mlir;
using namespace circt::om;
using namespace circt::om::detail;

#define GET_TYPEDEF_CLASSES
#include "circt/Dialect/OM/OMTypes.cpp.inc"

void circt::om::OMDialect::registerTypes() {
  addTypes<
#define GET_TYPEDEF_LIST
#include "circt/Dialect/OM/OMTypes.cpp.inc"
      >();
}

mlir::LogicalResult
circt::om::MapType::verify(llvm::function_ref<mlir::InFlightDiagnostic()> diag,
                           mlir::Type keyType, mlir::Type elementType) {
  if (!llvm::isa<om::StringType, mlir::IntegerType>(keyType))
    return diag() << "map key type must be either string or integer but got "
                  << keyType;
  return mlir::success();
}

bool circt::om::isMapKeyValuePairType(mlir::Type type) {
  auto tuple = llvm::dyn_cast<mlir::TupleType>(type);
  return tuple && tuple.getTypes().size() == 2 &&
         llvm::isa<om::StringType, mlir::IntegerType>(tuple.getTypes().front());
}

namespace circt {
namespace om {
namespace detail {
bool operator==(const EnumElement &a, const EnumElement &b) {
  return a.name == b.name && a.type == b.type;
}

llvm::hash_code
hash_value(const EnumElement &fi) { // NOLINT(readability-identifier-naming)
  return llvm::hash_combine(fi.name, fi.type);
}
} // namespace detail
} // namespace om
} // namespace circt

/// Parse a list of field names and types within <>. E.g.:
/// <foo: i7, bar: i8>
static mlir::ParseResult
parseEnumFields(AsmParser &p,
                SmallVectorImpl<EnumType::EnumElement> &parameters) {
  return p.parseCommaSeparatedList(
      mlir::AsmParser::Delimiter::LessGreater, [&]() -> ParseResult {
        StringRef name;
        Type type;
        if (p.parseKeyword(&name) || p.parseColon() || p.parseType(type))
          return failure();
        parameters.push_back(
            EnumType::EnumElement{StringAttr::get(p.getContext(), name), type});
        return success();
      });
}

/// Print out a list of named fields surrounded by <>.
static void printEnumFields(AsmPrinter &p,
                            ArrayRef<EnumType::EnumElement> fields) {
  p << '<';
  llvm::interleaveComma(fields, p, [&](const EnumType::EnumElement &field) {
    p << field.name.getValue() << ": " << field.type;
  });
  p << '>';
}

mlir::Type circt::om::EnumType::parse(AsmParser &p) {
  llvm::SmallVector<EnumType::EnumElement, 4> parameters;
  if (parseEnumFields(p, parameters))
    return Type();
  return get(p.getContext(), parameters);
}

void circt::om::EnumType::print(AsmPrinter &p) const {
  printEnumFields(p, getElements());
}
