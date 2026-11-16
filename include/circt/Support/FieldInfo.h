//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines FieldInfo, a name and type pair describing the fields of
// aggregate types such as `!hw.struct` and `!sim.variant`, along with helpers
// to parse and print lists of fields.
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_SUPPORT_FIELDINFO_H
#define CIRCT_SUPPORT_FIELDINFO_H

#include "circt/Support/LLVM.h"
#include "mlir/IR/AttrTypeSubElements.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/Types.h"
#include "llvm/ADT/Hashing.h"

namespace circt {

/// A named and typed field of an aggregate type.
struct FieldInfo {
  mlir::StringAttr name;
  mlir::Type type;
};

inline bool operator==(const FieldInfo &a, const FieldInfo &b) {
  return a.name == b.name && a.type == b.type;
}
inline llvm::hash_code hash_value(const FieldInfo &fi) {
  return llvm::hash_combine(fi.name, fi.type);
}

/// Parse a list of named fields within `<>`, e.g. `<foo: i7, bar: i8>`. Field
/// names may be bare keywords or quoted strings. The location and name of every
/// field whose name was already used by a preceding field is appended to
/// `duplicates`, leaving it to the caller to report them.
ParseResult
parseFieldList(AsmParser &p, SmallVectorImpl<FieldInfo> &fields,
               SmallVectorImpl<std::pair<llvm::SMLoc, StringAttr>> &duplicates);

/// Print a list of fields within `<>`, e.g. `<foo: i7, bar: i8>`.
void printFieldList(AsmPrinter &p, ArrayRef<FieldInfo> fields);

} // namespace circt

namespace mlir {
/// Expose the names and types of fields to the generic attribute and type
/// walking and replacement infrastructure. This allows walkers to recurse into
/// the fields of aggregate types, and replacers such as
/// `mlir::AttrTypeReplacer` to replace types nested within the fields.
template <>
struct AttrTypeSubElementHandler<circt::FieldInfo> {
  static void walk(const circt::FieldInfo &param,
                   AttrTypeImmediateSubElementWalker &walker) {
    walker.walk(param.name);
    walker.walk(param.type);
  }
  static circt::FieldInfo replace(const circt::FieldInfo &param,
                                  AttrSubElementReplacements &attrRepls,
                                  TypeSubElementReplacements &typeRepls) {
    return {cast<StringAttr>(attrRepls.take_front(1)[0]),
            typeRepls.take_front(1)[0]};
  }
};
} // namespace mlir

#endif // CIRCT_SUPPORT_FIELDINFO_H
