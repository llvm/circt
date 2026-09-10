//===- BuilderUtils.h - Operation builder utilities -------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_SUPPORT_BUILDERUTILS_H
#define CIRCT_SUPPORT_BUILDERUTILS_H

#include <variant>

#include "circt/Support/LLVM.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/ImplicitLocOpBuilder.h"
#include "llvm/ADT/StringSet.h"
#include "llvm/ADT/TypeSwitch.h"

namespace circt {

/// A helper union that can represent a `StringAttr`, `StringRef`, or `Twine`.
/// It is intended to be used as arguments to an op's `build` function. This
/// allows a single builder to accept any flavor value for a string attribute.
/// The `get` function can then be used to obtain a `StringAttr` from any of the
/// possible variants `StringAttrOrRef` can take.
class StringAttrOrRef {
  std::variant<StringAttr, StringRef, Twine, const char *> value;

public:
  StringAttrOrRef() : value() {}
  StringAttrOrRef(StringAttr attr) : value(attr) {}
  StringAttrOrRef(const StringRef &str) : value(str) {}
  StringAttrOrRef(const char *ptr) : value(ptr) {}
  StringAttrOrRef(const std::string &str) : value(StringRef(str)) {}
  StringAttrOrRef(const Twine &twine) : value(twine) {}

  /// Return the represented string as a `StringAttr`.
  StringAttr get(MLIRContext *context) const {
    if (auto *attr = std::get_if<StringAttr>(&value))
      return *attr;
    if (auto *ref = std::get_if<StringRef>(&value))
      return StringAttr::get(context, *ref);
    if (auto *twine = std::get_if<Twine>(&value))
      return StringAttr::get(context, *twine);
    if (auto *ptr = std::get_if<const char *>(&value))
      return StringAttr::get(context, *ptr);
    return StringAttr{};
  }
};

} // namespace circt

#endif // CIRCT_SUPPORT_BUILDERUTILS_H
