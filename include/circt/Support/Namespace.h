//===- Namespace.h - Utilities for generating names -------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file provides utilities for generating new names that do not conflict
// with existing names.
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_SUPPORT_NAMESPACE_H
#define CIRCT_SUPPORT_NAMESPACE_H

#include "circt/Support/LLVM.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/StringSet.h"
#include "llvm/ADT/Twine.h"

namespace circt {

/// A namespace that is used to store existing names and generate new names in
/// some scope within the IR. This exists to work around limitations of
/// SymbolTables. This acts as a base class providing facilities common to all
/// namespaces implementations.
class Namespace {
public:
  Namespace() {}
  Namespace(const Namespace &other) = default;
  Namespace(Namespace &&other) : nextIndex(std::move(other.nextIndex)) {}

  Namespace &operator=(const Namespace &other) = default;
  Namespace &operator=(Namespace &&other) {
    nextIndex = std::move(other.nextIndex);
    return *this;
  }

  /// Empty the namespace.
  void clear() { nextIndex.clear(); }

  /// Return a unique name, derived from the input `name`, and add the new name
  /// to the internal namespace.  There are two possible outcomes for the
  /// returned name:
  ///
  /// 1. The original name is returned.
  /// 2. The name is given a `_<n>` suffix where `<n>` is a number starting from
  ///    `_0` and incrementing by one each time.
  StringRef newName(const Twine &name) {
    // Special case the situation where there is no name collision to avoid
    // messing with the SmallString allocation below.
    llvm::SmallString<64> tryName;
    auto inserted = nextIndex.insert({name.toStringRef(tryName), 0});
    if (inserted.second)
      return inserted.first->getKey();

    // Try different suffixes until we get a collision-free one.
    if (tryName.empty())
      name.toVector(tryName); // toStringRef may leave tryName unfilled

    // Indexes less than nextIndex[tryName] are lready used, so skip them.
    // Indexes larger than nextIndex[tryName] may be used in another name.
    size_t &i = nextIndex[tryName];
    tryName.push_back('_');
    size_t baseLength = tryName.size();
    do {
      tryName.resize(baseLength);
      Twine(i++).toVector(tryName); // append integer to tryName
      inserted = nextIndex.insert({tryName, 0});
    } while (!inserted.second);

    return inserted.first->getKey();
  }

protected:
  // The "next index" that will be tried when trying to unique a string within a
  // namespace.  It follows that all values less than the "next index" value are
  // already used.
  llvm::StringMap<size_t> nextIndex;
};

} // namespace circt

#endif // CIRCT_SUPPORT_NAMESPACE_H
