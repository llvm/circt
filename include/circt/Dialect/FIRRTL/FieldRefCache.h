//===- FieldRefCache.h - FieldRef cache -------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file declares FieldRefCache, a caching getFieldRefFromValue that
// caching FieldRef's for each queried value and all indexing operations
// visited during the walk.
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_DIALECT_FIRRTL_FIELDREFCACHE_H
#define CIRCT_DIALECT_FIRRTL_FIELDREFCACHE_H

#include "circt/Support/FieldRef.h"
#include "circt/Support/LLVM.h"

namespace circt {
namespace firrtl {

/// Caching version of getFieldRefFromValue.  Computes the requested FieldRef
/// and for all operations visited along the way.  Tracks some stats in debug.
class FieldRefCache {
  using Key = llvm::PointerIntPair<Value, 1, bool>;
  DenseMap<Key, FieldRef> refs;
#ifndef NDEBUG
  size_t computed = 0;
  size_t hits = 0;
  size_t queries = 0;
#endif

public:
  /// Caching version of getFieldRefFromValue.
  FieldRef getFieldRefFromValue(Value value, bool lookThroughCasts = false);

  /// Drop all cached entries.
  void clear() { refs.clear(); }

#ifndef NDEBUG
  void printStats(llvm::raw_ostream &os) const;
  void addToTotals(size_t &totalHits, size_t &totalComputed,
                   size_t &totalQueries) const;
  void verifyImpl() const;
#endif // NDEBUG

  /// Verify cached fieldRefs against firrtl::getFieldRefFromValue.
  /// No-op in release builds.
  void verify() const {
#ifndef NDEBUG
    verifyImpl();
#endif // NDEBUG
  }
};

} // namespace firrtl
} // namespace circt

#endif // CIRCT_DIALECT_FIRRTL_FIELDREFCACHE_H
