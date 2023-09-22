//===- FieldRefCache.cpp - FieldRef cache ---------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines FieldRefCache, a caching getFieldRefFromValue that caching
// FieldRef's for each queried value and all indexing operations visited during
// the walk.
//
//===----------------------------------------------------------------------===//

#include "circt/Dialect/FIRRTL/FieldRefCache.h"
#include "circt/Dialect/FIRRTL/FIRRTLUtils.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/FormatVariadic.h"

using namespace circt;
using namespace firrtl;

FieldRef firrtl::FieldRefCache::getFieldRefFromValue(Value value,
                                                     bool lookThroughCasts) {
  using Info = std::pair<Value, size_t>;
  // Vector of values and delta to next entry.
  // Last will be root (delta == 0), so walking backwards constructs
  // FieldRef's for all visited operations.
  SmallVector<Info> indexing;
  // Ignore null value for simplicity.
  if (!value)
    return FieldRef();

#ifndef NDEBUG
  ++queries;
#endif

  while (value) {
    // Check cache to see if already visited.
    auto it = refs.find(Key(value, lookThroughCasts));
    if (it != refs.end()) {
      // Found! If entire query is hit, we're done.
#ifndef NDEBUG
      ++hits;
#endif
      auto ref = it->second;
      if (indexing.empty())
        return ref;
      // Otherwise, add entry for this using cached FieldRef,
      // and add the FieldRef's value as last (root) entry.
      indexing.emplace_back(value, ref.getFieldID());
      indexing.emplace_back(ref.getValue(), 0);
      break;
    }
#ifndef NDEBUG
    ++computed;
#endif

    Operation *op = value.getDefiningOp();

    // If this is a block argument, we are done.
    if (!op) {
      indexing.emplace_back(value, 0);
      break;
    }

    auto [newValue, adj] =
        TypeSwitch<Operation *, Info>(op)
            .Case<RefCastOp, ConstCastOp, UninferredResetCastOp>([&](auto op) {
              if (!lookThroughCasts)
                return Info{};
              return Info{op.getInput(), 0};
            })
            .Case<SubfieldOp, OpenSubfieldOp>([&](auto subfieldOp) {
              typename decltype(subfieldOp)::InputType bundleType =
                  subfieldOp.getInput().getType();
              return Info{subfieldOp.getInput(),
                          bundleType.getFieldID(subfieldOp.getFieldIndex())};
            })
            .Case<SubindexOp, OpenSubindexOp>([&](auto subindexOp) {
              typename decltype(subindexOp)::InputType vecType =
                  subindexOp.getInput().getType();
              return Info{subindexOp.getInput(),
                          vecType.getFieldID(subindexOp.getIndex())};
            })
            .Case<RefSubOp>([&](RefSubOp refSubOp) {
              auto refInputType = refSubOp.getInput().getType();
              size_t delta = FIRRTLTypeSwitch<FIRRTLBaseType, size_t>(
                                 refInputType.getType())
                                 .Case<FVectorType, BundleType>([&](auto type) {
                                   return type.getFieldID(refSubOp.getIndex());
                                 });
              return Info{refSubOp.getInput(), delta};
            })
            .Default(Info{});
    indexing.emplace_back(value, adj); // adj is zero in 'unhandled' case.
    value = newValue;
  }
  // Last entry in indexing is the root.
  assert(!indexing.empty());
  assert(indexing.back().second == 0);

  auto root = indexing.back().first;
  size_t id = 0;
  FieldRef cur(root, 0);
  for (auto &info : llvm::reverse(indexing)) {
    id += info.second;
    cur = FieldRef(root, id);
    refs.try_emplace({info.first, lookThroughCasts}, cur);
  }
  return cur;
}

#ifndef NDEBUG
void firrtl::FieldRefCache::printStats(llvm::raw_ostream &os) const {
  os << llvm::formatv("FieldRefCache stats:\n"
                      "\thits:     {0}\n"
                      "\tcomputed: {1}\n"
                      "\tqueries:  {2}\n",
                      hits, computed, queries);
}
void firrtl::FieldRefCache::addToTotals(size_t &totalHits,
                                        size_t &totalComputed,
                                        size_t &totalQueries) const {
  totalHits += hits;
  totalComputed += computed;
  totalQueries += queries;
}

void firrtl::FieldRefCache::verifyImpl() const {
  // (Guarding under EXPENSIVE_CHECKS may be appropriate.)
  for (auto &[key, ref] : refs) {
    assert(ref == firrtl::getFieldRefFromValue(key.getPointer(), key.getInt()));
  }
}
#endif // NDEBUG
