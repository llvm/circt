//===- Dependence.h - Dependences in scheduling problems --------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This header defines the Dependence class and sets up the required machinery
// to use it as a key inside an llvm::DenseMap.
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_SUPPORT_SCHEDULING_DEPENDENCE_H
#define CIRCT_SUPPORT_SCHEDULING_DEPENDENCE_H

#include "circt/Support/LLVM.h"

#include "llvm/ADT/DenseMapInfo.h"
#include "llvm/ADT/Hashing.h"

namespace circt {
namespace sched {

/// This class models a dependence between two operations in context of a
/// scheduling problem. Conceptually, an instance of this class is a tuple
///   (`src`, `srcIdx`, `dst`, `dstIdx`)
/// representing a dependence from a specific result (`srcIdx`) of the source
/// operation (`src`) to a specific operand (`dstIdx`) of the destination
/// operation (`dst`).
class Dependence {
private:
  Operation *src, *dst;
  unsigned srcIdx, dstIdx;

  Dependence() : Dependence(nullptr, nullptr) {}
  friend llvm::DenseMapInfo<Dependence>;

public:
  /// Construct a dependence that distinguishes result and operands indices.
  Dependence(Operation *src, unsigned srcIdx, Operation *dst, unsigned dstIdx)
      : src(src), dst(dst), srcIdx(srcIdx), dstIdx(dstIdx) {}

  /// Construct a dependence that uses the default result and operand indices.
  Dependence(Operation *src, Operation *dst) : Dependence(src, 0, dst, 0) {}

  Operation *getSrc() const { return src; }
  Operation *getDst() const { return dst; }
  unsigned getSrcIdx() const { return srcIdx; }
  unsigned getDstIdx() const { return dstIdx; }

  bool operator==(const Dependence &other) const {
    return src == other.src && dst == other.dst && srcIdx == other.srcIdx &&
           dstIdx == other.dstIdx;
  }
};

/// Get a hash code for a Dependence.
inline ::llvm::hash_code hash_value(const Dependence &dep) {
  return llvm::hash_combine(dep.getSrc(), dep.getSrcIdx(), dep.getDst(),
                            dep.getDstIdx());
}

} // namespace sched
} // namespace circt

namespace llvm {
/// Allow using Dependences with DenseMaps.
template <>
struct DenseMapInfo<circt::sched::Dependence> {
  static inline circt::sched::Dependence getEmptyKey() {
    return circt::sched::Dependence();
  }
  static inline circt::sched::Dependence getTombstoneKey() {
    return circt::sched::Dependence();
  }
  static unsigned getHashValue(const circt::sched::Dependence &val) {
    return circt::sched::hash_value(val);
  }
  static bool isEqual(const circt::sched::Dependence &lhs,
                      const circt::sched::Dependence &rhs) {
    return lhs == rhs;
  }
};
} // namespace llvm

#endif // CIRCT_SUPPORT_SCHEDULING_DEPENDENCE_H
