//===- DependenceIterator.h - Uniform handling of dependences ---*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines utilities to let algorithms iterate over different flavors
// of dependences in a uniform way.
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_SCHEDULING_DEPENDENCEITERATOR_H
#define CIRCT_SCHEDULING_DEPENDENCEITERATOR_H

#include "circt/Support/LLVM.h"

#include "llvm/ADT/Optional.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/iterator.h"

namespace circt {
namespace scheduling {

class Scheduler;

namespace detail {

/// Optional identifier, used to access dependence properties.
using DependenceId = unsigned;

/// A POD type to pass around the info comprising a dependence edge. This is
/// intended to be used as a tuple with nicer element names, not for long-term
/// storage.
struct Dependence {
  /// The source of the dependence.
  Operation *src;
  /// The destination of the dependence.
  Operation *dst;
  /// The source operation's result number, if applicable.
  Optional<unsigned> srcIdx;
  /// The destination operation's operand number, if applicable.
  Optional<unsigned> dstIdx;
  /// The dependence's index into the property maps, if previously assigned.
  Optional<DependenceId> id;
  bool operator==(const Dependence &other) const;
};

/// An iterator to transparently surface an operation's def-use dependences from
/// the SSA subgraph (induced by the registered operations), as well as
/// auxiliary, operation-to-operation dependences explicitly provided by the
/// client.
class DependenceIterator
    : public llvm::iterator_facade_base<DependenceIterator,
                                        std::forward_iterator_tag, Dependence> {
public:
  static constexpr Dependence invalid = {nullptr, nullptr, None, None, None};

  /// Construct an iterator over the \p op's def-use dependences (i.e. result
  /// values of other operations registered in the scheduling problem, which are
  /// used by one of \p op's operands), and over auxiliary dependences (i.e.
  /// from other operation to \p op).
  DependenceIterator(Scheduler &scheduler, Operation *op, bool end = false);

  bool operator==(const DependenceIterator &other) const {
    return dependence == other.dependence;
  }

  const Dependence &operator*() const { return dependence; }

  DependenceIterator &operator++() {
    findNextDependence();
    return *this;
  }

private:
  void findNextDependence();

  Scheduler &scheduler;
  Operation *op;

  unsigned operandIdx;
  unsigned auxPredIdx;
  llvm::SmallSetVector<Operation *, 4> *auxPreds;

  Dependence dependence;
};

} // namespace detail
} // namespace scheduling
} // namespace circt

#endif // CIRCT_SCHEDULING_DEPENDENCEITERATOR_H
