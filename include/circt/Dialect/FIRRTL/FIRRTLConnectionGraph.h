//===- FIRRTLConnectionGraph.h - Graph of connections in FIRRTL -*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file provides utilities for working with FIRRTL operations using LLVM
// graph utilties.
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_DIALECT_FIRRTL_FIRRTLCONNECTIONGRAPH_H
#define CIRCT_DIALECT_FIRRTL_FIRRTLCONNECTIONGRAPH_H

#include "circt/Dialect/FIRRTL/FIRRTLOps.h"
#include "circt/Support/LLVM.h"
#include "mlir/IR/Operation.h"
#include "llvm/ADT/GraphTraits.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/ADT/iterator.h"
#include <mlir/IR/Value.h>
#include <mlir/IR/ValueRange.h>
#include <variant>

namespace circt {
namespace firrtl {
namespace detail {

// "Using" is used here to avoid polluting the global namespace with
// CIRCT-specific graph traits.  This pattern is borrowed from HWModuleGraph.h.
using FIRRTLOperation = mlir::Operation;

/// An iterator over connections to a module's ports.
class FModuleOpIterator
    : public llvm::iterator_facade_base<
          FModuleOpIterator, std::forward_iterator_tag, FIRRTLOperation> {

  FModuleOp op;

  Block::iterator iterator, iteratorEnd;

public:
  FModuleOpIterator() = default;

  FModuleOpIterator(FModuleOp op);

  bool operator==(const FModuleOpIterator &other) const;

  FIRRTLOperation &operator*() const {
    // return *(operations.back());
    return *iterator;
  }

  FModuleOpIterator &operator++();
};

/// An iterator over the connections of an FConnectLikeOp.
class FConnectLikeIterator
    : public llvm::iterator_facade_base<
          FConnectLikeIterator, std::forward_iterator_tag, FIRRTLOperation> {

  FConnectLike op;

  bool visited = true;

public:
  FConnectLikeIterator() = default;

  FConnectLikeIterator(FConnectLike op) : op(op), visited(false) {}

  bool operator==(const FConnectLikeIterator &other) const;

  FIRRTLOperation &operator*();

  FConnectLikeIterator &operator++();
};

/// Generic iterator for anything that is not an FConnectLike or an FModuleOp.
class ResultIterator
    : public llvm::iterator_facade_base<
          ResultIterator, std::forward_iterator_tag, FIRRTLOperation> {

  FIRRTLOperation *op;

  Value::use_iterator resultIterator, resultIteratorEnd;
  int resultIndex = 0, resultIndexEnd = 0;

  /// Iterate through results which have no users or are used as the
  /// dsetinations of connects.
  void fastforward();

public:
  ResultIterator() = default;

  ResultIterator(FIRRTLOperation *op);

  bool operator==(const ResultIterator &other) const;

  FIRRTLOperation &operator*() const;

  ResultIterator &operator++();
};

class ConnectionIterator {

  using VariantIterator = std::variant<std::monostate, FModuleOpIterator,
                                       FConnectLikeIterator, ResultIterator>;

  FIRRTLOperation *op = nullptr;

  VariantIterator iterator;

public:
  ConnectionIterator(FIRRTLOperation *op, bool empty = false);

  bool operator==(const ConnectionIterator &other) const;

  bool operator!=(const ConnectionIterator &other) const;

  FIRRTLOperation *operator*();

  ConnectionIterator &operator++();

  ConnectionIterator operator++(int);

  static ConnectionIterator childBegin(FIRRTLOperation *op) {
    return ConnectionIterator(op);
  }

  static ConnectionIterator childEnd(FIRRTLOperation *op) {
    return ConnectionIterator(op, /*empty=*/true);
  }
};
} // namespace detail
} // namespace firrtl
} // namespace circt

namespace llvm {

template <>
struct GraphTraits<circt::firrtl::detail::FIRRTLOperation *> {
  using ChildIteratorType = circt::firrtl::detail::ConnectionIterator;
  using Node = circt::firrtl::detail::FIRRTLOperation;
  using NodeRef = Node *;

  static NodeRef getEntryNode(NodeRef op) { return op; }
  // NOLINTNEXTLINE(readability-identifier-naming)
  static ChildIteratorType child_begin(NodeRef op) {
    return circt::firrtl::detail::ConnectionIterator::childBegin(op);
  }
  // NOLINTNEXTLINE(readability-identifier-naming)
  static ChildIteratorType child_end(NodeRef op) {
    return circt::firrtl::detail::ConnectionIterator::childEnd(op);
  }
};

} // namespace llvm

#endif // CIRCT_DIALECT_FIRRTL_FIRRTLCONNECTIONGRAPH_H
