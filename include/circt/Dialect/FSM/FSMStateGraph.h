//===- FSMStateGraph.h - Define graph of FSM states -----------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_DIALECT_FSM_FSMSTATEGRAPH_H
#define CIRCT_DIALECT_FSM_FSMSTATEGRAPH_H

#include "FSMOps.h"
#include "llvm/ADT/GraphTraits.h"
#include "llvm/Support/PointerLikeTypeTraits.h"

using namespace circt;
using namespace fsm;

namespace llvm {
/// Allow stealing the low bits of StateOp.
template <>
struct PointerLikeTypeTraits<StateOp> {
public:
  static inline void *getAsVoidPointer(StateOp state) {
    return const_cast<void *>(state.getAsOpaquePointer());
  }
  static inline StateOp getFromVoidPointer(void *p) {
    return StateOp::getFromOpaquePointer(p);
  }
  static constexpr int NumLowBitsAvailable = 3;
};

/// GraphTrait on MachineOp.
template <>
struct GraphTraits<MachineOp> {
  using NodeRef = Operation *;

  /// Helper for getting the next state referenced by the transition op. This
  /// implementation is not efficient because the `getReferencedNextState` has
  /// an complexity of O(n), where n is the number of states in the machine.
  static NodeRef getNextState(TransitionOp transition) {
    return transition.getReferencedNextState();
  }

  using TransitionIterator =
      mlir::detail::op_iterator<TransitionOp, Region::OpIterator>;
  using ChildIteratorType =
      mapped_iterator<TransitionIterator, decltype(&getNextState)>;

  static NodeRef getEntryNode(MachineOp machine) {
    return machine.getDefaultState();
  }
  static ChildIteratorType child_begin(NodeRef node) {
    auto &region = cast<StateOp>(node).transitions();
    return {region.getOps<TransitionOp>().begin(), &getNextState};
  }
  static ChildIteratorType child_end(NodeRef node) {
    auto &region = cast<StateOp>(node).transitions();
    return {region.getOps<TransitionOp>().end(), &getNextState};
  }

  using nodes_iterator = mlir::detail::op_iterator<StateOp, Region::OpIterator>;

  static nodes_iterator nodes_begin(MachineOp machine) {
    return machine.getOps<StateOp>().begin();
  }
  static nodes_iterator nodes_end(MachineOp machine) {
    return machine.getOps<StateOp>().end();
  }
};
} // namespace llvm

#endif // CIRCT_DIALECT_FSM_FSMSTATEGRAPH_H
