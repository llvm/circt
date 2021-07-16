//===- StateGraph.h - Define graph of FSM states --------------------------===//
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_DIALECT_FSM_STATEGRAPH_H
#define CIRCT_DIALECT_FSM_STATEGRAPH_H

#include "FSMOps.h"
#include "llvm/ADT/GraphTraits.h"
#include "llvm/Support/PointerLikeTypeTraits.h"

namespace llvm {
/// Allow stealing the low bits of StateOp.
template <>
struct PointerLikeTypeTraits<circt::fsm::StateOp> {
public:
  static inline void *getAsVoidPointer(circt::fsm::StateOp state) {
    return const_cast<void *>(state.getAsOpaquePointer());
  }
  static inline circt::fsm::StateOp getFromVoidPointer(void *p) {
    return circt::fsm::StateOp::getFromOpaquePointer(p);
  }
  static constexpr int NumLowBitsAvailable = 3;
};

/// GraphTrait on MachineOp.
template <>
struct GraphTraits<circt::fsm::MachineOp> {
  using NodeRef = circt::fsm::StateOp;

  // Helper for getting the next state referenced by the transition op.
  static NodeRef getNextState(circt::fsm::TransitionOp transition) {
    return transition.getReferencedNextState();
  }

  using TransitionIterator =
      mlir::detail::op_iterator<circt::fsm::TransitionOp,
                                mlir::Region::OpIterator>;
  using ChildIteratorType =
      llvm::mapped_iterator<TransitionIterator, decltype(&getNextState)>;

  static NodeRef getEntryNode(circt::fsm::MachineOp machine) {
    return machine.getDefaultState();
  }
  static ChildIteratorType child_begin(NodeRef node) {
    return {node.transitions().getOps<circt::fsm::TransitionOp>().begin(),
            &getNextState};
  }
  static ChildIteratorType child_end(NodeRef node) {
    return {node.transitions().getOps<circt::fsm::TransitionOp>().end(),
            &getNextState};
  }
};
} // namespace llvm

#endif // CIRCT_DIALECT_FSM_STATEGRAPH_H
