//===- FSMGraph.cpp - FSM Graph ---------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "circt/Dialect/FSM/FSMGraph.h"

using namespace circt;
using namespace fsm;

FSMTransitionEdge *FSMStateNode::addTransition(FSMStateNode *nextState,
                                               TransitionOp transition) {
  auto *transitionEdge = new FSMTransitionEdge(this, transition, nextState);
  transitions.push_back(transitionEdge);
  return transitionEdge;
}

FSMGraph::FSMGraph(Operation *op) {
  machine = dyn_cast<MachineOp>(op);
  assert(machine && "Expected a fsm::MachineOp");

  // Find all states in the machine.
  for (auto stateOp : machine.getOps<StateOp>()) {
    // Add an edge to indicate that this state transitions to some other state.
    auto *currentStateNode = getOrAddNode(stateOp);

    for (auto transitionOp : stateOp.transitions().getOps<TransitionOp>()) {
      auto *nextStateNode = getOrAddNode(transitionOp.getNextState());
      currentStateNode->addTransition(nextStateNode, transitionOp);
    }
  }
}

FSMStateNode *FSMGraph::lookup(StringAttr name) {
  auto it = nodeMap.find(name);
  assert(it != nodeMap.end() && "Module not in InstanceGraph!");
  return it->second;
}

FSMStateNode *FSMGraph::lookup(StateOp state) {
  return lookup(state.getNameAttr());
}

FSMStateNode *FSMGraph::getOrAddNode(StateOp state) {
  // Try to insert an FSMStateNode. If its not inserted, it returns
  // an iterator pointing to the node.
  auto *&node = nodeMap[state.getNameAttr()];
  if (!node) {
    node = new FSMStateNode(state);
    nodes.push_back(node);
  }
  return node;
}
