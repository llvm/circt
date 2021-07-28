//===- PrintStateGraph.cpp - Print graph of FSM states ----------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "PassDetails.h"
#include "circt/Dialect/FSM/FSMPasses.h"
#include "circt/Dialect/FSM/FSMStateGraph.h"
#include "llvm/Support/DOTGraphTraits.h"
#include "llvm/Support/GraphWriter.h"
#include "llvm/Support/raw_ostream.h"

using namespace circt;
using namespace fsm;

template <>
struct llvm::DOTGraphTraits<MachineOp> : public llvm::DefaultDOTGraphTraits {
  using DefaultDOTGraphTraits::DefaultDOTGraphTraits;

  static std::string getNodeLabel(Operation *node, MachineOp) {
    return cast<StateOp>(node).sym_name().str();
  }
};

namespace {
struct PrintStateGraphPass : public PrintStateGraphBase<PrintStateGraphPass> {
  PrintStateGraphPass(raw_ostream &os) : os(os) {}
  void runOnOperation() override {
    for (auto machine : getOperation().getOps<MachineOp>())
      llvm::WriteGraph(os, machine, /*ShortNames=*/false, machine.getName());
    markAllAnalysesPreserved();
  }
  raw_ostream &os;
};
} // end anonymous namespace

std::unique_ptr<mlir::Pass> circt::fsm::createPrintStateGraphPass() {
  return std::make_unique<PrintStateGraphPass>(llvm::errs());
}
