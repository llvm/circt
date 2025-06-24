//===- KanagawaEliminateRedundantOps.cpp ----------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "circt/Dialect/Kanagawa/KanagawaDialect.h"
#include "circt/Dialect/Kanagawa/KanagawaOps.h"
#include "circt/Dialect/Kanagawa/KanagawaPasses.h"
#include "circt/Dialect/Kanagawa/KanagawaTypes.h"

#include "mlir/Pass/Pass.h"

#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "kanagawa-eliminate-redundant-ops"

namespace circt {
namespace kanagawa {
#define GEN_PASS_DEF_KANAGAWAELIMINATEREDUNDANTOPS
#include "circt/Dialect/Kanagawa/KanagawaPasses.h.inc"
} // namespace kanagawa
} // namespace circt

using namespace circt;
using namespace kanagawa;

// Helper function to eliminate redundant GetPortOps in a container
static void eliminateRedundantGetPortOps(ContainerOp containerOp) {
  // Structure to track accesses to each port of each instance
  struct PortAccesses {
    GetPortOp getAsInput;
    GetPortOp getAsOutput;
  };

  llvm::DenseMap</*instance*/ Value,
                 /*portName*/ llvm::DenseMap<StringAttr, PortAccesses>>
      instancePortAccessMap;

  // Collect all GetPortOps and identify redundant ones
  llvm::SmallVector<GetPortOp> redundantOps;

  for (auto getPortOp : containerOp.getOps<GetPortOp>()) {
    PortAccesses &portAccesses =
        instancePortAccessMap[getPortOp.getInstance()]
                             [getPortOp.getPortSymbolAttr().getAttr()];

    if (getPortOp.getDirection() == Direction::Input) {
      if (portAccesses.getAsInput) {
        // Found redundant input GetPortOp - mark the current one for removal
        // and replace its uses with the existing one
        getPortOp.replaceAllUsesWith(portAccesses.getAsInput.getResult());
        redundantOps.push_back(getPortOp);
      } else {
        portAccesses.getAsInput = getPortOp;
      }
    } else {
      if (portAccesses.getAsOutput) {
        // Found redundant output GetPortOp - mark the current one for removal
        // and replace its uses with the existing one
        getPortOp.replaceAllUsesWith(portAccesses.getAsOutput.getResult());
        redundantOps.push_back(getPortOp);
      } else {
        portAccesses.getAsOutput = getPortOp;
      }
    }
  }

  // Remove all redundant GetPortOps
  for (auto redundantOp : redundantOps)
    redundantOp.erase();
}

// Helper function to eliminate redundant PortReadOps in a container
static void eliminateRedundantPortReadOps(ContainerOp containerOp) {
  // Map to track the first PortReadOp for each port being read from
  llvm::DenseMap<Value, PortReadOp> portFirstReadMap;
  llvm::SmallVector<PortReadOp> redundantOps;

  for (auto portReadOp : containerOp.getOps<PortReadOp>()) {
    Value portBeingRead =
        portReadOp.getPort(); // The port operand being read from

    if (auto existingRead = portFirstReadMap.lookup(portBeingRead)) {
      // Found redundant PortReadOp - mark the current one for removal
      // and replace its uses with the existing one
      portReadOp.replaceAllUsesWith(existingRead.getResult());
      redundantOps.push_back(portReadOp);
    } else {
      portFirstReadMap[portBeingRead] = portReadOp;
    }
  }

  // Remove all redundant PortReadOps
  for (auto redundantOp : redundantOps)
    redundantOp.erase();
}

namespace {

struct EliminateRedundantOpsPass
    : public circt::kanagawa::impl::KanagawaEliminateRedundantOpsBase<
          EliminateRedundantOpsPass> {
  void runOnOperation() override {
    ContainerOp containerOp = getOperation();
    eliminateRedundantGetPortOps(containerOp);
    eliminateRedundantPortReadOps(containerOp);
  }
};

} // anonymous namespace

std::unique_ptr<mlir::Pass> circt::kanagawa::createEliminateRedundantOpsPass() {
  return std::make_unique<EliminateRedundantOpsPass>();
}
