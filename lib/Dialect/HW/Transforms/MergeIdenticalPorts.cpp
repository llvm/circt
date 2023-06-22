//===- MergeIdenticalPorts.cpp - Identical port merging pass---------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This pass merges ports for any hw::HWMutableModuleLike which, at all of
// its instantiation points, are always driven by the same inputs.
// Similarly, it removes any ports of modules which are driven by the
// same values.
//
//===----------------------------------------------------------------------===//

#include "PassDetails.h"
#include "circt/Dialect/HW/HWInstanceGraph.h"
#include "circt/Dialect/HW/HWOps.h"
#include "circt/Dialect/HW/HWPasses.h"
#include "mlir/IR/Builders.h"
#include "llvm/Support/Debug.h"

#include <map>

using namespace circt;
using namespace hw;

//===----------------------------------------------------------------------===//
// MergeIdenticalPortsPass
//===----------------------------------------------------------------------===//

namespace {

struct MergeIdenticalPortsPass
    : public hw::MergeIdenticalPortsBase<MergeIdenticalPortsPass> {
  void runOnOperation() override;

private:
  void visit(InstanceGraphNode *node);
};
} // end anonymous namespace

// Set vector for determinism.
using IndexSet = llvm::SetVector<size_t>;
using IdenticalIndices = llvm::SmallVector<IndexSet>;

static void getIdenticalIndices(ValueRange values, IdenticalIndices &ii) {
  DenseMap<Value, std::vector<size_t>> valueIndices;

  // Insert indices into the DenseMap
  for (size_t i = 0; i < values.size(); ++i)
    valueIndices[values[i]].push_back(i);

  // Collect indices of identical values
  for (const auto &pair : valueIndices) {
    IndexSet indices(pair.second.begin(), pair.second.end());
    ii.push_back(std::move(indices));
  }
}

void MergeIdenticalPortsPass::visit(InstanceGraphNode *node) {
  auto nodeMod = node->getModule();

  // Will only merge ports for HWModuleOp. HWMutableModuleLike would be an
  // ideal candidate, but we also need access to the block arguments and
  // terminator of the module, which is not part of the
  auto mod = dyn_cast<HWModuleOp>(nodeMod.getOperation());
  if (!mod)
    return;

  auto *ctx = mod->getContext();
  // Maintain a mapping between the users and the ports which they drive
  // identically.
  DenseMap<HWInstanceLike, IdenticalIndices> userIdenticalIndices;

  for (auto userIt : node->uses()) {
    HWInstanceLike user = userIt->getInstance();
    auto operands = user->getOperands();
    // Determine the indices of operands which are identical.
    auto &uii = userIdenticalIndices[user];
    getIdenticalIndices(operands, uii);

    if (uii.empty()) {
      // Early exit if there are no identical indices.
      return;
    }
  }

  // Determine the set of identical indices which are shared across all users.
  IdenticalIndices sharedIdenticalPorts;
  for (auto [_, identicalIndices] : userIdenticalIndices) {
    if (sharedIdenticalPorts.empty()) {
      // Initialize the shared set with the first user's identical ports.
      sharedIdenticalPorts = identicalIndices;
      continue;
    }

    IdenticalIndices newSharedIdenticalPorts;
    std::set_intersection(sharedIdenticalPorts.begin(),
                          sharedIdenticalPorts.end(), identicalIndices.begin(),
                          identicalIndices.end(),
                          std::back_inserter(newSharedIdenticalPorts));
    sharedIdenticalPorts = newSharedIdenticalPorts;
  }

  if (sharedIdenticalPorts.empty())
    return;

  // Check for identical outputs.
  hw::OutputOp outputOp =
      cast<hw::OutputOp>(mod.getBodyBlock()->getTerminator());
  IdenticalIndices identicalOutputPorts;
  getIdenticalIndices(outputOp.getOperands(), identicalOutputPorts);

  // We now have a set of port indices which are guaranteed to be identically
  // driven across all instantiations.
  // We erase all of the identical input and output indices, and create
  // a new port of each, which is named as the union of the names of the
  // ports which it replaces.
  llvm::SmallVector<unsigned> eraseInputs;
  llvm::SmallVector<unsigned> eraseOutputs;

  struct PortChange {
    // The set of indices which are being merged.
    IndexSet indices;
    // The portInfo for the new port which is replacing the merged ports.
    hw::PortInfo portInfo;

    // In case of a merged input port, the block argument of the new port.
    Value v;
  };
  llvm::SmallVector<PortChange> newInPorts, newOutPorts;

  auto modPortInfo = mod.getPorts();

  // Function for generating a name for a merged port by concatenating the
  // names of the ports which it replaces.
  auto generateName = [&](IndexSet indices, bool isOutput) -> StringAttr {
    std::string s;
    llvm::raw_string_ostream ss(s);
    llvm::interleave(
        indices, ss,
        [&](size_t index) { ss << modPortInfo.inputs[index].name.str(); }, "_");
    return StringAttr::get(ctx, s);
  };

  for (auto indices : sharedIdenticalPorts) {
    llvm::append_range(eraseInputs, indices);
    newInPorts.push_back(PortChange{
        indices,
        hw::PortInfo{generateName(indices, /*isOutput*/ false),
                     PortDirection::INPUT, modPortInfo.inputs[indices[0]].type},
        nullptr});
  }

  for (auto indices : identicalOutputPorts) {
    llvm::append_range(eraseOutputs, indices);
    newOutPorts.push_back(
        PortChange{indices,
                   hw::PortInfo{generateName(indices, /*isOutput*/ true),
                                PortDirection::OUTPUT,
                                modPortInfo.outputs[indices[0]].type},
                   nullptr});
  }

  llvm::SmallVector<std::pair<unsigned, hw::PortInfo>> newInsList, newOutsList;
  size_t insertInputsAt = mod.getNumInputs();
  size_t insertOutputsAt = mod.getNumOutputs();
  llvm::transform(newInPorts, std::back_inserter(newInsList),
                  [&](const PortChange &pc) {
                    return std::make_pair(insertInputsAt, pc.portInfo);
                  });
  llvm::transform(newOutPorts, std::back_inserter(newOutsList),
                  [&](const PortChange &pc) {
                    return std::make_pair(insertOutputsAt, pc.portInfo);
                  });

  // Ensure that the erasure indices are sorted in ascending order (requirement
  // for HWMutableModuleLike::modifyPorts).
  llvm::sort(eraseInputs);
  llvm::sort(eraseOutputs);

  // We now have all of the information we need to modify the module - Go!
  mod.modifyPorts(newInsList, newOutsList, eraseInputs, eraseOutputs);

  // Modify the block arguments of the module to correspond to the change
  // in ports. We do this by first adding new block arguments for the newly
  // added ports. Then, we iterate through the port replacement list and replace
  // all uses of the erased ports. Finally, we erase the old block arguments.
  Block *body = mod.getBodyBlock();
  for (auto &portChange : newInPorts) {
    // Generate a fused location for the new port.
    llvm::SmallVector<Location> locs;
    for (auto &idx : portChange.indices)
      locs.push_back(body->getArgument(idx).getLoc());

    body->addArgument(portChange.portInfo.type, FusedLoc::get(ctx, locs));
    portChange.v = body->getArgument(body->getNumArguments() - 1);

    // Replace all uses of the old ports with the new port.
    for (auto &idx : portChange.indices)
      body->getArgument(idx).replaceAllUsesWith(portChange.v);
  }

  // Erase the old block arguments. We do this in reverse order of eraseInputs
  // to ensure that the indices remain valid.
  for (auto idx : llvm::reverse(eraseInputs))
    body->eraseArgument(idx);

  // Modify the operands of the output op to correspond to the change in ports.
  llvm::SmallVector<Value> newOutputOperands;

  // Next, we need to handle the instantiation sites.
}

void MergeIdenticalPortsPass::runOnOperation() {
  circt::hw::InstanceGraph &analysis = getAnalysis<circt::hw::InstanceGraph>();
  auto res = analysis.getInferredTopLevelNodes();

  if (failed(res)) {
    signalPassFailure();
    return;
  }

  DenseSet<InstanceGraphNode *> visited;
  for (InstanceGraphNode *topModule : res.value()) {
    for (InstanceRecord *node : topModule->uses()) {
      if (!visited.insert(node->getTarget()).second)
        continue;
      visit(node->getTarget());
    }
  }
}

std::unique_ptr<Pass> circt::hw::createMergeIdenticalPortsPass() {
  return std::make_unique<MergeIdenticalPortsPass>();
}
