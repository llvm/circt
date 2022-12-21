//===- InferReadWrite.cpp - Infer Read Write Memory -----------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines the InferReadWrite pass.
//
//===----------------------------------------------------------------------===//

#include "../Transforms/PassDetails.h"

#include "mlir/IR/BuiltinOps.h"

#include <set>

using namespace mlir;
using namespace circt::firrtl;

namespace {
class LatticeValue {
  enum Kind {
    /// A value with a yet-to-be-determined value.  This is an unprocessed
    /// value.
    Unknown,

    // Value is live, but derives from an indeterminate value.
    Undefined,

    // Value is live and derived from a controlled value.
    Valid,

    // Value is live and derived from an external signal.
    External
  };

public:
  LatticeValue() : tag(Kind::Unknown) {}

  bool markUndefined() {
    if (tag == Undefined)
      return false;
    tag = Undefined;
    return true;
  }
  bool markValid() {
    if (tag == Valid)
      return false;
    tag = Valid;
    return true;
  }
  bool markExternal() {
    if (tag == External)
      return false;
    tag = External;
    return true;
  }

  bool isUndefined() { return tag == Undefined; }
  bool isValid() { return tag == Valid; }
  bool isExternal() { return tag == External; }

private:
  Kind tag;
};

struct InstPath {
    SmallVector<InstanceOp> path;
};

using ValueKey = std::tuple<Value, const InstPath*>;
using BlockKey = std::tuple<Block*, const InstPath*>;

} // namespace
/*
undef = undef op *
valid = valid op valid
Extern = Extern op (valid, Extern)

undef = reg no-init
undef = reg init sig x, val y (x | y is undef)
extern = reg init sig x, val y (x | y is extern and x & y is not undef)


map<Value, LatticeValue>


for each M : Modules {
    for each e : M {
        visitor(e);
    }
}
while (!WL.empty()) {
    visit(wl.pop());
}
*/

namespace {
struct UndefAnalysisPass : public UndefAnalysisBase<UndefAnalysisPass> {
  void runOnOperation() override;

  void markBlockExecutable(BlockKey block);
  void visitOperation(Operation *op);

  /// Returns true if the given block is executable.
  bool isBlockExecutable(BlockKey block) const {
    return executableBlocks.count(block);
  }

  void markUndefined(ValueKey value) {
    auto &entry = latticeValues[value];
    if (!entry.isUndefined()) {
      entry.markUndefined();
      changedLatticeValueWorklist.push_back(value);
    }
  }

  void markExternal(ValueKey value) {
    auto &entry = latticeValues[value];
    if (!entry.isExternal()) {
      entry.markExternal();
      changedLatticeValueWorklist.push_back(value);
    }
  }

  void markValid(ValueKey value) {
    auto &entry = latticeValues[value];
    if (!entry.isValid()) {
      entry.markValid();
      changedLatticeValueWorklist.push_back(value);
    }
  }

  InstPath* ExtendPath(InstPath*, InstanceOp);

private:

  /// This keeps track of the current state of each tracked value.
  DenseMap<ValueKey, LatticeValue> latticeValues;

  /// A worklist of values whose LatticeValue recently changed, indicating the
  /// users need to be reprocessed.
  SmallVector<ValueKey, 64> changedLatticeValueWorklist;

  /// The set of blocks that are known to execute, or are intrinsically live.
  std::set<BlockKey> executableBlocks;

  std::set<InstPath> uniqPaths;
};
} // namespace

void UndefAnalysisPass::runOnOperation() {
  auto circuit = getOperation();

  // Mark the input ports of the top-level modules as being external.  We ignore
  // all other public modules.
  auto top = cast<FModuleOp>(circuit.getMainModule());
  auto* topPath = &*uniqPaths.emplace().first;
  for (auto port : top.getBodyBlock()->getArguments())
    markExternal({(Value)port, topPath});
  markBlockExecutable({top.getBodyBlock(), topPath});

  // If a value changed lattice state then reprocess any of its users.
  while (!changedLatticeValueWorklist.empty()) {
    auto [changedVal,path] = changedLatticeValueWorklist.pop_back_val();
    for (Operation *user : changedVal.getUsers()) {
      if (isBlockExecutable({user->getBlock(), path}))
        visitOperation({user, path});
    }
  }
}

void UndefAnalysisPass::markBlockExecutable(BlockKey block) {
  if (!executableBlocks.insert(block).second)
    return;

  for (auto &op : *get<0>(block)) {
    visitOperation(op);
  }
}

void UndefAnalysisPass::visitOperation(Operation *op) {
  if (auto reg = dyn_cast<RegOp>(op))
    return markUndefined(reg);
  if (isa<ConstantOp>(op))
    return markValid(op)

        for (auto operand : op->getOperands()) for (auto result :
                                                    op->getResults())
            mergeValues(result, operand);
}

InstPath* ExtendPath(InstPath* path, InstanceOp inst) {
    auto newPath = *path.path;
    newPath.push_back(inst);
    auto ins = uniqPaths.insert(newPath);
    return &*inst.first;
}

std::unique_ptr<mlir::Pass> circt::firrtl::createUndefAnalysisPass() {
  return std::make_unique<UndefAnalysisPass>();
}
