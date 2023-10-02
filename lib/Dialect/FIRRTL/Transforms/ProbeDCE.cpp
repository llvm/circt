//===- ProbeDCE.cpp - Delete input probes and dead dead probe ops ---------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines the ProbeDCE pass.  This pass ensures all input probes
// are removed and diagnoses where that is not possible.
//
//===----------------------------------------------------------------------===//

#include "PassDetails.h"
#include "circt/Dialect/FIRRTL/FIRRTLInstanceGraph.h"
#include "circt/Dialect/FIRRTL/FIRRTLOps.h"
#include "circt/Dialect/FIRRTL/FIRRTLTypes.h"
#include "circt/Dialect/FIRRTL/Passes.h"
#include "mlir/IR/ImplicitLocOpBuilder.h"
#include "mlir/IR/Iterators.h"
#include "mlir/IR/Threading.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/SaveAndRestore.h"

#define DEBUG_TYPE "firrtl-probe-dce"

using namespace circt;
using namespace firrtl;

//===----------------------------------------------------------------------===//
// Pass Infrastructure
//===----------------------------------------------------------------------===//

namespace {
struct ProbeDCEPass : public ProbeDCEBase<ProbeDCEPass> {
  using ProbeDCEBase::ProbeDCEBase;
  void runOnOperation() override;

  /// Process the specified module, instance graph only used to
  /// keep it up-to-date if specified.
  /// Returns true if changes were made (or reports failure).
  FailureOr<bool>
  process(FModuleLike mod,
          std::optional<std::reference_wrapper<InstanceGraph>> ig);
};
} // end anonymous namespace

void ProbeDCEPass::runOnOperation() {
  LLVM_DEBUG(llvm::dbgs()
             << "===- Running ProbeDCE Pass "
                "--------------------------------------------------===\n");

  SmallVector<Operation *, 0> ops(getOperation().getOps<FModuleLike>());

  auto ig = getCachedAnalysis<InstanceGraph>();

  std::atomic<bool> anyChanges(false);
  auto result = failableParallelForEach(&getContext(), ops, [&](Operation *op) {
    auto failOrChanged = process(cast<FModuleLike>(op), ig);
    if (failed(failOrChanged))
      return failure();
    auto changed = *failOrChanged;
    if (changed)
      anyChanges = true;
    return success();
  });

  if (result.failed())
    signalPassFailure();

  if (ig)
    markAnalysesPreserved<InstanceGraph>();
  if (!anyChanges)
    markAllAnalysesPreserved();
}

/// This is the pass constructor.
std::unique_ptr<mlir::Pass> circt::firrtl::createProbeDCEPass() {
  return std::make_unique<ProbeDCEPass>();
}

//===----------------------------------------------------------------------===//
// Per-module input probe removal.
//===----------------------------------------------------------------------===//

/// Forward slice of input probes.
class InputProbeForwardSlicer {
  // Build cumulatively, slice individually for diagnostic specificity.
  DenseSet<Operation *> slice;

  // Current slice source, only valid while slicing.
  BlockArgument currentSliceSource;

  /// Operation is in forward slice, add.
  void chase(SmallVectorImpl<Operation *> &worklist, Operation *op) {
    if (!slice.insert(op).second)
      return;
    worklist.push_back(op);
  }

  /// Forward slice through value -> users.
  void chaseUsers(SmallVectorImpl<Operation *> &worklist, Value v) {
    for (auto *user : v.getUsers())
      chase(worklist, user);
  }

  /// Upwards chase for connect, and chase all users.
  LogicalResult chaseVal(SmallVectorImpl<Operation *> &worklist, Value v) {
    if (auto depArg = dyn_cast<BlockArgument>(v)) {
      // Input probe flows to different argument?
      // With current (valid) IR this should not be possible.
      if (depArg != currentSliceSource)
        return emitError(depArg.getLoc(), "argument depends on input probe")
                   .attachNote(currentSliceSource.getLoc())
               << "input probe";
      // Shouldn't happen either, but safe to ignore.
      return success();
    }
    auto *op = v.getDefiningOp();
    assert(op);
    chase(worklist, op);
    chaseUsers(worklist, v);
    return success();
  }

public:
  bool contains(Operation *op) const { return slice.contains(op); }
  bool empty() const { return slice.empty(); }
  size_t size() const { return slice.size(); }

  const DenseSet<Operation *> &get() const { return slice; }

  /// Forward slice through the given input probe argument, diagnosing
  /// illegal/unsupported uses if encountered.
  LogicalResult add(BlockArgument arg) {
    // Set current slice source for use by helpers, clear when done.
    llvm::SaveAndRestore<BlockArgument> x(this->currentSliceSource, arg);
    // Slice worklist.
    SmallVector<Operation *> worklist;

    // Start with all users of the input probe.
    chaseUsers(worklist, arg);

    while (!worklist.empty()) {
      auto *op = worklist.pop_back_val();

      // Generally just walk users.  Only "backwards" chasing is for connect,
      // which only flows to the destination which must not be a refsub or cast.
      auto result =
          TypeSwitch<Operation *, LogicalResult>(op)
              .Case([&](InstanceOp inst) {
                // Ignore, only reachable via connect which also chases users
                // for us.  Don't chase results not in slice.
                return success();
              })
              .Case([&](FConnectLike connect) {
                return chaseVal(worklist, connect.getDest());
              })
              .Case([&](RefSubOp op) {
                chaseUsers(worklist, op.getResult());
                return success();
              })
              .Case([&](RefCastOp op) {
                chaseUsers(worklist, op.getResult());
                return success();
              })
              .Case([&](WireOp op) {
                // Ignore, only reachable via connect which also chases users
                // for us.
                return success();
              })
              .Default([&](auto *op) -> LogicalResult {
                return emitError(op->getLoc(), "input probes cannot be used")
                           .attachNote(arg.getLoc())
                       << "input probe here";
              });
      if (failed(result))
        return result;
    }
    return success();
  }
};

FailureOr<bool>
ProbeDCEPass::process(FModuleLike mod,
                      std::optional<std::reference_wrapper<InstanceGraph>> ig) {
  SmallVector<size_t> probePortIndices;

  // Find input probes.
  for (auto idx : llvm::seq(mod.getNumPorts())) {
    if (mod.getPortDirection(idx) == Direction::In &&
        type_isa<RefType>(mod.getPortType(idx)))
      probePortIndices.push_back(idx);
  }

  // Reject input probes across boundaries.
  if (!probePortIndices.empty() && mod.isPublic()) {
    auto idx = probePortIndices.front();
    return mlir::emitError(mod.getPortLocation(idx),
                           "input probe not allowed on public module");
  }
  auto modOp = dyn_cast<FModuleOp>(mod.getOperation());
  if (!modOp) {
    if (!probePortIndices.empty()) {
      auto idx = probePortIndices.front();
      return mlir::emitError(mod.getPortLocation(idx),
                             "input probe not allowed on this module kind");
    }
    // No changes made.
    return false;
  }

  BitVector portsToErase(mod.getNumPorts());

  // Forward slice over users of each.
  // Build cumulatively, slice individually for diagnostic specificity.
  InputProbeForwardSlicer slice;
  for (auto idx : probePortIndices) {
    portsToErase.set(idx);
    if (failed(slice.add(modOp.getArgument(idx))))
      return failure();
  }

  // Track whether any changes were made.
  bool changes = portsToErase.any();

  // Walk the module, removing all operations in forward slice from input probes
  // and updating all instances to reflect no more input probes anywhere.
  // Walk post-order, reverse, so can erase as we encounter operations.
  modOp.walk<mlir::WalkOrder::PostOrder, mlir::ReverseIterator>(
      [&](Operation *op) {
        auto inst = dyn_cast<InstanceOp>(op);
        // For everything that's not an instance, remove if in slice.
        if (!inst) {
          if (slice.contains(op)) {
            ++numErasedOps;
            changes = true;
            op->erase();
          }
          return;
        }
        // Handle all instances, regardless of presence in slice.

        // All input probes will be removed.
        // The processing of the instantiated module ensures these are dead.
        ImplicitLocOpBuilder builder(inst.getLoc(), inst);
        builder.setInsertionPointAfter(op);
        BitVector instPortsToErase(inst.getNumResults());
        for (auto [idx, result] : llvm::enumerate(inst.getResults())) {
          if (inst.getPortDirection(idx) == Direction::Out)
            continue;
          if (!type_isa<RefType>(result.getType()))
            continue;
          instPortsToErase.set(idx);

          // Convert to wire if not (newly) dead -- there's some local flow
          // that uses them, perhaps what was previously a u-turn/passthrough.
          if (result.use_empty())
            continue;

          auto wire = builder.create<WireOp>(result.getType());
          result.replaceAllUsesWith(wire.getDataRaw());
        }
        if (instPortsToErase.none())
          return;
        changes = true;
        auto newInst = inst.erasePorts(builder, instPortsToErase);
        // Keep InstanceGraph up-to-date if available.  Assumes safe to update
        // distinct entries from multiple threads.
        if (ig)
          ig->get().replaceInstance(inst, newInst);
        inst.erase();
      });

  numErasedPorts += portsToErase.count();
  mod.erasePorts(portsToErase);

  return changes;
}
