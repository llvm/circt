//===- RewriteRWProbes.cpp - Rewrite rwprobes -----------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//===----------------------------------------------------------------------===//
//
// This file defines the RewriteRWProbes pass.
// This pass rewrites rwprobe's into more stable and manageable forms.
//
//===----------------------------------------------------------------------===//

#include "PassDetails.h"

#include "circt/Dialect/FIRRTL/FIRRTLUtils.h"
#include "circt/Dialect/FIRRTL/Passes.h"

#define DEBUG_TYPE "firrtl-rewrite-rwprobes"

using namespace circt;
using namespace firrtl;

//===----------------------------------------------------------------------===//
// Pass Infrastructure
//===----------------------------------------------------------------------===//

namespace {
struct RewriteRWProbesPass : public RewriteRWProbesBase<RewriteRWProbesPass> {
  RewriteRWProbesPass() = default;
  void runOnOperation() override;

  /// Get InnerSymTarget for specified ref, inserting wire if needed.
  /// Returns null target on error.
  hw::InnerSymTarget getTargetFor(FieldRef ref, OpBuilder &builder);
};
} // end anonymous namespace

void RewriteRWProbesPass::runOnOperation() {
  auto mod = getOperation();

  // Lazily create ModuleNamespace for symbols.
  ModuleNamespace ns;
  bool namespaceInitialized = false;
  auto getLazyNS = [&](auto _) -> ModuleNamespace & {
    if (!namespaceInitialized) {
      ns.add(mod);
      namespaceInitialized = true;
    }
    return ns;
  };

  auto builder = mod.getBodyBuilder();
  size_t numRWSSA = 0;
  auto result = mod.walk([&](RWProbeSSAOp rwSSA) -> WalkResult {
    // Get target for this rwSSA op, inserting wire if needed.
    auto ist = getTargetFor(getFieldRefFromValue(rwSSA.getInput()), builder);
    if (!ist)
      return failure();
    ++numRWSSA;

    // Replace with rwSym op.
    builder.setInsertionPoint(rwSSA);
    auto rwSymRef =
        builder.create<RWProbeOp>(rwSSA.getLoc(), getInnerRefTo(ist, getLazyNS),
                                  rwSSA.getInput().getType());
    rwSSA.replaceAllUsesWith(rwSymRef.getResult());
    rwSSA.erase();
    return success();
  });

  // Update statistic.
  numSSAToSym += numRWSSA;

  // Report failure.
  if (result.wasInterrupted())
    return signalPassFailure();

  // If nothing happened, record no changes.
  if (!numRWSSA)
    markAllAnalysesPreserved();
}

/// Get InnerSymTarget for specified ref, inserting wire if needed.
hw::InnerSymTarget RewriteRWProbesPass::getTargetFor(FieldRef ref,
                                                     OpBuilder &builder) {
  if (auto arg = dyn_cast<BlockArgument>(ref.getValue())) {
    return hw::InnerSymTarget(arg.getArgNumber(), ref.getDefiningOp(),
                              ref.getFieldID());
  }
  assert(isa<OpResult>(ref.getValue()));

  auto root = ref.getValue();
  auto *defOp = ref.getDefiningOp();

  // If SSA result is the target result of an op we can put a suitable
  // inner symbol on, target that.
  auto symOp = dyn_cast<hw::InnerSymbolOpInterface>(defOp);
  if (symOp && symOp.getTargetResult() == root &&
      (symOp.supportsPerFieldSymbols() || ref.getFieldID() == 0))
    return hw::InnerSymTarget(ref.getDefiningOp(), ref.getFieldID());

  // Limit to instance ops for now.  They are special.
  if (isa<InstanceOp>(defOp))
    ++numInstanceResultToWire;
  else {
    mlir::emitError(ref.getLoc()) << "unsupported rwprobe target";
    return {};
  }

  // Otherwise, insert wire and put symbol on that.
  // RAUW original to this wire, and connect them.

  // Generate helpful name for the replacement wire.
  auto [name, found] =
      getFieldName(getFieldRefFromValue(root), /*nameSafe=*/true);
  assert(found);

  // Generate wire.
  builder.setInsertionPointAfterValue(root);
  auto wire = builder.create<WireOp>(root.getLoc(), ref.getValue().getType(),
                                     name, NameKindEnum::InterestingName);

  // Move all users to the wire.
  ref.getValue().replaceAllUsesWith(wire.getResult());

  // Hook this to original decl according to flow.
  auto lhs = ref.getValue();
  auto rhs = wire.getResult();
  if (foldFlow(lhs) == Flow::Source)
    std::swap(lhs, rhs);
  emitConnect(builder, root.getLoc(), lhs, rhs);

  return hw::InnerSymTarget(wire, ref.getFieldID());
}

/// This is the pass constructor.
std::unique_ptr<mlir::Pass> circt::firrtl::createRewriteRWProbesPass() {
  return std::make_unique<RewriteRWProbesPass>();
}
