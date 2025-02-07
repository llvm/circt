//===- InlineArcs.cpp -----------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "circt/Dialect/Arc/ArcOps.h"
#include "circt/Dialect/Arc/ArcPasses.h"
#include "circt/Dialect/HW/HWOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/InliningUtils.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "arc-inline"

namespace circt {
namespace arc {
#define GEN_PASS_DEF_INLINEARCS
#include "circt/Dialect/Arc/ArcPasses.h.inc"
} // namespace arc
} // namespace circt

using namespace circt;
using namespace arc;

namespace {

/// Statistics collected during the pass. The fields should match the statistics
/// declared in the pass definition and are just bundled here to conveniently
/// pass around a reference to them.
struct InlineArcsStatistics {
  size_t numInlinedArcs = 0;
  size_t numRemovedArcs = 0;
  size_t numTrivialArcs = 0;
  size_t numSingleUseArcs = 0;
};

/// An analysis that analyses the call graph and can be queried by the inliner
/// for inlining decisions on a per-call basis. The inliner should use the
/// notification calls to keep the analysis up-to-date.
class InlineArcsAnalysis {
public:
  InlineArcsAnalysis(InlineArcsStatistics &statistics,
                     const InlineArcsOptions &options)
      : statistics(statistics), options(options) {}

  /// Clears the current analysis state and recomputes it. The first argument
  /// is a list of regions that can contain calls that should possibly be
  /// inlined. The second argument is the list of arcs to which calls in the
  /// regions of the first argument could refer to.
  void analyze(ArrayRef<Region *> regionsWithCalls,
               ArrayRef<DefineOp> arcDefinitions);

  /// Notify the analysis that a call was inlined such that it can adjust
  /// future inlining decisions accordingly.
  void notifyInlinedCallInto(mlir::CallOpInterface callOp, Region *region);

  /// Notify the analysis that an arc was removed.
  void notifyArcRemoved(DefineOp arc);

  /// Query the analysis for a decision whether the given call should be
  /// inlined, taking the pass options into account.
  bool shouldInline(mlir::CallOpInterface callOp) const;

  /// Get the arc referred to by the call from the cache.
  DefineOp getArc(mlir::CallOpInterface callOp) const;

  /// Get the number of calls within the regions passed as argument to the
  /// initial `analyze` call that refer to the given arc.
  size_t getNumArcUses(StringAttr arcName) const;

private:
  DenseMap<StringAttr, SmallVector<StringAttr>> callsInArcBody;
  DenseMap<StringAttr, size_t> numOpsInArc;
  DenseMap<StringAttr, size_t> usersPerArc;
  DenseMap<StringAttr, DefineOp> arcMap;

  InlineArcsStatistics &statistics;
  const InlineArcsOptions &options;
};

/// The actual inliner performing the transformation. Has to be given an
/// analysis that it can query for inlining decisions.
class ArcInliner {
public:
  explicit ArcInliner(InlineArcsAnalysis &analysis) : analysis(analysis) {}

  /// Inline calls in the given list of regions according to the analysis. The
  /// second argument has to contain all arcs that calls could refer to.
  /// Only pass true as the last argument if it is guarenteed that there cannot
  /// be any calls to arcs in `arcDefinitions` outside of the passed regions.
  void inlineCallsInRegion(ArrayRef<Region *> regionsWithCalls,
                           ArrayRef<DefineOp> arcDefinitions,
                           bool removeUnusedArcs = false);

  /// Remove arcs given by the second argument that that aren't used in the
  /// given region.
  void removeUnusedArcs(Region *unusedIn, ArrayRef<DefineOp> arcs);

private:
  void inlineCallsInRegion(Region *region);
  void removeUnusedArcsInternal(ArrayRef<DefineOp> arcs);

  InlineArcsAnalysis &analysis;
};

/// The inliner pass that sets up the analysis and inliner to operate on the
/// root builtin module.
struct InlineArcsPass : public arc::impl::InlineArcsBase<InlineArcsPass> {
  using InlineArcsBase::InlineArcsBase;

  void runOnOperation() override;
};

} // namespace

void ArcInliner::inlineCallsInRegion(Region *region) {
  for (auto &block : region->getBlocks()) {
    for (auto iter = block.begin(); iter != block.end(); ++iter) {
      Operation &op = *iter;
      if (auto callOp = dyn_cast<mlir::CallOpInterface>(op);
          callOp && analysis.shouldInline(callOp)) {
        DefineOp arc = analysis.getArc(callOp);
        auto args = arc.getBodyBlock().getArguments();

        IRMapping localMapping;
        for (auto [arg, operand] : llvm::zip(args, callOp.getArgOperands()))
          localMapping.map(arg, operand);

        OpBuilder builder(callOp);
        builder.setInsertionPointAfter(callOp);
        for (auto &op : arc.getBodyBlock().without_terminator())
          builder.clone(op, localMapping);

        for (auto [returnVal, result] :
             llvm::zip(arc.getBodyBlock().getTerminator()->getOperands(),
                       callOp->getResults()))
          result.replaceAllUsesWith(localMapping.lookup(returnVal));

        analysis.notifyInlinedCallInto(callOp, region);
        --iter;
        callOp->erase();
        continue;
      }

      // Note: this is a recursive call where the max depth is the max number of
      // nested regions. In Arc we don't have deep regions nestings thus this is
      // fine for now.
      for (Region &region : op.getRegions())
        inlineCallsInRegion(&region);
    }
  }
}

void ArcInliner::removeUnusedArcsInternal(ArrayRef<DefineOp> arcs) {
  for (auto arc : llvm::make_early_inc_range(arcs)) {
    if (analysis.getNumArcUses(arc.getSymNameAttr()) == 0) {
      analysis.notifyArcRemoved(arc);
      arc->erase();
    }
  }
}

void ArcInliner::removeUnusedArcs(Region *unusedIn, ArrayRef<DefineOp> arcs) {
  analysis.analyze({unusedIn}, arcs);
  removeUnusedArcsInternal(arcs);
}

void InlineArcsAnalysis::analyze(ArrayRef<Region *> regionsWithCalls,
                                 ArrayRef<DefineOp> arcDefinitions) {
  callsInArcBody.clear();
  numOpsInArc.clear();
  usersPerArc.clear();
  arcMap.clear();

  // Count the number of non-trivial ops in the arc. If there are only a few
  // (determined by the pass option), the arc will be inlined.
  for (auto arc : arcDefinitions) {
    auto arcName = arc.getSymNameAttr();
    arcMap[arcName] = arc;
    numOpsInArc[arcName] = 0;
    arc->walk([&](Operation *op) {
      if (!op->hasTrait<OpTrait::ConstantLike>() && !isa<OutputOp>(op))
        ++numOpsInArc[arcName];
      if (isa<mlir::CallOpInterface>(op))
        // TODO: make safe
        callsInArcBody[arcName].push_back(
            cast<mlir::SymbolRefAttr>(
                cast<mlir::CallOpInterface>(op).getCallableForCallee())
                .getLeafReference());
    });
    if (numOpsInArc[arcName] <= options.maxNonTrivialOpsInBody)
      ++statistics.numTrivialArcs;

    LLVM_DEBUG(llvm::dbgs() << "Arc " << arc.getSymName() << " has "
                            << numOpsInArc[arcName] << " non-trivial ops\n");

    // Make sure an entry is present such that we don't have to lookup the
    // symbol below but can just check if we already have an initialized entry
    // in this map.
    usersPerArc[arc.getSymNameAttr()] = 0;
  }

  for (auto *regionWithCalls : regionsWithCalls) {
    regionWithCalls->walk([&](mlir::CallOpInterface op) {
      if (!isa<SymbolRefAttr>(op.getCallableForCallee()))
        return;

      StringAttr arcName =
          cast<SymbolRefAttr>(op.getCallableForCallee()).getLeafReference();
      if (!usersPerArc.contains(arcName))
        return;

      ++usersPerArc[arcName];
    });
  }

  // Provide the user with some statistics on how many arcs are only ever used
  // once.
  for (auto arc : arcDefinitions)
    if (usersPerArc[arc.getSymNameAttr()] == 1)
      ++statistics.numSingleUseArcs;
}

bool InlineArcsAnalysis::shouldInline(mlir::CallOpInterface callOp) const {
  // Arcs are always referenced via symbol.
  if (!isa<SymbolRefAttr>(callOp.getCallableForCallee()))
    return false;

  if (!callOp->getParentOfType<DefineOp>() && options.intoArcsOnly)
    return false;

  // The `numOpsInArc` map contains an entry for all arcs considered. If the
  // callee symbol is not present, it is either not an arc or an arc that we
  // don't consider and thus don't want to inline.
  StringAttr arcName = llvm::cast<SymbolRefAttr>(callOp.getCallableForCallee())
                           .getLeafReference();
  if (!numOpsInArc.contains(arcName))
    return false;

  // Query the inliner interface which will always be the one of the Arc dialect
  // at this point. This is to make sure no StateOps with latency > 1 are
  // are inlined.
  auto *inlinerInterface =
      dyn_cast<mlir::DialectInlinerInterface>(callOp->getDialect());
  if (!inlinerInterface ||
      !inlinerInterface->isLegalToInline(callOp, getArc(callOp), true))
    return false;

  if (numOpsInArc.at(arcName) <= options.maxNonTrivialOpsInBody)
    return true;

  return usersPerArc.at(arcName) == 1;
}

DefineOp InlineArcsAnalysis::getArc(mlir::CallOpInterface callOp) const {
  StringAttr arcName =
      cast<SymbolRefAttr>(callOp.getCallableForCallee()).getLeafReference();
  return arcMap.at(arcName);
}

void ArcInliner::inlineCallsInRegion(ArrayRef<Region *> regionsWithCalls,
                                     ArrayRef<DefineOp> arcDefinitions,
                                     bool removeUnusedArcs) {
  analysis.analyze(regionsWithCalls, arcDefinitions);
  for (auto *regionWithCalls : regionsWithCalls)
    inlineCallsInRegion(regionWithCalls);

  if (removeUnusedArcs)
    removeUnusedArcsInternal(arcDefinitions);
}

size_t InlineArcsAnalysis::getNumArcUses(StringAttr arcName) const {
  return usersPerArc.at(arcName);
}

void InlineArcsAnalysis::notifyInlinedCallInto(mlir::CallOpInterface callOp,
                                               Region *region) {
  StringAttr calledArcName =
      cast<mlir::SymbolRefAttr>(callOp.getCallableForCallee())
          .getLeafReference();
  --usersPerArc[calledArcName];
  ++statistics.numInlinedArcs;

  auto arc = dyn_cast<DefineOp>(region->getParentOp());
  if (!arc)
    return;

  StringAttr arcName = arc.getSymNameAttr();
  // Minus one for the call op that gets removed
  numOpsInArc[arcName] += numOpsInArc[calledArcName] - 1;
  auto &calls = callsInArcBody[arcName];
  auto *iter = llvm::find(calls, calledArcName);
  if (iter != calls.end())
    calls.erase(iter);

  for (auto calleeName : callsInArcBody[calledArcName]) {
    if (!usersPerArc.contains(calleeName))
      continue;

    ++usersPerArc[calleeName];
    callsInArcBody[arcName].push_back(calleeName);
  }
}

void InlineArcsAnalysis::notifyArcRemoved(DefineOp arc) {
  for (auto calleeName : callsInArcBody[arc.getSymNameAttr()])
    --usersPerArc[calleeName];

  callsInArcBody[arc.getSymNameAttr()].clear();
  ++statistics.numRemovedArcs;
}

void InlineArcsPass::runOnOperation() {
  // This is a big ugly, TableGen should add the options as a field of this
  // struct to the pass to make passing them around easier.
  InlineArcsOptions options;
  options.intoArcsOnly = intoArcsOnly;
  options.maxNonTrivialOpsInBody = maxNonTrivialOpsInBody;
  InlineArcsStatistics statistics;
  InlineArcsAnalysis analysis(statistics, options);
  ArcInliner inliner(analysis);

  // The order of elements in `regions` determines which calls are inlined first
  // (region by region). It is thus possible to pre-compute a ordering that
  // leads to calls being inlined top-down in the call-graph, buttom-up, or
  // anything inbetween. Currently, we just use the existing order in the IR.
  SmallVector<DefineOp> arcDefinitions;
  SmallVector<Region *> regions;
  for (Operation &op : *getOperation().getBody()) {
    if (auto arc = dyn_cast<DefineOp>(&op)) {
      arcDefinitions.emplace_back(arc);
      regions.push_back(&arc.getBody());
    }
    // TODO: instead of hardcoding these ops we might also be able to query the
    // inliner interface for legality
    if (isa<hw::HWModuleOp, mlir::func::FuncOp, ModelOp>(&op))
      regions.push_back(&op.getRegion(0));
  }

  // Inline the calls and remove all arcs that don't have any uses. It doesn't
  // matter if all uses got inlined or if they already had no uses to begin
  // with.
  inliner.inlineCallsInRegion(regions, arcDefinitions,
                              /*removeUnusedArcs=*/true);

  // This is a bit ugly, but TableGen doesn't generate a container for all
  // statistics to easily pass them around unfortunately.
  numInlinedArcs = statistics.numInlinedArcs;
  numRemovedArcs = statistics.numRemovedArcs;
  numSingleUseArcs = statistics.numSingleUseArcs;
  numTrivialArcs = statistics.numTrivialArcs;
}

std::unique_ptr<Pass> arc::createInlineArcsPass() {
  return std::make_unique<InlineArcsPass>();
}
