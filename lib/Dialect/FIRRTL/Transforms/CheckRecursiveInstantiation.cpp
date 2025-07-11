//===- CheckRecursiveInstantiation.cpp - Check recurisve instantiation ----===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "circt/Dialect/FIRRTL/FIRRTLInstanceGraph.h"
#include "circt/Dialect/FIRRTL/Passes.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/SCCIterator.h"

namespace circt {
namespace firrtl {
#define GEN_PASS_DEF_CHECKRECURSIVEINSTANTIATION
#include "circt/Dialect/FIRRTL/Passes.h.inc"
} // namespace firrtl
} // namespace circt

using namespace circt;
using namespace firrtl;

static void printPath(InstanceGraph &instanceGraph,
                      ArrayRef<InstanceGraphNode *> nodes) {
  assert(nodes.size() > 0 && "an scc should have at least one node");
  auto diag =
      emitError(nodes.front()->getModule().getLoc(), "recursive instantiation");
  llvm::SmallPtrSet<InstanceGraphNode *, 8> scc(nodes.begin(), nodes.end());
  for (auto *node : nodes) {
    for (auto *record : *node) {
      auto *target = record->getTarget();
      if (!scc.contains(target))
        continue;
      auto &note = diag.attachNote(record->getInstance().getLoc());
      note << record->getParent()->getModule().getModuleName();
      note << " instantiates "
           << record->getTarget()->getModule().getModuleName() << " here";
    }
  }
}

namespace {
class CheckRecursiveInstantiationPass
    : public impl::CheckRecursiveInstantiationBase<
          CheckRecursiveInstantiationPass> {
public:
  void runOnOperation() override {
    auto &instanceGraph = getAnalysis<InstanceGraph>();
    for (auto it = llvm::scc_begin(&instanceGraph),
              end = llvm::scc_end(&instanceGraph);
         it != end; ++it) {
      if (it.hasCycle()) {
        printPath(instanceGraph, *it);
        signalPassFailure();
      }
    }
    markAllAnalysesPreserved();
  }
};
} // namespace
