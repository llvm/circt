//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This pass resolves arc.inferred_context operations to their closest
// provided context. Context providers are:
// - arc.model operations
// - arc.sim.instantiate operations
// - Operations implementing FunctionOpInterface with a context-typed argument
//
// Any InferredContextOp nested under a context provider is resolved directly
// to the provided context.
// If an InferredContextOp is found in a private function that does not
// provide a context, a new context argument is added and it is recursively
// resolved at its call sites.
//
//===----------------------------------------------------------------------===//

#include "circt/Dialect/Arc/ArcOps.h"
#include "circt/Dialect/Arc/ArcPasses.h"
#include "circt/Dialect/Arc/ArcTypes.h"
#include "circt/Dialect/Arc/ModelInfo.h"
#include "circt/Support/LLVM.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/IR/Threading.h"
#include "mlir/Interfaces/CallInterfaces.h"
#include "mlir/Interfaces/FunctionInterfaces.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/WalkResult.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/Debug.h"

#include <mutex>

#define DEBUG_TYPE "arc-infer-context"

namespace circt {
namespace arc {
#define GEN_PASS_DEF_INFERCONTEXT
#include "circt/Dialect/Arc/ArcPasses.h.inc"
} // namespace arc
} // namespace circt

using namespace mlir;
using namespace circt;
using namespace arc;

namespace {

struct CallerGraphNode {
  CallerGraphNode(FunctionOpInterface &fnOp) : fnOp(fnOp) {}
  FunctionOpInterface fnOp;
  SmallSetVector<CallerGraphNode *, 4> callers;
};

struct InferContextPass : public arc::impl::InferContextBase<InferContextPass> {
  void runOnOperation() override;

private:
  /// Build the graph pointing from callees to their callers.
  void buildCallerGraph(ArrayRef<FunctionOpInterface> functions,
                        SymbolTableCollection &symbolTable);
  /// Add functions to `needsContextSet` that can reach a function that is
  /// already in the set.
  void backPropagateNeedsContext();
  /// Walk the region and wire-in the inferred context.
  void updateRegion(Region *region, SymbolTableCollection &symbolTable);

  /// Functions providing context.
  llvm::SmallDenseSet<FunctionOpInterface> hasContextSet;
  /// Functions needing a new context argument.
  llvm::SmallDenseSet<FunctionOpInterface> needsContextSet;
  /// Callee-to-caller graph; must remain const after creation to not invalidate
  /// pointers.
  DenseMap<FunctionOpInterface, CallerGraphNode> callerGraph;
};

void InferContextPass::updateRegion(Region *region,
                                    SymbolTableCollection &symbolTable) {
  assert(!region->empty());
  IRRewriter rewriter(region->getContext());
  rewriter.setInsertionPointToStart(&region->front());
  FunctionOpInterface containingFn =
      dyn_cast<FunctionOpInterface>(region->getParentOp());

  // Obtain the inferred context value.
  Value inferredContextVal = {};
  if (auto modelOp = llvm::dyn_cast<ModelOp>(region->getParentOp())) {
    // Arc model op body: Context derived from the storage argument.
    auto storageArg = region->getArgument(0);
    inferredContextVal =
        AsContextOp::create(rewriter, modelOp->getLoc(), storageArg);
    LLVM_DEBUG(auto fnName = modelOp.getSymName();
               llvm::dbgs()
               << "Updating body of model \"" << fnName << "\"\n";);
  } else if (auto instantiateOp =
                 llvm::dyn_cast<SimInstantiateOp>(region->getParentOp())) {
    // Instance op body: Context derived from the instance handle.
    auto instanceArg = region->getArgument(0);
    inferredContextVal =
        AsContextOp::create(rewriter, instantiateOp->getLoc(), instanceArg);
    LLVM_DEBUG(llvm::dbgs() << "Updating body of instance\n";);
  } else if (containingFn && (needsContextSet.contains(containingFn) ||
                              hasContextSet.contains(containingFn))) {
    // A function that has a new or pre-existing context argument.
    auto *ctxtArg = llvm::find_if(region->getArguments(), [](Value arg) {
      return isa<ContextType>(arg.getType());
    });
    assert(ctxtArg && "Expected function to have a context argument");
    inferredContextVal = *ctxtArg;
    LLVM_DEBUG(auto fnName = cast<FunctionOpInterface>(region->getParentOp())
                                 .getNameAttr()
                                 .getValue();
               llvm::dbgs()
               << "Updating body of function \"" << fnName << "\"\n";);
  } else {
    // A function that has no context argument, but may contain instances that
    // we recurse into.
    LLVM_DEBUG(auto fnName = cast<FunctionOpInterface>(region->getParentOp())
                                 .getNameAttr()
                                 .getValue();
               llvm::dbgs()
               << "Traversing body of function \"" << fnName << "\"\n";);
  }

  // Do the update walk.
  region->walk<WalkOrder::PreOrder>([&](Operation *op) -> WalkResult {
    if (op->getNumRegions() > 0) {
      // Recurse into instances
      if (auto instOp = dyn_cast<SimInstantiateOp>(op)) {
        updateRegion(&instOp.getBody(), symbolTable);
        return WalkResult::skip();
      }
      if (op->hasTrait<OpTrait::IsIsolatedFromAbove>())
        return WalkResult::skip();
    }
    // Replace InferredContextOps
    if (auto ctxtOp = dyn_cast<arc::InferredContextOp>(op)) {
      assert(inferredContextVal && "No context to propagate");
      rewriter.replaceOp(ctxtOp, inferredContextVal);
      return WalkResult::skip();
    }

    // Update calls to callees that need context, if any.
    if (needsContextSet.empty())
      return WalkResult::advance();
    auto callOp = dyn_cast<CallOpInterface>(op);
    if (!callOp)
      return WalkResult::advance();
    auto callee = dyn_cast_or_null<FunctionOpInterface>(
        callOp.resolveCallableInTable(&symbolTable));
    if (!callee || !needsContextSet.contains(callee))
      return WalkResult::advance();
    assert(inferredContextVal && "No context to propagate");
    callOp.getArgOperandsMutable().append({inferredContextVal});
    return WalkResult::advance();
  });
}

} // namespace

void InferContextPass::buildCallerGraph(ArrayRef<FunctionOpInterface> functions,
                                        SymbolTableCollection &symbolTable) {
  // Allocate the nodes for the caller graph. Nodes point to each other, so
  // we must not mutate the map afterwards.
  callerGraph.reserve(functions.size());
  for (auto fn : functions)
    if (!fn.getFunctionBody().empty())
      callerGraph.emplace_or_assign(fn, CallerGraphNode(fn));

  // Find all callees in the function bodies and add the reversed edges to the
  // graph.
  for (auto fn : functions) {
    if (fn.getFunctionBody().empty())
      continue;
    auto *caller = &callerGraph.at(fn);
    fn.getFunctionBody().walk<WalkOrder::PreOrder>(
        [&](Operation *op) -> WalkResult {
          if (op->getNumRegions() > 0) {
            // SimInstantiateOps provide a context on their own, so we ignore
            // calls nested under them.
            if (auto instOp = dyn_cast<SimInstantiateOp>(op))
              return WalkResult::skip();
            if (op->hasTrait<OpTrait::IsIsolatedFromAbove>())
              return WalkResult::skip();
          }
          auto callOp = dyn_cast<CallOpInterface>(op);
          if (!callOp)
            return WalkResult::advance();
          auto callee = llvm::dyn_cast_or_null<FunctionOpInterface>(
              callOp.resolveCallableInTable(&symbolTable));
          if (callee) {
            auto calleeIt = callerGraph.find(callee);
            if (calleeIt != callerGraph.end())
              calleeIt->second.callers.insert(caller);
          }
          return WalkResult::advance();
        });
  }
}

void InferContextPass::backPropagateNeedsContext() {

  // Propagate "needsContext" to callers.
  struct DFSFrame {
    DFSFrame(CallerGraphNode *node) : node(node) {}
    CallerGraphNode *const node;
    unsigned index = 0;
    bool isFinished() const { return index >= node->callers.size(); }
    DFSFrame getNext() {
      assert(!isFinished());
      return DFSFrame(node->callers[index++]);
    }
  };

  // Seed the DFS with the already marked functions.
  SmallVector<DFSFrame> dfsStack;
  for (auto seed : needsContextSet) {
    LLVM_DEBUG(auto fnName = seed.getNameAttr().getValue();
               llvm::dbgs()
               << "Seeding needsContext with function \"" << fnName << "\"\n";);
    auto fnNode = callerGraph.find(seed);
    assert(fnNode != callerGraph.end() && "Function not in caller graph");
    dfsStack.emplace_back(&fnNode->second);
  }

  // Mark functions reached on the inverted call graph.
  while (!dfsStack.empty()) {
    if (dfsStack.back().isFinished()) {
      dfsStack.pop_back();
      continue;
    }
    auto next = dfsStack.back().getNext();
    // Stop propagation at context providers
    if (hasContextSet.contains(next.node->fnOp))
      continue;
    if (needsContextSet.insert(next.node->fnOp).second) {
      LLVM_DEBUG(auto fnName = next.node->fnOp.getNameAttr().getValue();
                 llvm::dbgs() << "Propagating needsContext to function \""
                              << fnName << "\"\n";);
      dfsStack.push_back(next);
    }
  }
}

void InferContextPass::runOnOperation() {
  SymbolTableCollection symbolTable;
  ModuleOp moduleOp = getOperation();

  // Functions containing instances.
  llvm::SmallDenseSet<FunctionOpInterface> hasInstancesSet;

  hasContextSet.clear();
  needsContextSet.clear();
  callerGraph.clear();

  // Guards hasContextSet, needsContextSet and hasInstancesSet.
  std::mutex setMutex;

  // Collect all interesting functions. A function is interesting if it
  // contains any instances or any InferredContextOps or provides a context.
  SmallVector<FunctionOpInterface> funcOps =
      llvm::to_vector(moduleOp.getOps<FunctionOpInterface>());

  parallelForEach(
      moduleOp.getContext(), funcOps, [&](FunctionOpInterface funcOp) {
        bool needsContext = false;
        bool hasInstances = false;

        bool hasContext = llvm::any_of(funcOp.getArgumentTypes(), [](Type ty) {
          return isa<ContextType>(ty);
        });

        funcOp.getFunctionBody().walk<WalkOrder::PreOrder>(
            [&](Operation *op) -> WalkResult {
              if (op->getNumRegions() > 0) {
                if (isa<SimInstantiateOp>(op)) {
                  hasInstances = true;
                  return WalkResult::skip();
                }
                // TODO: Should we handle nested IsolatedFromAbove ops?
                if (op->hasTrait<OpTrait::IsIsolatedFromAbove>())
                  return WalkResult::skip();
              } else if (!hasContext) {
                needsContext |= isa<InferredContextOp>(op);
              }
              // Early out
              if ((hasContext || needsContext) && hasInstances)
                return WalkResult::interrupt();
              return WalkResult::advance();
            });
        assert(!(hasContext && needsContext));
        if (hasContext || needsContext || hasInstances) {
          std::lock_guard<std::mutex> lock(setMutex);
          if (hasContext)
            hasContextSet.insert(funcOp);
          if (needsContext)
            needsContextSet.insert(funcOp);
          if (hasInstances)
            hasInstancesSet.insert(funcOp);
        }
      });

  SmallPtrSet<Region *, 4> regionsToUpdate;

  if (!needsContextSet.empty()) {
    // Find functions that we have to thread the context into.
    buildCallerGraph(funcOps, symbolTable);
    backPropagateNeedsContext();

    // Now we know which functions need a context argument. Add it to their
    // signature.
    auto ctxtType = arc::ContextType::get(getOperation()->getContext());
    bool anyFailed = false;
    for (auto fn : needsContextSet) {
      if (fn.isPublic()) {
        fn.emitError("Cannot infer an Arc context through a public function. A "
                     "context argument must be provided explicitly.");
        anyFailed = true;
        continue;
      }
      if (failed(fn.insertArgument(fn.getNumArguments(), ctxtType,
                                   /*argAttrs=*/{}, fn.getLoc()))) {
        fn.emitError("Failed to add context argument to function.");
        anyFailed = true;
      }
      regionsToUpdate.insert(&fn.getFunctionBody());
    }

    if (anyFailed) {
      signalPassFailure();
      return;
    }
  } else {
    // If there are none, we only have to replace InferredContextOps in
    // the body of context providers.
    LLVM_DEBUG(llvm::dbgs() << "No function needs a context argument.\n");
  }

  // Traverse the interesting regions to replace `arc.inferred_context` ops and
  // propagate the context to callees in `needsContextSet`.
  for (auto instFn : hasInstancesSet)
    regionsToUpdate.insert(&instFn.getFunctionBody());
  for (auto fnOp : hasContextSet)
    regionsToUpdate.insert(&fnOp.getFunctionBody());
  for (auto modelOp : moduleOp.getOps<ModelOp>())
    regionsToUpdate.insert(&modelOp.getBody());

  for (Region *region : regionsToUpdate)
    updateRegion(region, symbolTable);

  markAnalysesPreserved<ModelInfoAnalysis>();
}
