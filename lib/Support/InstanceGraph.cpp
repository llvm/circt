//===- InstanceGraph.cpp - Instance Graph -----------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "circt/Support/InstanceGraph.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Threading.h"
#include "llvm/Support/ThreadPool.h"
#include <atomic>
#include <functional>

using namespace circt;
using namespace igraph;

//===----------------------------------------------------------------------===//
// Instance Node
//===----------------------------------------------------------------------===//

void InstanceRecord::erase() {
  // Update the prev node to point to the next node.
  if (prevUse)
    prevUse->nextUse = nextUse;
  else
    target->firstUse = nextUse;
  // Update the next node to point to the prev node.
  if (nextUse)
    nextUse->prevUse = prevUse;
  getParent()->instances.erase(this);
}

//===----------------------------------------------------------------------===//
// Module Node
//===----------------------------------------------------------------------===//

InstanceRecord *InstanceGraphNode::addInstance(InstanceOpInterface instance,
                                               InstanceGraphNode *target) {
  auto *instanceRecord = new InstanceRecord(this, instance, target);
  target->recordUse(instanceRecord);
  instances.push_back(instanceRecord);
  return instanceRecord;
}

void InstanceGraphNode::recordUse(InstanceRecord *record) {
  record->nextUse = firstUse;
  if (firstUse)
    firstUse->prevUse = record;
  firstUse = record;
}

//===----------------------------------------------------------------------===//
// Instance Graph
//===----------------------------------------------------------------------===//

InstanceGraphNode *InstanceGraph::getOrAddNode(StringAttr name) {
  // Try to insert an InstanceGraphNode. If its not inserted, it returns
  // an iterator pointing to the node.
  auto *&node = nodeMap[name];
  if (!node) {
    node = new InstanceGraphNode();
    nodes.push_back(node);
  }
  return node;
}

InstanceGraph::InstanceGraph(Operation *parent) : parent(parent) {
  assert(parent->hasTrait<mlir::OpTrait::SingleBlock>() &&
         "top-level operation must have a single block");
  SmallVector<std::pair<ModuleOpInterface, SmallVector<InstanceOpInterface>>>
      moduleToInstances;
  // First accumulate modules inside the parent op.
  for (auto module :
       parent->getRegion(0).front().getOps<igraph::ModuleOpInterface>())
    moduleToInstances.push_back({module, {}});

  // Populate instances in the module parallelly.
  mlir::parallelFor(parent->getContext(), 0, moduleToInstances.size(),
                    [&](size_t idx) {
                      auto module = moduleToInstances[idx].first;
                      auto &instances = moduleToInstances[idx].second;
                      // Find all instance operations in the module body.
                      module.walk([&](InstanceOpInterface instanceOp) {
                        instances.push_back(instanceOp);
                      });
                    });

  // Construct an instance graph sequentially.
  for (auto &[module, instances] : moduleToInstances) {
    auto name = module.getModuleNameAttr();
    auto *currentNode = getOrAddNode(name);
    currentNode->module = module;
    for (auto instanceOp : instances) {
      // Add an edge to indicate that this module instantiates the target.
      for (auto targetNameAttr : instanceOp.getReferencedModuleNamesAttr()) {
        auto *targetNode = getOrAddNode(cast<StringAttr>(targetNameAttr));
        currentNode->addInstance(instanceOp, targetNode);
      }
    }
  }
}

InstanceGraphNode *InstanceGraph::addModule(ModuleOpInterface module) {
  assert(!nodeMap.count(module.getModuleNameAttr()) && "module already added");
  auto *node = new InstanceGraphNode();
  node->module = module;
  nodeMap[module.getModuleNameAttr()] = node;
  nodes.push_back(node);
  return node;
}

void InstanceGraph::erase(InstanceGraphNode *node) {
  assert(node->noUses() &&
         "all instances of this module must have been erased.");
  // Erase all instances inside this module.
  for (auto *instance : llvm::make_early_inc_range(*node))
    instance->erase();
  nodeMap.erase(node->getModule().getModuleNameAttr());
  nodes.erase(node);
}

InstanceGraphNode *InstanceGraph::lookupOrNull(StringAttr name) {
  auto it = nodeMap.find(name);
  if (it == nodeMap.end())
    return nullptr;
  return it->second;
}

void InstanceGraph::replaceInstance(InstanceOpInterface inst,
                                    InstanceOpInterface newInst) {
  assert(inst.getReferencedModuleNamesAttr() ==
             newInst.getReferencedModuleNamesAttr() &&
         "Both instances must be targeting the same modules");

  // Replace all edges between the module of the instance and all targets.
  for (Attribute targetNameAttr : inst.getReferencedModuleNamesAttr()) {
    // Find the instance record of this instance.
    auto *node = lookup(cast<StringAttr>(targetNameAttr));
    for (InstanceRecord *record : node->uses()) {
      if (record->getInstance() == inst) {
        // We can just replace the instance op in the InstanceRecord without
        // updating any instance lists.
        record->instance = newInst;
      }
    }
  }
}

bool InstanceGraph::isAncestor(
    ModuleOpInterface child, ModuleOpInterface parent,
    llvm::function_ref<bool(InstanceRecord *)> skipInstance) {
  DenseSet<InstanceGraphNode *> seen;
  SmallVector<InstanceGraphNode *> worklist;
  auto *cn = lookup(child);
  worklist.push_back(cn);
  seen.insert(cn);
  while (!worklist.empty()) {
    auto *node = worklist.back();
    worklist.pop_back();
    if (node->getModule() == parent)
      return true;
    for (auto *use : node->uses()) {
      if (skipInstance(use))
        continue;
      auto *mod = use->getParent();
      if (!seen.count(mod)) {
        seen.insert(mod);
        worklist.push_back(mod);
      }
    }
  }
  return false;
}

FailureOr<llvm::ArrayRef<InstanceGraphNode *>>
InstanceGraph::getInferredTopLevelNodes() {
  if (!inferredTopLevelNodes.empty())
    return {inferredTopLevelNodes};

  /// Topologically sort the instance graph.
  llvm::SetVector<InstanceGraphNode *> visited, marked;
  llvm::SetVector<InstanceGraphNode *> candidateTopLevels(this->begin(),
                                                          this->end());
  SmallVector<InstanceGraphNode *> cycleTrace;

  // Recursion function; returns true if a cycle was detected.
  std::function<bool(InstanceGraphNode *, SmallVector<InstanceGraphNode *>)>
      cycleUtil =
          [&](InstanceGraphNode *node, SmallVector<InstanceGraphNode *> trace) {
            if (visited.contains(node))
              return false;
            trace.push_back(node);
            if (marked.contains(node)) {
              // Cycle detected.
              cycleTrace = trace;
              return true;
            }
            marked.insert(node);
            for (auto *use : *node) {
              InstanceGraphNode *targetModule = use->getTarget();
              candidateTopLevels.remove(targetModule);
              if (cycleUtil(targetModule, trace))
                return true; // Cycle detected.
            }
            marked.remove(node);
            visited.insert(node);
            return false;
          };

  bool cyclic = false;
  for (auto *moduleIt : *this) {
    if (visited.contains(moduleIt))
      continue;

    cyclic |= cycleUtil(moduleIt, {});
    if (cyclic)
      break;
  }

  if (cyclic) {
    auto err = getParent()->emitOpError();
    err << "cannot deduce top level module - cycle "
           "detected in instance graph (";
    llvm::interleave(
        cycleTrace, err,
        [&](auto node) { err << node->getModule().getModuleName(); }, "->");
    err << ").";
    return err;
  }
  assert((visited.empty() || !candidateTopLevels.empty()) &&
         "if non-cyclic, there should be at least 1 candidate top level");

  inferredTopLevelNodes = llvm::SmallVector<InstanceGraphNode *>(
      candidateTopLevels.begin(), candidateTopLevels.end());
  return {inferredTopLevelNodes};
}

//===----------------------------------------------------------------------===//
// Instance Paths
//===----------------------------------------------------------------------===//

static InstancePath empty{};

ArrayRef<InstancePath>
InstancePathCache::getAbsolutePaths(ModuleOpInterface op) {
  return getPaths(op, instanceGraph.getTopLevelNode(), absolutePathsCache);
}

ArrayRef<InstancePath>
InstancePathCache::getRelativePaths(ModuleOpInterface op,
                                    InstanceGraphNode *node) {
  if (node == instanceGraph.getTopLevelNode())
    return getAbsolutePaths(op);
  return getPaths(op, node, relativePathsCache[node]);
}

// NOLINTBEGIN(misc-no-recursion)
ArrayRef<InstancePath> InstancePathCache::getPaths(ModuleOpInterface op,
                                                   InstanceGraphNode *top,
                                                   PathsCache &cache) {
  InstanceGraphNode *node = instanceGraph[op];

  if (node == top) {
    return empty;
  }

  // Fast path: hit the cache.
  auto cached = cache.find(op);
  if (cached != cache.end())
    return cached->second;

  // For each instance, collect the instance paths to its parent and append the
  // instance itself to each.
  SmallVector<InstancePath, 8> extendedPaths;
  for (auto *inst : node->uses()) {
    if (auto module = inst->getParent()->getModule()) {
      auto instPaths = getPaths(module, top, cache);
      extendedPaths.reserve(instPaths.size());
      for (auto path : instPaths) {
        extendedPaths.push_back(appendInstance(
            path, cast<InstanceOpInterface>(*inst->getInstance())));
      }
    } else if (inst->getParent() == top) {
      // Special case when `inst` is a top-level instance and
      // `inst->getParent()` is a pseudo top-level node.
      extendedPaths.emplace_back(empty);
    }
  }

  // Move the list of paths into the bump allocator for later quick retrieval.
  ArrayRef<InstancePath> pathList;
  if (!extendedPaths.empty()) {
    auto *paths = allocator.Allocate<InstancePath>(extendedPaths.size());
    std::copy(extendedPaths.begin(), extendedPaths.end(), paths);
    pathList = ArrayRef<InstancePath>(paths, extendedPaths.size());
  }
  cache.insert({op, pathList});
  return pathList;
}
// NOLINTEND(misc-no-recursion)

void InstancePath::print(llvm::raw_ostream &into) const {
  into << "$root";
  for (unsigned i = 0, n = path.size(); i < n; ++i) {
    auto inst = path[i];

    into << "/" << inst.getInstanceName() << ":";
    auto names = inst.getReferencedModuleNamesAttr();
    if (names.size() == 1) {
      // If there is a unique target, print it.
      into << cast<StringAttr>(names[0]).getValue();
    } else {
      if (i + 1 < n) {
        // If this is not a leaf node, the target module should be the
        // parent of the next instance operation in the path.
        into << path[i + 1]
                    ->getParentOfType<ModuleOpInterface>()
                    .getModuleName();
      } else {
        // Otherwise, print the whole set of targets.
        into << "{";
        llvm::interleaveComma(names, into, [&](Attribute name) {
          into << cast<StringAttr>(name).getValue();
        });
        into << "}";
      }
    }
  }
}

InstancePath InstancePathCache::appendInstance(InstancePath path,
                                               InstanceOpInterface inst) {
  size_t n = path.size() + 1;
  auto *newPath = allocator.Allocate<InstanceOpInterface>(n);
  std::copy(path.begin(), path.end(), newPath);
  newPath[path.size()] = inst;
  return InstancePath(ArrayRef(newPath, n));
}

InstancePath InstancePathCache::prependInstance(InstanceOpInterface inst,
                                                InstancePath path) {
  size_t n = path.size() + 1;
  auto *newPath = allocator.Allocate<InstanceOpInterface>(n);
  std::copy(path.begin(), path.end(), newPath + 1);
  newPath[0] = inst;
  return InstancePath(ArrayRef(newPath, n));
}

InstancePath InstancePathCache::concatPath(InstancePath path1,
                                           InstancePath path2) {
  size_t n = path1.size() + path2.size();
  auto *newPath = allocator.Allocate<InstanceOpInterface>(n);
  std::copy(path1.begin(), path1.end(), newPath);
  std::copy(path2.begin(), path2.end(), newPath + path1.size());
  return InstancePath(ArrayRef(newPath, n));
}

void InstancePathCache::replaceInstance(InstanceOpInterface oldOp,
                                        InstanceOpInterface newOp) {

  instanceGraph.replaceInstance(oldOp, newOp);

  // Iterate over all the paths, and search for the old InstanceOpInterface. If
  // found, then replace it with the new InstanceOpInterface, and create a new
  // copy of the paths and update the cache.
  auto instanceExists = [&](const ArrayRef<InstancePath> &paths) -> bool {
    return llvm::any_of(
        paths, [&](InstancePath p) { return llvm::is_contained(p, oldOp); });
  };
  auto updateCache = [&](PathsCache &cache) {
    for (auto &iter : cache) {
      if (!instanceExists(iter.getSecond()))
        continue;
      SmallVector<InstancePath, 8> updatedPaths;
      for (auto path : iter.getSecond()) {
        const auto *iter = llvm::find(path, oldOp);
        if (iter == path.end()) {
          // path does not contain the oldOp, just copy it as is.
          updatedPaths.push_back(path);
          continue;
        }
        auto *newPath = allocator.Allocate<InstanceOpInterface>(path.size());
        llvm::copy(path, newPath);
        newPath[iter - path.begin()] = newOp;
        updatedPaths.push_back(InstancePath(ArrayRef(newPath, path.size())));
      }
      // Move the list of paths into the bump allocator for later quick
      // retrieval.
      auto *paths = allocator.Allocate<InstancePath>(updatedPaths.size());
      llvm::copy(updatedPaths, paths);
      iter.getSecond() = ArrayRef<InstancePath>(paths, updatedPaths.size());
    }
  };
  updateCache(absolutePathsCache);
  for (auto &iter : relativePathsCache)
    updateCache(iter.getSecond());
}

//===----------------------------------------------------------------------===//
// Parallel walk implementation
//===----------------------------------------------------------------------===//

LogicalResult InstanceGraph::walkParallelImpl(
    llvm::function_ref<LogicalResult(InstanceGraphNode &)> fn, bool forward) {
  // Direction-dependent neighbor functions.  When `forward` is true the walk
  // is post-order: children are prereqs, parents are notifyees.  When false
  // the walk is inverse post-order: parents are prereqs, children are
  // notifyees.
  auto appendChildren = [](InstanceGraphNode &node,
                           SmallVectorImpl<InstanceGraphNode *> &out) {
    for (auto *record : node)
      out.push_back(record->getTarget());
  };
  auto appendParents = [](InstanceGraphNode &node,
                          SmallVectorImpl<InstanceGraphNode *> &out) {
    for (auto *use : node.uses())
      out.push_back(use->getParent());
  };
  auto prereqs = [&](InstanceGraphNode &node,
                     SmallVectorImpl<InstanceGraphNode *> &out) {
    if (forward)
      appendChildren(node, out);
    else
      appendParents(node, out);
  };
  auto notify = [&](InstanceGraphNode &node,
                    SmallVectorImpl<InstanceGraphNode *> &out) {
    if (forward)
      appendParents(node, out);
    else
      appendChildren(node, out);
  };

  // If multithreading is disabled fall back to the sequential walk so that
  // callers don't have to reason about two code paths.
  MLIRContext *ctx = parent->getContext();
  if (!ctx->isMultithreadingEnabled()) {
    if (forward)
      return walkPostOrder(fn);
    // walkInversePostOrder may visit sentinel nodes reachable via graph
    // traversal (e.g., the virtual top-level entry in hw::InstanceGraph).
    // Filter to managed nodes so both paths visit the same set.
    DenseSet<InstanceGraphNode *> managed;
    for (auto &node : nodes)
      managed.insert(&node);
    return walkInversePostOrder([&](InstanceGraphNode &node) -> LogicalResult {
      if (!managed.count(&node))
        return success();
      return fn(node);
    });
  }

  // --- Phase 1: compute per-node pending counts (sequential) --------------
  //
  // `prereqs(node, out)` fills `out` with the nodes that must finish before
  // `node` becomes ready.  The pending count of a node is the number of
  // distinct managed prereqs it has.  A node whose count is zero is a seed
  // for the traversal.
  //
  // `notify(node, out)` fills `out` with the nodes to notify when `node`
  // finishes.
  //
  // Filtering to managed nodes (via nodeIndex) means that sentinel nodes
  // introduced by subclasses (e.g., the virtual top-level entry in
  // hw::InstanceGraph) are automatically excluded from both.
  //
  // std::atomic is not movable, so we store atomics in a flat vector indexed
  // by a stable integer assigned to each node.
  DenseMap<InstanceGraphNode *, size_t> nodeIndex;
  size_t numNodes = 0;
  for (auto &node : nodes) {
    (void)node;
    ++numNodes;
  }
  nodeIndex.reserve(numNodes);

  // Allocate one atomic<unsigned> per node (value initialised to 0).
  std::vector<std::atomic<unsigned>> pending(numNodes);
  // Collect the initial seeds (nodes with zero managed prereqs) while
  // computing the pending counts.  Seeds must be gathered before submitting
  // any tasks.  Once a task is running it may decrement another node's
  // pending count to zero concurrently with this loop.  That node is
  // submitted by the decrementing thread and must not be resubmitted here.
  SmallVector<InstanceGraphNode *, 8> seeds;
  {
    // First assign indices so nodeIndex is populated for the filtering below.
    size_t idx = 0;
    for (auto &node : nodes)
      nodeIndex[&node] = idx++;

    SmallVector<InstanceGraphNode *, 4> buf;
    for (auto &node : nodes) {
      buf.clear();
      prereqs(node, buf);
      // Count distinct managed prereqs.
      SmallPtrSet<InstanceGraphNode *, 4> seen;
      unsigned count = 0;
      for (auto *nb : buf)
        if (nodeIndex.count(nb) && seen.insert(nb).second)
          ++count;
      size_t nodeIdx = nodeIndex[&node];
      pending[nodeIdx].store(count, std::memory_order_relaxed);
      if (count == 0)
        seeds.push_back(&node);
    }
  }

  // --- Phase 2: parallel traversal via thread pool ------------------------

  // All three accesses to processingFailed use relaxed ordering.  The flag
  // carries no associated payload.  It is a pure boolean signal.  The
  // happens-before edge established by taskGroup.wait() provides the
  // synchronization needed for the final read.
  std::atomic<bool> processingFailed(false);

  // visitedCount tracks how many nodes were actually executed.  After
  // taskGroup.wait() it must equal numNodes for a cycle-free DAG.
  std::atomic<size_t> visitedCount(0);

  llvm::ThreadPoolInterface &threadPool = ctx->getThreadPool();
  llvm::ThreadPoolTaskGroup taskGroup(threadPool);

  // Submit a single node for execution on the thread pool.  After the
  // callback completes, decrement the pending count of each unique managed
  // notifyee.  Those whose count drops to zero are immediately resubmitted.
  //
  // `submitNode` calls itself transitively (on different threads) to schedule
  // newly ready nodes, so we declare it as a std::function to allow the
  // lambda to capture itself by reference.  taskGroup.async() only enqueues
  // the task.  It does not execute it on the current stack, so there is no
  // actual call-stack recursion.
  //
  // NOTE: taskGroup.wait() below is load-bearing for the lifetime of
  // `submitNode`.  Worker threads capture `submitNode` by reference.  The
  // outer stack frame must not be destroyed until wait() returns.
  std::function<void(InstanceGraphNode *)> submitNode;
  submitNode = [&](InstanceGraphNode *node) {
    taskGroup.async([&, node] {
      // Skip remaining work after a failure.
      if (processingFailed.load(std::memory_order_relaxed))
        return;

      if (failed(fn(*node))) {
        processingFailed.store(true, std::memory_order_relaxed);
        return;
      }

      visitedCount.fetch_add(1, std::memory_order_relaxed);

      // Notify each unique managed notifyee that one of their prereqs is
      // done.  We use relaxed ordering on the fetch_sub because the user
      // callback is responsible for its own synchronization and we only need
      // the counter itself to be atomically decremented.  The overall
      // happens-before relationship across tasks is guaranteed by the thread
      // pool.
      SmallVector<InstanceGraphNode *, 4> toNotify;
      notify(*node, toNotify);
      SmallPtrSet<InstanceGraphNode *, 4> notified;
      for (auto *nb : toNotify) {
        auto it = nodeIndex.find(nb);
        if (it == nodeIndex.end())
          continue;
        if (!notified.insert(nb).second)
          continue;
        // If the notifyee's pending count reaches zero it is now ready.
        if (pending[it->second].fetch_sub(1, std::memory_order_relaxed) == 1)
          submitNode(nb);
      }
    });
  };

  // Seed the traversal by submitting all nodes whose initial pending count
  // was zero.  We use the seeds collected during Phase 1 rather than
  // re-reading `pending` here because worker threads started by earlier
  // `submitNode` calls may already be decrementing entries in `pending` to
  // zero.  Those nodes are submitted by the decrementing thread and must not
  // be submitted again from this loop.
  for (auto *node : seeds)
    submitNode(node);

  taskGroup.wait();

  // If every node was visited the graph is a DAG.  Otherwise there is a
  // cycle.  Only check this when no failure was reported because a failure
  // stops scheduling and naturally leaves nodes unvisited.
  assert((processingFailed.load(std::memory_order_relaxed) ||
          visitedCount.load(std::memory_order_relaxed) == numNodes) &&
         "InstanceGraph has a cycle. "
         "walkParallelPostOrder and walkParallelInversePostOrder require a "
         "DAG");

  return failure(processingFailed.load(std::memory_order_relaxed));
}

//===----------------------------------------------------------------------===//
// Generated
//===----------------------------------------------------------------------===//

#include "circt/Support/InstanceGraphInterface.cpp.inc"
