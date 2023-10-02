//===- CheckCombLoops.cpp - FIRRTL check combinational cycles ------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements the FIRRTL combinational cycles detection pass. The
// algorithm handles aggregates and sub-index/field/access ops.
// 1. Traverse each module in the Instance Graph bottom up.
// 2. Preprocess step: Construct the reaching definitions directed graph.
// Perform a dataflow analysis, on the domain of FieldRefs. Interpret each
// operation, to record the corresponding reaching def. Each node in the graph
// is a fieldRef. Each edge represents a dataflow from the source to the sink.
// 3. Perform a DFS traversal on the graph, to detect combinational loops and
// paths between ports.
// 4. Inline the combinational paths discovered in each module to its instance
// and continue the analysis through the instance graph.
//===----------------------------------------------------------------------===//

#include "PassDetails.h"
#include "circt/Dialect/FIRRTL/FIRRTLInstanceGraph.h"
#include "circt/Dialect/FIRRTL/FIRRTLUtils.h"
#include "circt/Dialect/FIRRTL/FIRRTLVisitors.h"
#include "circt/Dialect/FIRRTL/Passes.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/EquivalenceClasses.h"
#include "llvm/ADT/PostOrderIterator.h"

#define DEBUG_TYPE "check-comb-loops"

using namespace circt;
using namespace firrtl;

using ReachingDefsMapType = DenseMap<FieldRef, DenseSet<FieldRef>>;

class DiscoverLoops {

  /// Adjacency list representation.
  /// Each entry is a pair, the first element is the FieldRef corresponding to
  /// the graph vertex. The second element is the list of vertices, that have an
  /// edge to this vertex.
  /// The directed edges represent a reaching definition from the source to the
  /// sink.
  using ReachingDefgraphType =
      SmallVector<std::pair<FieldRef, SmallVector<unsigned>>, 64>;

public:
  DiscoverLoops(
      FModuleOp module, InstanceGraph &instanceGraph,
      const DenseMap<FModuleLike, ReachingDefsMapType> &otherModulePortPaths,
      ReachingDefsMapType &thisModulePortPaths)
      : module(module), instanceGraph(instanceGraph),
        modulePortPaths(otherModulePortPaths), portPaths(thisModulePortPaths) {}

  LogicalResult processModule() {
    LLVM_DEBUG(llvm::dbgs() << "\n processing module :" << module.getName());
    constructReachingDefGraph(module);
    return dfsTraverse(reachingDefGraph);
  }

  void constructReachingDefGraph(FModuleOp module) {
    LLVM_DEBUG(llvm::dbgs() << "\n Module :" << module.getName());

    // ALl the module output ports must be added as the initial nodes to the
    // reaching def graph.
    for (auto port : module.getArguments()) {
      if (module.getPortDirection(port.getArgNumber()) != Direction::Out)
        continue;
      walkGroundTypes(port.getType().cast<FIRRTLType>(),
                      [&](uint64_t index, FIRRTLBaseType t) {
                        getOrAddNode(FieldRef(port, index));
                      });
    }
    walk(module, [&](Operation *op) {
      llvm::TypeSwitch<Operation *>(op)
          .Case<RegOp, RegResetOp>([&](auto) {})
          .Case<Forceable>([&](Forceable forceableOp) {
            // Any declaration that can be forced.
            if (auto node = dyn_cast<NodeOp>(op))
              recordDataflow(node.getData(), node.getInput());
            if (!forceableOp.isForceable() ||
                forceableOp.getDataRef().use_empty())
              return;
            auto data = forceableOp.getData();
            auto ref = forceableOp.getDataRef();
            // Record dataflow from data to the probe.
            recordDataflow(ref, data);
            recordProbe(data, ref);
          })
          .Case<MemOp>([&](MemOp mem) { handleMemory(mem); })
          .Case<RefSendOp>([&](RefSendOp send) {
            recordDataflow(send.getResult(), send.getBase());
          })
          .Case<RefDefineOp>([&](RefDefineOp def) {
            // Dataflow from src to dst.
            recordDataflow(def.getDest(), def.getSrc());
            if (!def.getDest().getType().getForceable())
              return;
            // Dataflow from dst to src, for RWProbe.
            probesReferToSameData(def.getSrc(), def.getDest());
          })
          .Case<RefForceOp, RefForceInitialOp>(
              [&](auto ref) { handleRefForce(ref.getDest(), ref.getSrc()); })
          .Case<InstanceOp>([&](auto inst) { handleInstanceOp(inst); })
          .Case<SubindexOp>([&](SubindexOp sub) {
            recordValueRefersToFieldRef(
                sub.getInput(),
                sub.getInput().getType().get().getFieldID(sub.getIndex()),
                sub.getResult());
          })
          .Case<SubfieldOp>([&](SubfieldOp sub) {
            recordValueRefersToFieldRef(
                sub.getInput(),
                sub.getInput().getType().get().getFieldID(sub.getFieldIndex()),
                sub.getResult());
          })
          .Case<SubaccessOp>([&](SubaccessOp sub) {
            auto vecType = sub.getInput().getType().get();
            auto input = sub.getInput();
            auto result = sub.getResult();
            // Flatten the subaccess. The result can refer to any of the
            // elements.
            for (size_t index = 0; index < vecType.getNumElements(); ++index)
              recordValueRefersToFieldRef(input, vecType.getFieldID(index),
                                          result);
          })
          .Case<RefSubOp>([&](RefSubOp sub) {
            size_t fieldID = TypeSwitch<FIRRTLBaseType, size_t>(
                                 sub.getInput().getType().getType())
                                 .Case<FVectorType, BundleType>([&](auto type) {
                                   return type.getFieldID(sub.getIndex());
                                 });
            recordValueRefersToFieldRef(sub.getInput(), fieldID,
                                        sub.getResult());
          })
          .Case<BundleCreateOp, VectorCreateOp>([&](auto op) {
            auto type = op.getType();
            auto res = op.getResult();
            auto getFieldId = [&](unsigned index) {
              size_t fieldID =
                  TypeSwitch<FIRRTLBaseType, size_t>(type)
                      .Case<FVectorType, BundleType>(
                          [&](auto type) { return type.getFieldID(index); });
              return fieldID;
            };
            for (auto [index, v] : llvm::enumerate(op.getOperands()))
              recordValueRefersToFieldRef(res, getFieldId(index), v);
          })
          .Case<FConnectLike>([&](FConnectLike connect) {
            recordDataflow(connect.getDest(), connect.getSrc());
          })
          .Default([&](Operation *op) {
            // All other expressions.
            if (op->getNumResults() == 1) {
              auto res = op->getResult(0);
              for (auto src : op->getOperands())
                recordDataflow(res, src);
            }
          });
    });
  }

  static std::string getName(FieldRef v) { return getFieldName(v).first; };

  unsigned getOrAddNode(Value v) {
    auto iter = valToFieldRefs.find(v);
    if (iter == valToFieldRefs.end())
      return getOrAddNode({v, 0});
    return getOrAddNode(*iter->second.begin());
  }

  // Get the node id if it exists, else add it to the graph.
  unsigned getOrAddNode(FieldRef f) {
    auto iter = nodes.find(f);
    if (iter != nodes.end())
      return iter->second;
    // Add the fieldRef to the graph.
    auto id = reachingDefGraph.size();
    // The node id can be used to index into the graph. The entry is a pair,
    // first element is the corresponding FieldRef, and the second entry is a
    // list of adjacent nodes.
    reachingDefGraph.push_back({f, {}});
    nodes[f] = id;
    return id;
  }

  // Construct the reaching def graph, by adding `dst` and `src` as new nodes,
  // if not already existing. Then add an edge from `src` to `dst`.
  void addReachingDef(FieldRef dst, FieldRef src) {
    auto srcNode = getOrAddNode(src);
    auto dstNode = getOrAddNode(dst);
    reachingDefGraph[dstNode].second.push_back(srcNode);
  }

  // Add `srcVal` as a reaching def to `dstVal`.
  void recordDataflow(Value dstVal, Value srcVal) {
    // Ignore reaching def from constants.
    if (auto *def = srcVal.getDefiningOp())
      if (def->hasTrait<OpTrait::ConstantLike>())
        return;
    // Check if srcVal/dstVal is a fieldRef to an aggregate. Then, there may
    // exist other values, that refer to the same fieldRef. Add a reaching def
    // from all such "aliasing" values.
    auto dstIt = valToFieldRefs.find(dstVal);
    auto srcIt = valToFieldRefs.find(srcVal);

    // Block the dataflow through registers.
    // dstVal refers to a register, If dstVal is not recorded as the fieldref of
    // an aggregate, and its either a register, or result of a sub op.
    if (dstIt == valToFieldRefs.end())
      if (auto *def = dstVal.getDefiningOp())
        if (isa<RegOp, RegResetOp, SubaccessOp, SubfieldOp, SubindexOp,
                RefSubOp>(def))
          return;

    auto dstValType = getBaseType(dstVal.getType());
    auto srcValType = getBaseType(srcVal.getType());

    // Handle Properties.
    if (!(srcValType && dstValType))
      return addReachingDef({dstVal, 0}, {srcVal, 0});

    auto addDef = [&](FieldRef dst, FieldRef src) {
      // If the dstVal and srcVal are aggregate types, then record the dataflow
      // between each individual ground type. This is equivalent to flattening
      // the type to ensure all the contained FieldRefs are also recorded.
      if (dstValType && !dstValType.isGround())
        walkGroundTypes(dstValType, [&](uint64_t dstIndex, FIRRTLBaseType t) {
          // Handle the case when the dst and src are not of the same type.
          // For each dst ground type, and for each src ground type.
          if (srcValType == dstValType)
            addReachingDef(dst.getSubField(dstIndex),
                           src.getSubField(dstIndex));
          else if (srcValType && !srcValType.isGround())
            walkGroundTypes(srcValType,
                            [&](uint64_t srcIndex, FIRRTLBaseType t) {
                              addReachingDef(dst.getSubField(dstIndex),
                                             src.getSubField(srcIndex));
                            });
          // Else, the src is ground type.
          else
            addReachingDef(dst.getSubField(dstIndex), src);
        });

      addReachingDef(dst, src);
    };

    // Both the dstVal and srcVal, can refer to multiple FieldRefs, ensure that
    // we capture the dataflow between each pair. Occurs when val result of
    // subaccess op.

    // Case 1: None of src and dst are fields of an aggregate. But they can be
    // aggregate values.
    if (dstIt == valToFieldRefs.end() && srcIt == valToFieldRefs.end())
      return addDef({dstVal, 0}, {srcVal, 0});
    // Case 2: Only src is the field of an aggregate. Get all the fields that
    // the src refers to.
    if (dstIt == valToFieldRefs.end() && srcIt != valToFieldRefs.end()) {
      llvm::for_each(srcIt->getSecond(), [&](const FieldRef &srcField) {
        addDef(FieldRef(dstVal, 0), srcField);
      });
      return;
    }
    // Case 3: Only dst is the field of an aggregate. Get all the fields that
    // the dst refers to.
    if (dstIt != valToFieldRefs.end() && srcIt == valToFieldRefs.end()) {
      llvm::for_each(dstIt->getSecond(), [&](const FieldRef &dstField) {
        addDef(dstField, FieldRef(srcVal, 0));
      });
      return;
    }
    // Case 4: Both src and dst are the fields of an aggregate. Get all the
    // fields that they refers to.
    llvm::for_each(srcIt->second, [&](const FieldRef &srcField) {
      llvm::for_each(dstIt->second, [&](const FieldRef &dstField) {
        addDef(dstField, srcField);
      });
    });
  }

  // Record srcVal as a reaching def to the original data value that the probe
  // refers to.
  void handleRefForce(Value dstRef, Value srcVal) {
    recordDataflow(dstRef, srcVal);
    auto dstNode = getOrAddNode(dstRef);
    // Now add srcVal as a reaching def to the data that dstRef refers to.
    auto leader = rwProbeClasses.findLeader(dstNode);
    if (leader == rwProbeClasses.member_end())
      return;
    auto iter = rwProbeRefersTo.find(*leader);
    assert(iter != rwProbeRefersTo.end());
    if (iter->second != dstNode)
      reachingDefGraph[iter->second].second.push_back(getOrAddNode(srcVal));
  }

  // Check the referenced module paths and add input ports as the reaching defs
  // for the corresponding output port. The granularity of the reaching
  // definitions is per field.
  void handleInstanceOp(InstanceOp inst) {
    auto refMod =
        dyn_cast_or_null<FModuleOp>(*instanceGraph.getReferencedModule(inst));
    // TODO: External modules not handled !!
    if (!refMod)
      return;
    auto modulePaths = modulePortPaths.find(refMod);
    if (modulePaths == modulePortPaths.end())
      return;
    // Note: Handling RWProbes.
    // 1. For RWProbes, output ports can be source of dataflow.
    // 2. All the RWProbes that refer to the same base value form a strongly
    // connected component. Each has a reaching def from the other, including
    // itself.
    // Steps to add the instance ports to the reaching def graph::
    // 1. Find the set of RWProbes that refer to the same base value.
    // 2. Add them to the same rwProbeClasses.
    // 3. Chose the first RWProbe port from this set as a representative base
    //    value. And add it as the source reaching def for every other RWProbe
    //    port in the set.
    // 4. This will ensure we can detect cycles involving different RWProbes to
    //    the same base value.
    for (auto &path : modulePaths->second) {
      auto modSinkPortField = path.first;
      auto sinkArgNum =
          cast<BlockArgument>(modSinkPortField.getValue()).getArgNumber();
      FieldRef sinkPort(inst.getResult(sinkArgNum),
                        modSinkPortField.getFieldID());
      auto sinkNode = getOrAddNode(sinkPort);
      bool sinkPortIsForceable = false;
      if (auto refResultType =
              type_dyn_cast<RefType>(inst.getResult(sinkArgNum).getType()))
        sinkPortIsForceable = refResultType.getForceable();

      DenseSet<unsigned> setOfEquivalentRWProbes;
      unsigned minArgNum = sinkArgNum;
      unsigned basePortNode = sinkNode;
      for (auto &modSrcPortField : path.second) {
        auto srcArgNum =
            cast<BlockArgument>(modSrcPortField.getValue()).getArgNumber();
        // Self loop will exist for RWProbes, ignore them.
        if (modSrcPortField == modSinkPortField)
          continue;

        FieldRef srcPort(inst.getResult(srcArgNum),
                         modSrcPortField.getFieldID());
        bool srcPortIsForceable = false;
        if (auto refResultType =
                type_dyn_cast<RefType>(inst.getResult(srcArgNum).getType()))
          srcPortIsForceable = refResultType.getForceable();
        // RWProbes can potentially refer to the same base value. Such ports
        // have a path from each other, a false loop, detect such cases.
        if (sinkPortIsForceable && srcPortIsForceable) {
          auto srcNode = getOrAddNode(srcPort);
          // If a path is already recorded, continue.
          if (rwProbeClasses.findLeader(srcNode) !=
                  rwProbeClasses.member_end() &&
              rwProbeClasses.findLeader(sinkNode) ==
                  rwProbeClasses.findLeader(srcNode))
            continue;
          // Check if sinkPort is a reaching def of sourcePort.
          auto reachingDefsToSrcPort =
              modulePaths->second.find(modSrcPortField);
          if (reachingDefsToSrcPort != modulePaths->second.end())
            if (llvm::find(reachingDefsToSrcPort->second, modSinkPortField) !=
                reachingDefsToSrcPort->second.end()) {
              // This case can occur when there are multiple RWProbes on the
              // port, which refer to the same base value. So, each of such
              // probes are reaching defs of each other. Hence the false
              // loops. Instead of recording this in the reachingDefGraph,
              // record it separately with the rwProbeClasses.
              setOfEquivalentRWProbes.insert(srcNode);
              if (minArgNum > srcArgNum) {
                // Make one of the RWProbe port the base node. Use the first
                // port for deterministic error messages.
                minArgNum = srcArgNum;
                basePortNode = srcNode;
              }
              continue;
            }
        }
        addReachingDef(sinkPort, srcPort);
      }
      if (setOfEquivalentRWProbes.empty())
        continue;

      // Add all the rwprobes to the same class.
      for (auto probe : setOfEquivalentRWProbes)
        rwProbeClasses.unionSets(probe, sinkNode);

      // Make the first port as the base value.
      // Note: this is a port and the actual reference base exists in another
      // module.
      auto leader = rwProbeClasses.getLeaderValue(sinkNode);
      rwProbeRefersTo[leader] = basePortNode;

      setOfEquivalentRWProbes.insert(sinkNode);
      // Add a reaching def  from base RWProbe port to all other RWProbe ports.
      for (auto probe : setOfEquivalentRWProbes)
        if (probe != basePortNode)
          reachingDefGraph[probe].second.push_back(basePortNode);
    }
  }

  // Record the FieldRef, corresponding to the result of the sub op
  // `result = base[index]`
  void recordValueRefersToFieldRef(Value base, unsigned fieldID, Value result) {

    // Check if base is itself field of an aggregate.
    auto it = valToFieldRefs.find(base);
    if (it != valToFieldRefs.end()) {
      // Rebase it to the original aggregate.
      // Because of subaccess op, each value can refer to multiple FieldRefs.
      SmallVector<FieldRef> entry;
      for (auto &sub : it->second)
        entry.emplace_back(sub.getValue(), sub.getFieldID() + fieldID);
      // Update the map at the end, to avoid invaliding the iterator.
      valToFieldRefs[result].append(entry.begin(), entry.end());
      return;
    }
    // Break cycles from registers.
    if (auto *def = base.getDefiningOp()) {
      if (isa<RegOp, RegResetOp, SubfieldOp, SubindexOp, SubaccessOp>(def))
        return;
    }
    valToFieldRefs[result].emplace_back(base, fieldID);
  }

  void handleMemory(MemOp mem) {
    if (mem.getReadLatency() > 0)
      return;
    // Add the enable and address fields as the reaching definitions to the data
    // field.
    for (auto memPort : mem.getResults())
      // TODO: Can reftype ports create cycle ?
      if (auto type = type_dyn_cast<BundleType>(memPort.getType())) {
        auto enableFieldId = type.getFieldID((unsigned)ReadPortSubfield::en);
        auto addressFieldId = type.getFieldID((unsigned)ReadPortSubfield::addr);
        auto dataFieldId = type.getFieldID((unsigned)ReadPortSubfield::data);
        addReachingDef({memPort, static_cast<unsigned int>(dataFieldId)},
                       {memPort, static_cast<unsigned int>(enableFieldId)});
        addReachingDef({memPort, static_cast<unsigned int>(dataFieldId)},
                       {memPort, static_cast<unsigned int>(addressFieldId)});
      }
  }

  // Perform an iterative DFS traversal of the given graph. Record paths between
  // the ports and detect and report any cycles in the graph.
  LogicalResult dfsTraverse(const ReachingDefgraphType &graph) {
    auto numNodes = graph.size();
    SmallVector<bool> onStack(numNodes, false);
    SmallVector<unsigned> dfsStack;

    auto hasCycle = [&](unsigned rootNode, DenseSet<unsigned> &visited,
                        bool recordPortPaths = false) {
      if (visited.contains(rootNode))
        return success();
      dfsStack.push_back(rootNode);

      while (!dfsStack.empty()) {
        auto currentNode = dfsStack.back();

        if (!visited.contains(currentNode)) {
          visited.insert(currentNode);
          onStack[currentNode] = true;
          LLVM_DEBUG(llvm::dbgs()
                     << "\n visiting :"
                     << reachingDefGraph[currentNode].first.getValue().getType()
                     << reachingDefGraph[currentNode].first.getValue() << ","
                     << reachingDefGraph[currentNode].first.getFieldID() << "\n"
                     << getName(reachingDefGraph[currentNode].first));

          FieldRef currentF = reachingDefGraph[currentNode].first;
          if (recordPortPaths && currentNode != rootNode) {
            if (isa<mlir::BlockArgument>(currentF.getValue()))
              portPaths[reachingDefGraph[rootNode].first].insert(currentF);
            // Even if the current node is not a port, there can be RWProbes of
            // the current node at the port.
            addToPortPathsIfRWProbe(
                currentNode, portPaths[reachingDefGraph[rootNode].first]);
          }
        } else {
          onStack[currentNode] = false;
          dfsStack.pop_back();
        }

        for (auto neighbor : graph[currentNode].second) {
          if (!visited.contains(neighbor)) {
            dfsStack.push_back(neighbor);
          } else if (onStack[neighbor]) {
            // Cycle found !!
            SmallVector<FieldRef, 16> path;
            auto loopNode = neighbor;
            // Construct the cyclic path.
            do {
              SmallVector<unsigned>::iterator it =
                  llvm::find_if(reachingDefGraph[loopNode].second,
                                [&](unsigned node) { return onStack[node]; });
              if (it == reachingDefGraph[loopNode].second.end())
                break;

              path.push_back(reachingDefGraph[loopNode].first);
              loopNode = *it;
            } while (loopNode != neighbor);

            reportLoopFound(path, reachingDefGraph[neighbor].first.getLoc());
            return failure();
          }
        }
      }
      return success();
    };

    DenseSet<unsigned> visited;
    for (unsigned node = 0; node < graph.size(); ++node) {
      bool isPort = false;
      if (auto arg =
              dyn_cast<BlockArgument>(reachingDefGraph[node].first.getValue()))
        if (module.getPortDirection(arg.getArgNumber()) == Direction::Out) {
          // For output ports, reset the visited. Required to revisit the entire
          // graph, to discover all the paths that exist from any input port.
          visited.clear();
          isPort = true;
        }

      if (hasCycle(node, visited, isPort).failed())
        return failure();
    }
    return success();
  }

  void reportLoopFound(SmallVectorImpl<FieldRef> &path, Location loc) {
    auto errorDiag = mlir::emitError(
        module.getLoc(), "detected combinational cycle in a FIRRTL module");
    // Find a value we can name
    std::string firstName;
    FieldRef *it = llvm::find_if(path, [&](FieldRef v) {
      firstName = getName(v);
      return !firstName.empty();
    });
    if (it == path.end()) {
      errorDiag.append(", but unable to find names for any involved values.");
      errorDiag.attachNote(loc) << "cycle detected here";
      return;
    }
    // Begin the path from the "smallest string".
    for (circt::FieldRef *findStartIt = it; findStartIt != path.end();
         ++findStartIt) {
      auto n = getName(*findStartIt);
      if (!n.empty() && n < firstName) {
        firstName = n;
        it = findStartIt;
      }
    }
    errorDiag.append(", sample path: ");

    bool lastWasDots = false;
    errorDiag << module.getName() << ".{" << getName(*it);
    for (auto v : llvm::concat<FieldRef>(
             llvm::make_range(std::next(it), path.end()),
             llvm::make_range(path.begin(), std::next(it)))) {
      auto name = getName(v);
      if (!name.empty()) {
        errorDiag << " <- " << name;
        lastWasDots = false;
      } else {
        if (!lastWasDots)
          errorDiag << " <- ...";
        lastWasDots = true;
      }
    }
    errorDiag << "}";
  }

  void dumpMap() {
    LLVM_DEBUG({
      llvm::dbgs() << "\n Reaching Def Graph ==>";
      for (const auto &[index, i] : llvm::enumerate(reachingDefGraph)) {
        llvm::dbgs() << "\n ===>dst:" << getName(i.first)
                     << "::" << i.first.getValue();
        for (auto s : i.second)
          llvm::dbgs() << "<---" << getName(reachingDefGraph[s].first)
                       << "::" << reachingDefGraph[s].first.getValue();
      }

      llvm::dbgs() << "\n Value to FieldRef :";
      for (const auto &fields : valToFieldRefs) {
        llvm::dbgs() << "\n Val:" << fields.first;
        for (auto f : fields.second)
          llvm::dbgs() << ", FieldRef:" << getName(f) << "::" << f.getFieldID();
      }
      llvm::dbgs() << "\n Port paths:";
      for (const auto &p : portPaths) {
        llvm::dbgs() << "\n Output :" << getName(p.first);
        for (auto i : p.second)
          llvm::dbgs() << "\n Input  :" << getName(i);
      }
      llvm::dbgs() << "\n rwprobes:";
      for (auto node : rwProbeRefersTo) {
        llvm::dbgs() << "\n node:"
                     << getName(reachingDefGraph[node.first].first)
                     << "=> probe:"
                     << getName(reachingDefGraph[node.second].first);
      }
      for (auto i = rwProbeClasses.begin(), e = rwProbeClasses.end(); i != e;
           ++i) { // Iterate over all of the equivalence sets.
        if (!i->isLeader())
          continue; // Ignore non-leader sets.
        // Print members in this set.
        llvm::interleave(llvm::make_range(rwProbeClasses.member_begin(i),
                                          rwProbeClasses.member_end()),
                         llvm::dbgs(), "\n");
        llvm::dbgs() << "\n dataflow at leader::" << i->getData() << "\n =>"
                     << rwProbeRefersTo[i->getData()];
        llvm::dbgs() << "\n Done\n"; // Finish set.
      }
    });
  }

  void recordProbe(Value data, Value ref) {
    auto refNode = getOrAddNode(ref);
    auto dataNode = getOrAddNode(data);
    rwProbeRefersTo[rwProbeClasses.getOrInsertLeaderValue(refNode)] = dataNode;
  }

  // Add both the probes to the same equivalence class, to record that they
  // refer to the same data value.
  void probesReferToSameData(Value probe1, Value probe2) {
    auto p1Node = getOrAddNode(probe1);
    auto p2Node = getOrAddNode(probe2);
    rwProbeClasses.unionSets(p1Node, p2Node);
  }

  void addToPortPathsIfRWProbe(unsigned srcNode,
                               DenseSet<FieldRef> &inputPortPaths) {
    // Check if there exists any RWProbe for the srcNode.
    auto baseFieldRef = reachingDefGraph[srcNode].first;
    if (auto defOp = dyn_cast_or_null<Forceable>(baseFieldRef.getDefiningOp()))
      if (defOp.isForceable() && !defOp.getDataRef().use_empty()) {
        // Assumption, the probe must exist in the equivalence classes.
        auto rwProbeNode =
            rwProbeClasses.getLeaderValue(getOrAddNode(defOp.getDataRef()));
        // For all the probes, that are in the same eqv class, i.e., refer to
        // the same value.
        for (auto probe :
             llvm::make_range(rwProbeClasses.member_begin(
                                  rwProbeClasses.findValue(rwProbeNode)),
                              rwProbeClasses.member_end())) {
          auto probeVal = reachingDefGraph[probe].first;
          // If the probe is a port, then record the path from the probe to the
          // input port.
          if (probeVal.getValue().isa<BlockArgument>()) {
            inputPortPaths.insert(probeVal);
          }
        }
      }
  }

private:
  FModuleOp module;
  InstanceGraph &instanceGraph;

  /// Map of value to the corresponding FieldRef.
  DenseMap<Value, SmallVector<FieldRef>> valToFieldRefs;
  /// Comb paths that exist between module ports. This is maintained across
  /// modules.
  const DenseMap<FModuleLike, ReachingDefsMapType> &modulePortPaths;
  /// The comb paths between the ports of this module. This is the final
  /// output of this intra-procedural analysis, that is used to construct the
  /// inter-procedural dataflow.
  ReachingDefsMapType &portPaths;

  /// This is an adjacency list representation of the Reaching definitions
  /// graph. This can be indexed by the graph node id, and each entry is the
  /// list of graph nodes that has an edge to it. Each graph node represents a
  /// FieldRef and each edge represents an immediate reaching definition from
  /// source to the sink of the edge.
  ReachingDefgraphType reachingDefGraph;
  /// Map of FieldRef to its corresponding graph node.
  DenseMap<FieldRef, size_t> nodes;

  /// The base value that the RWProbe refers to. Used to add an edge to the base
  /// value, when the probe is forced.
  DenseMap<unsigned, unsigned> rwProbeRefersTo;

  /// An eqv class of all the RWProbes that refer to the same base value.
  llvm::EquivalenceClasses<unsigned> rwProbeClasses;
};

/// This pass constructs a local graph for each module to detect
/// combinational cycles. To capture the cross-module combinational cycles,
/// this pass inlines the combinational paths between IOs of its
/// subinstances into a subgraph and encodes them in `modulePortPaths`.
class CheckCombLoopsPass : public CheckCombLoopsBase<CheckCombLoopsPass> {
public:
  void runOnOperation() override {
    auto &instanceGraph = getAnalysis<InstanceGraph>();
    DenseMap<FModuleLike, ReachingDefsMapType> modulePortPaths;

    // Traverse modules in a post order to make sure the combinational paths
    // between IOs of a module have been detected and recorded in
    // `modulePortPaths` before we handle its parent modules.
    for (auto *igNode : llvm::post_order<InstanceGraph *>(&instanceGraph)) {
      if (auto module = dyn_cast<FModuleOp>(*igNode->getModule())) {
        DiscoverLoops rdf(module, instanceGraph, modulePortPaths,
                          modulePortPaths[module]);
        if (rdf.processModule().failed()) {
          return signalPassFailure();
        }
      }
    }
    markAllAnalysesPreserved();
  }
};

std::unique_ptr<mlir::Pass> circt::firrtl::createCheckCombLoopsPass() {
  return std::make_unique<CheckCombLoopsPass>();
}
