//===- CheckCombLoops.cpp - FIRRTL check combinational cycles ------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements the FIRRTL combinational cycles detection pass.
// Terminology:
// In the context of the circt that the MLIR represents, a Value is called the
// driver of another Value, if the driver actively sets the the other Value. The
// driver Value is responsible for determining the logic level of the driven
// Value.
// This pass is a dataflow analysis that interprets the operations of the
// circt to build a connectivity graph, which represents the driver
// relationships. Each node in this connectivity graph is a FieldRef, and an
// edge exists from a source node to a desination node if the source drives the
// destination..
// Algorithm to construct the connectivity graph.
// 1. Traverse each module in the Instance Graph bottom up.
// 2. Preprocess step: Construct the circt connectivity directed graph.
// Perform a dataflow analysis, on the domain of FieldRefs. Interpret each
// operation, to record the values that can potentially drive another value.
// Each node in the graph is a fieldRef. Each edge represents a dataflow from
// the source to the sink.
// 3. Perform a DFS traversal on the graph, to detect combinational loops and
// paths between ports.
// 4. Inline the combinational paths discovered in each module to its instance
// and continue the analysis through the instance graph.
//===----------------------------------------------------------------------===//

#include "circt/Dialect/FIRRTL/CHIRRTLDialect.h"
#include "circt/Dialect/FIRRTL/FIRRTLInstanceGraph.h"
#include "circt/Dialect/FIRRTL/FIRRTLOps.h"
#include "circt/Dialect/FIRRTL/FIRRTLUtils.h"
#include "circt/Dialect/FIRRTL/FIRRTLVisitors.h"
#include "circt/Dialect/FIRRTL/Passes.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/EquivalenceClasses.h"
#include "llvm/ADT/PostOrderIterator.h"

#define DEBUG_TYPE "check-comb-loops"

namespace circt {
namespace firrtl {
#define GEN_PASS_DEF_CHECKCOMBLOOPS
#include "circt/Dialect/FIRRTL/Passes.h.inc"
} // namespace firrtl
} // namespace circt

using namespace circt;
using namespace firrtl;

using DrivenBysMapType = DenseMap<FieldRef, DenseSet<FieldRef>>;

/// The results of a module, that are inlined at its instances.
struct ModuleSummary {
  /// The comb paths between the ports of the module.
  DrivenBysMapType portPaths;
  /// The RWProbe ports of registers, forcing them is not combinational.
  DenseSet<unsigned> registerProbePorts;
};

class DiscoverLoops {

  /// Adjacency list representation.
  /// Each entry is a pair, the first element is the FieldRef corresponding to
  /// the graph vertex. The second element is the list of vertices, that have an
  /// edge to this vertex.
  /// The directed edges represent a connectivity relation, of a source that
  /// drives the sink.
  using DrivenByGraphType =
      SmallVector<std::pair<FieldRef, SmallVector<unsigned>>, 64>;

public:
  DiscoverLoops(
      FModuleOp module, InstanceGraph &instanceGraph,
      hw::InnerRefNamespace &irn,
      const DenseMap<FModuleLike, ModuleSummary> &otherModuleSummaries,
      ModuleSummary &thisModuleSummary)
      : module(module), instanceGraph(instanceGraph), irn(irn),
        moduleSummaries(otherModuleSummaries),
        portPaths(thisModuleSummary.portPaths),
        registerProbePorts(thisModuleSummary.registerProbePorts) {}

  LogicalResult processModule() {
    LLVM_DEBUG(llvm::dbgs() << "\n processing module :" << module.getName());
    constructConnectivityGraph(module);
    recordRegisterProbePorts();
    auto result = dfsTraverse(drivenBy);
    dumpMap();
    return result;
  }

  void constructConnectivityGraph(FModuleOp module) {
    LLVM_DEBUG(llvm::dbgs() << "\n Module :" << module.getName());

    // ALl the module output ports must be added as the initial nodes.
    for (auto port : module.getArguments()) {
      if (module.getPortDirection(port.getArgNumber()) != Direction::Out)
        continue;
      walkGroundTypes(cast<FIRRTLType>(port.getType()),
                      [&](uint64_t index, FIRRTLBaseType t, auto isFlip) {
                        getOrAddNode(FieldRef(port, index));
                      });
    }

    walk(module, [&](Operation *op) {
      llvm::TypeSwitch<Operation *>(op)
          .Case<hw::CombDataFlow>([&](hw::CombDataFlow df) {
            // computeDataFlow returns a pair of FieldRefs, first element is the
            // destination and the second is the source.
            for (auto [dest, source] : df.computeDataFlow())
              addDrivenBy(dest, source);
            // Registers are CombDataFlow, so they miss the Forceable case.
            if (auto forceableOp = dyn_cast<Forceable>(op))
              recordForceableProbe(forceableOp);
          })
          .Case<Forceable>([&](Forceable forceableOp) {
            // Any declaration that can be forced.
            if (auto node = dyn_cast<NodeOp>(op))
              recordDataflow(node.getData(), node.getInput());
            recordForceableProbe(forceableOp);
          })
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
          .Case<RefCastOp>([&](RefCastOp cast) {
            recordDataflow(cast.getResult(), cast.getInput());
            // A cast RWProbe refers to the same data as its input.
            if (cast.getType().getForceable())
              probesReferToSameData(cast.getInput(), cast.getResult());
          })
          .Case<RefForceOp, RefForceInitialOp>([&](auto ref) {
            forces.emplace_back(ref.getDest(), ref.getSrc());
          })
          .Case<InstanceOp>([&](auto inst) { handleInstanceOp(inst); })
          .Case<InstanceChoiceOp>(
              [&](auto inst) { handleInstanceChoiceOp(inst); })
          .Case<SubindexOp>([&](SubindexOp sub) {
            recordValueRefersToFieldRef(
                sub.getInput(),
                sub.getInput().getType().base().getFieldID(sub.getIndex()),
                sub.getResult());
          })
          .Case<SubfieldOp>([&](SubfieldOp sub) {
            recordValueRefersToFieldRef(
                sub.getInput(),
                sub.getInput().getType().base().getFieldID(sub.getFieldIndex()),
                sub.getResult());
          })
          .Case<SubaccessOp>([&](SubaccessOp sub) {
            auto vecType = sub.getInput().getType().base();
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
            // Resolved after the walk, once the data of the input is known.
            if (sub.getInput().getType().getForceable())
              rwProbeSubs.push_back(sub.getResult());
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
          .Case<RWProbeOp>([&](RWProbeOp probe) {
            auto ist = irn.lookup(probe.getTarget());
            auto target = getFieldRefForTarget(ist);
            if (!target) {
              probe->emitWarning("check-comb-loops cannot resolve target");
              return;
            }
            auto res = probe.getResult();
            auto type = probe.getType().getType();
            walkGroundTypes(type, [&](uint64_t index, FIRRTLBaseType, bool) {
              addDrivenBy(FieldRef(res, index), target.getSubField(index));
            });
            if (!type.isGround())
              addDrivenBy({res, 0}, target);
            recordProbe(target, getOrAddNode(res));
          })
          .Default([&](Operation *op) {
            // All other expressions are assumed to be combinational, so record
            // the dataflow between all inputs to outputs.
            for (auto res : op->getResults())
              for (auto src : op->getOperands())
                recordDataflow(res, src);
          });
    });

    for (auto sub : rwProbeSubs)
      recordRefSubProbe(sub);
    for (auto [dest, src] : forces)
      handleRefForce(dest, src);
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
    auto id = drivenBy.size();
    // The node id can be used to index into the graph. The entry is a pair,
    // first element is the corresponding FieldRef, and the second entry is a
    // list of adjacent nodes.
    drivenBy.push_back({f, {}});
    nodes[f] = id;
    return id;
  }

  // Construct the connectivity graph, by adding `dst` and `src` as new nodes,
  // if not already existing. Then add an edge from `src` to `dst`.
  void addDrivenBy(FieldRef dst, FieldRef src) {
    auto srcNode = getOrAddNode(src);
    auto dstNode = getOrAddNode(dst);
    drivenBy[dstNode].second.push_back(srcNode);
  }

  // Add `dstVal` as being driven by `srcVal`.
  void recordDataflow(Value dstVal, Value srcVal) {
    // Ignore connectivity from constants.
    if (auto *def = srcVal.getDefiningOp())
      if (def->hasTrait<OpTrait::ConstantLike>())
        return;
    // Check if srcVal/dstVal is a fieldRef to an aggregate. Then, there may
    // exist other values, that refer to the same fieldRef. Add a connectivity
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
      return addDrivenBy({dstVal, 0}, {srcVal, 0});

    auto addDef = [&](FieldRef dst, FieldRef src) {
      // If the dstVal and srcVal are aggregate types, then record the dataflow
      // between each individual ground type. This is equivalent to flattening
      // the type to ensure all the contained FieldRefs are also recorded.
      if (dstValType && !dstValType.isGround())
        walkGroundTypes(dstValType, [&](uint64_t dstIndex, FIRRTLBaseType t,
                                        bool dstIsFlip) {
          // Handle the case when the dst and src are not of the same type.
          // For each dst ground type, and for each src ground type.
          if (srcValType == dstValType) {
            // This is the only case when the flip is valid. Flip is relevant
            // only for connect ops, and src and dst of a connect op must be
            // type equivalent!
            if (dstIsFlip)
              std::swap(dst, src);
            addDrivenBy(dst.getSubField(dstIndex), src.getSubField(dstIndex));
          } else if (srcValType && !srcValType.isGround())
            walkGroundTypes(srcValType, [&](uint64_t srcIndex, FIRRTLBaseType t,
                                            auto) {
              addDrivenBy(dst.getSubField(dstIndex), src.getSubField(srcIndex));
            });
          // Else, the src is ground type.
          else
            addDrivenBy(dst.getSubField(dstIndex), src);
        });

      addDrivenBy(dst, src);
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

  // Return the data of any RWProbe in the class of `probeNode`.
  std::optional<unsigned> getProbeData(unsigned probeNode) {
    auto leader = rwProbeClasses.findLeader(probeNode);
    if (leader == rwProbeClasses.member_end())
      return std::nullopt;
    for (auto probe : rwProbeClasses.members(*leader)) {
      auto iter = rwProbeRefersTo.find(probe);
      if (iter != rwProbeRefersTo.end())
        return iter->second;
    }
    return std::nullopt;
  }

  // Record a RefSubOp of a RWProbe as a probe of the same field of its data.
  void recordRefSubProbe(Value sub) {
    auto iter = valToFieldRefs.find(sub);
    if (iter == valToFieldRefs.end())
      return;
    for (auto field : iter->second) {
      auto inputNode = getOrAddNode(FieldRef(field.getValue(), 0));
      auto subNode = getOrAddNode(field);
      if (isRegisterProbe(inputNode)) {
        addProbeNode(subNode);
        registerProbes.insert(subNode);
        continue;
      }
      auto data = getProbeData(inputNode);
      if (!data)
        continue;
      recordProbe(drivenBy[*data].first.getSubField(field.getFieldID()),
                  subNode);
    }
  }

  // Record srcVal as driving the original data value that the probe refers to.
  void handleRefForce(Value dstProbe, Value srcVal) {
    auto dstNode = getOrAddNode(dstProbe);
    // Forcing a register is not combinational.
    if (isRegisterProbe(dstNode))
      return;
    recordDataflow(dstProbe, srcVal);
    // Now add srcVal as driving the data that dstProbe refers to.
    auto leader = rwProbeClasses.findLeader(dstNode);
    if (leader == rwProbeClasses.member_end())
      return;
    auto members = rwProbeClasses.members(*leader);
    std::optional<unsigned> base = getProbeData(dstNode);
    // An instance probe port stands in for the forced data inside the child.
    if (!base) {
      for (auto probe : members) {
        auto val = drivenBy[probe].first.getValue();
        if (!isa_and_nonnull<InstanceOp, InstanceChoiceOp>(val.getDefiningOp()))
          continue;
        if (auto refType = type_dyn_cast<RefType>(val.getType());
            refType && refType.getForceable()) {
          base = probe;
          break;
        }
      }
    }
    if (!base || *base == dstNode)
      return;
    recordForceDataflow(drivenBy[*base].first, srcVal);
  }

  // Record `srcVal` as driving the forced `data`, per field, since `data` may
  // be read by field.
  void recordForceDataflow(FieldRef data, Value srcVal) {
    // Ignore connectivity from constants.
    if (auto *def = srcVal.getDefiningOp())
      if (def->hasTrait<OpTrait::ConstantLike>())
        return;
    SmallVector<FieldRef> srcFields;
    auto srcIt = valToFieldRefs.find(srcVal);
    if (srcIt != valToFieldRefs.end())
      srcFields.append(srcIt->second.begin(), srcIt->second.end());
    else
      srcFields.emplace_back(srcVal, 0);
    auto type = getBaseType(srcVal.getType());
    for (auto src : srcFields) {
      addDrivenBy(data, src);
      if (type && !type.isGround())
        walkGroundTypes(type, [&](uint64_t index, FIRRTLBaseType, bool) {
          addDrivenBy(data.getSubField(index), src.getSubField(index));
        });
    }
  }

  // Return true if there is a path from the port `src` to the port `sink`.
  static bool hasPortPath(const DrivenBysMapType &paths, FieldRef sink,
                          FieldRef src) {
    auto iter = paths.find(sink);
    return iter != paths.end() && iter->second.contains(src);
  }

  // Helper to process instance ports for a given module and instance results.
  // This is used by both handleInstanceOp and handleInstanceChoiceOp.
  void processInstancePorts(FModuleOp refMod, ValueRange instResults) {
    auto summary = moduleSummaries.find(refMod);
    if (summary == moduleSummaries.end())
      return;
    const auto &modulePaths = summary->second.portPaths;
    // Note: Handling RWProbes.
    // 1. For RWProbes, output ports can be source of dataflow.
    // 2. All the RWProbes that refer to the same base value form a strongly
    // connected component. Each has a dataflow from the other, including
    // itself.
    // Steps to add the instance ports to the connectivity graph:
    // 1. Find the set of RWProbes that refer to the same base value.
    // 2. Add them to the same rwProbeClasses.
    // 3. Choose the first RWProbe port from this set as a representative base
    //    value. And add it as the source driver for every other RWProbe
    //    port in the set.
    // 4. This will ensure we can detect cycles involving different RWProbes to
    //    the same base value.
    // 5. A RWProbe port of another output port also has paths to and from it.
    //    Record it as a probe of that port, instead of a false loop.
    for (auto &path : modulePaths) {
      auto modSinkPortField = path.first;
      auto sinkArgNum =
          cast<BlockArgument>(modSinkPortField.getValue()).getArgNumber();
      FieldRef sinkPort(instResults[sinkArgNum], modSinkPortField.getFieldID());
      auto sinkNode = getOrAddNode(sinkPort);
      bool sinkPortIsForceable = false;
      if (auto refResultType =
              type_dyn_cast<RefType>(instResults[sinkArgNum].getType()))
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

        FieldRef srcPort(instResults[srcArgNum], modSrcPortField.getFieldID());
        bool srcPortIsForceable = false;
        if (auto refResultType =
                type_dyn_cast<RefType>(instResults[srcArgNum].getType()))
          srcPortIsForceable = refResultType.getForceable();
        // The RWProbe port refers to the sink port (5. above).
        if (!sinkPortIsForceable && srcPortIsForceable &&
            hasPortPath(modulePaths, modSrcPortField, modSinkPortField)) {
          recordProbe(sinkPort, getOrAddNode(srcPort));
          continue;
        }
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
          // Check if sinkPort is a driver of sourcePort.
          if (hasPortPath(modulePaths, modSrcPortField, modSinkPortField)) {
            // This case can occur when there are multiple RWProbes on the
            // port, which refer to the same base value. So, each of such
            // probes are drivers of each other. Hence the false
            // loops. Instead of recording this in the drivenByGraph,
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
        addDrivenBy(sinkPort, srcPort);
      }
      if (setOfEquivalentRWProbes.empty())
        continue;

      setOfEquivalentRWProbes.insert(sinkNode);
      // Add all the rwprobes to the same class.
      for (auto probe : setOfEquivalentRWProbes) {
        addProbeNode(probe);
        rwProbeClasses.unionSets(probe, sinkNode);
      }

      // Make the first port as the base value.
      // Note: this is a port and the actual reference base exists in another
      // module. Add the base RWProbe port as a driver to all other RWProbe
      // ports.
      for (auto probe : setOfEquivalentRWProbes) {
        rwProbeRefersTo.try_emplace(probe, basePortNode);
        if (probe != basePortNode)
          drivenBy[probe].second.push_back(basePortNode);
      }
    }
  }

  // Mark the instance ports that are register RWProbes in all the modules.
  void recordInstanceRegisterProbes(ArrayRef<FModuleOp> refMods,
                                    ValueRange instResults) {
    if (refMods.empty())
      return;
    for (unsigned argNum = 0, e = instResults.size(); argNum < e; ++argNum) {
      if (!llvm::all_of(refMods, [&](FModuleOp refMod) {
            auto summary = moduleSummaries.find(refMod);
            return summary != moduleSummaries.end() &&
                   summary->second.registerProbePorts.contains(argNum);
          }))
        continue;
      auto node = getOrAddNode(FieldRef(instResults[argNum], 0));
      addProbeNode(node);
      registerProbes.insert(node);
    }
  }

  // Check the referenced module paths and add input ports as the drivers for
  // the corresponding output port. The granularity of the connectivity
  // relations is per field.
  void handleInstanceOp(InstanceOp inst) {
    auto refMod = inst.getReferencedModule<FModuleOp>(instanceGraph);
    // Skip if the instance is not a module (e.g. external module).
    if (!refMod)
      return;
    recordInstanceRegisterProbes(refMod, inst.getResults());
    processInstancePorts(refMod, inst.getResults());
  }

  // For InstanceChoiceOp, conservatively process all possible target modules.
  // Since we cannot determine which module will be selected at runtime, we
  // must consider combinational paths through all alternatives.
  void handleInstanceChoiceOp(InstanceChoiceOp inst) {
    SmallVector<FModuleOp> refMods;
    // Conservatively, a port is a register RWProbe only if it is one in all
    // the alternatives.
    bool allAreModules = true;
    // Process all referenced modules (default + alternatives)
    for (auto moduleName : inst.getReferencedModuleNamesAttr()) {
      auto moduleNameStr = cast<StringAttr>(moduleName);
      auto *node = instanceGraph.lookup(moduleNameStr);
      if (!node) {
        allAreModules = false;
        continue;
      }

      // Skip if the instance is not a module (e.g. external module).
      if (auto refMod = dyn_cast<FModuleOp>(*node->getModule()))
        refMods.push_back(refMod);
      else
        allAreModules = false;
    }
    if (allAreModules)
      recordInstanceRegisterProbes(refMods, inst.getResults());
    for (auto refMod : refMods)
      processInstancePorts(refMod, inst.getResults());
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

  // Perform an iterative DFS traversal of the given graph. Record paths between
  // the ports and detect and report any cycles in the graph.
  LogicalResult dfsTraverse(const DrivenByGraphType &graph) {
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
                     << drivenBy[currentNode].first.getValue().getType()
                     << drivenBy[currentNode].first.getValue() << ","
                     << drivenBy[currentNode].first.getFieldID() << "\n"
                     << getName(drivenBy[currentNode].first));

          FieldRef currentF = drivenBy[currentNode].first;
          if (recordPortPaths) {
            auto &inputPortPaths = portPaths[drivenBy[rootNode].first];
            if (currentNode != rootNode &&
                isa<mlir::BlockArgument>(currentF.getValue()))
              inputPortPaths.insert(currentF);
            // Even if the current node is not a port, there can be RWProbes of
            // the current node at the port. Includes the root, which is driven
            // by a force of its RWProbes.
            addToPortPathsIfRWProbe(currentNode, inputPortPaths);
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
                  llvm::find_if(drivenBy[loopNode].second,
                                [&](unsigned node) { return onStack[node]; });
              if (it == drivenBy[loopNode].second.end())
                break;

              path.push_back(drivenBy[loopNode].first);
              loopNode = *it;
            } while (loopNode != neighbor);

            reportLoopFound(path, drivenBy[neighbor].first.getLoc());
            return failure();
          }
        }
      }
      return success();
    };

    DenseSet<unsigned> visited;
    for (unsigned node = 0; node < graph.size(); ++node) {
      bool isPort = false;
      if (auto arg = dyn_cast<BlockArgument>(drivenBy[node].first.getValue()))
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
      llvm::dbgs() << "\n Connectivity Graph ==>";
      for (const auto &[index, i] : llvm::enumerate(drivenBy)) {
        llvm::dbgs() << "\n ===>dst:" << getName(i.first)
                     << "::" << i.first.getValue();
        for (auto s : i.second)
          llvm::dbgs() << "<---" << getName(drivenBy[s].first)
                       << "::" << drivenBy[s].first.getValue();
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
        llvm::dbgs() << "\n node:" << getName(drivenBy[node.first].first)
                     << "=> probe:" << getName(drivenBy[node.second].first);
      }
      for (const auto &i :
           rwProbeClasses) { // Iterate over all of the equivalence sets.
        if (!i->isLeader())
          continue; // Ignore non-leader sets.
        // Print members in this set.
        llvm::interleave(rwProbeClasses.members(*i), llvm::dbgs(), "\n");
        llvm::dbgs() << "\n dataflow at leader::" << i->getData() << "\n =>"
                     << rwProbeRefersTo.lookup(i->getData());
        llvm::dbgs() << "\n Done\n"; // Finish set.
      }
    });
  }

  void recordForceableProbe(Forceable forceableOp) {
    if (!forceableOp.isForceable() || forceableOp.getDataRef().use_empty())
      return;
    auto data = forceableOp.getData();
    auto ref = forceableOp.getDataRef();
    // Record dataflow from data to the probe.
    recordDataflow(ref, data);
    recordProbe({data, 0}, getOrAddNode(ref));
  }

  static uint64_t getMaxFieldID(FieldRef ref) {
    auto type = hw::FieldIdImpl::getFinalTypeByFieldID(
        getBaseType(ref.getValue().getType()), ref.getFieldID());
    return hw::FieldIdImpl::getMaxFieldID(type);
  }

  // The probe is recorded in rwProbesOf, so that a read through it records a
  // port path to all the RWProbe ports in its class.
  void addProbeNode(unsigned probeNode) {
    if (rwProbeClasses.contains(probeNode))
      return;
    rwProbeClasses.insert(probeNode);
    auto probe = drivenBy[probeNode].first;
    uint64_t low = probe.getFieldID();
    rwProbesOf[probe.getValue()].emplace_back(low, low + getMaxFieldID(probe),
                                              probeNode);
  }

  // Return true if the RWProbe refers to a register.
  bool isRegisterProbe(unsigned probeNode) {
    if (registerProbes.contains(probeNode))
      return true;
    auto leader = rwProbeClasses.findLeader(probeNode);
    if (leader == rwProbeClasses.member_end())
      return false;
    return llvm::any_of(rwProbeClasses.members(*leader), [&](unsigned probe) {
      return registerProbes.contains(probe);
    });
  }

  void recordProbe(FieldRef data, unsigned refNode) {
    addProbeNode(refNode);
    // Forcing a register is not combinational, so a force must not add an edge
    // into the register.
    if (isa_and_nonnull<RegOp, RegResetOp>(data.getDefiningOp())) {
      registerProbes.insert(refNode);
      return;
    }
    auto dataNode = getOrAddNode(data);
    rwProbeRefersTo[refNode] = dataNode;
    uint64_t low = data.getFieldID();
    uint64_t high = low + getMaxFieldID(drivenBy[refNode].first);
    rwProbesOf[data.getValue()].emplace_back(low, high, refNode);
  }

  // Add both the probes to the same equivalence class, to record that they
  // refer to the same data value.
  void probesReferToSameData(Value probe1, Value probe2) {
    auto p1Node = getOrAddNode(probe1);
    auto p2Node = getOrAddNode(probe2);
    addProbeNode(p1Node);
    addProbeNode(p2Node);
    rwProbeClasses.unionSets(p1Node, p2Node);
  }

  // Record the register RWProbe ports, so that the instances of this module
  // do not treat forcing them as combinational.
  void recordRegisterProbePorts() {
    for (auto port : module.getArguments()) {
      auto refType = type_dyn_cast<RefType>(port.getType());
      if (!refType || !refType.getForceable())
        continue;
      auto iter = nodes.find(FieldRef(port, 0));
      if (iter != nodes.end() && isRegisterProbe(iter->second))
        registerProbePorts.insert(port.getArgNumber());
    }
  }

  void addToPortPathsIfRWProbe(unsigned srcNode,
                               DenseSet<FieldRef> &inputPortPaths) {
    // Check if there exists any RWProbe for the srcNode, which can also be a
    // RWProbe itself.
    auto ref = drivenBy[srcNode].first;
    auto it = rwProbesOf.find(ref.getValue());
    if (it == rwProbesOf.end())
      return;
    for (auto [low, high, probeNode] : it->second) {
      if (ref.getFieldID() < low || ref.getFieldID() > high)
        continue;
      if (isRegisterProbe(probeNode))
        continue;
      auto leader = rwProbeClasses.findLeader(probeNode);
      // If a probe in the same eqv class is a port, then record the path from
      // the corresponding field of the probe to the input port.
      for (auto probe : rwProbeClasses.members(*leader))
        if (isa<BlockArgument>(drivenBy[probe].first.getValue()))
          inputPortPaths.insert(
              drivenBy[probe].first.getSubField(ref.getFieldID() - low));
    }
  }

private:
  FModuleOp module;
  InstanceGraph &instanceGraph;
  hw::InnerRefNamespace &irn;

  /// Map of values to the set of all FieldRefs (same base) that this may be
  /// directly derived from through indexing operations.
  DenseMap<Value, SmallVector<FieldRef>> valToFieldRefs;
  /// The summaries of the modules. This is maintained across modules.
  const DenseMap<FModuleLike, ModuleSummary> &moduleSummaries;
  /// The comb paths between the ports of this module. This is the final
  /// output of this intra-procedural analysis, that is used to construct the
  /// inter-procedural dataflow.
  DrivenBysMapType &portPaths;
  /// The register RWProbe ports of this module.
  DenseSet<unsigned> &registerProbePorts;

  /// This is an adjacency list representation of the connectivity graph. This
  /// can be indexed by the graph node id, and each entry is the list of graph
  /// nodes that has an edge to it. Each graph node represents a FieldRef and
  /// each edge represents a source that directly drives the sink node.
  DrivenByGraphType drivenBy;
  /// Map of FieldRef to its corresponding graph node.
  DenseMap<FieldRef, size_t> nodes;

  /// The base value that the RWProbe node refers to. Used to add an edge to the
  /// base value, when the probe is forced.
  DenseMap<unsigned, unsigned> rwProbeRefersTo;

  /// The RWProbes of each value, as (lowFieldID, highFieldID, probeNode).
  DenseMap<Value, SmallVector<std::tuple<uint64_t, uint64_t, unsigned>>>
      rwProbesOf;

  /// An eqv class of all the RWProbes that refer to the same base value.
  llvm::EquivalenceClasses<unsigned> rwProbeClasses;

  /// The RWProbes that refer to a register.
  DenseSet<unsigned> registerProbes;

  /// The (dest, src) of all forces, handled after the walk once all the probes
  /// are known.
  SmallVector<std::pair<Value, Value>> forces;

  /// The results of all RefSubOps of RWProbes, handled after the walk once all
  /// the probes are known.
  SmallVector<Value> rwProbeSubs;
};

/// This pass constructs a local graph for each module to detect
/// combinational cycles. To capture the cross-module combinational cycles,
/// this pass inlines the combinational paths between IOs of its
/// subinstances into a subgraph and encodes them in `moduleSummaries`.
class CheckCombLoopsPass
    : public circt::firrtl::impl::CheckCombLoopsBase<CheckCombLoopsPass> {
public:
  void runOnOperation() override {
    auto &instanceGraph = getAnalysis<InstanceGraph>();
    DenseMap<FModuleLike, ModuleSummary> moduleSummaries;
    hw::InnerRefNamespace irn{getAnalysis<SymbolTable>(),
                              getAnalysis<hw::InnerSymbolTableCollection>()};

    // Traverse modules in a post order to make sure the combinational paths
    // between IOs of a module have been detected and recorded in
    // `moduleSummaries` before we handle its parent modules.
    for (auto *igNode : llvm::post_order<InstanceGraph *>(&instanceGraph)) {
      if (auto module = dyn_cast<FModuleOp>(*igNode->getModule())) {
        DiscoverLoops rdf(module, instanceGraph, irn, moduleSummaries,
                          moduleSummaries[module]);
        if (rdf.processModule().failed()) {
          return signalPassFailure();
        }
      }
    }
    markAllAnalysesPreserved();
  }
};
