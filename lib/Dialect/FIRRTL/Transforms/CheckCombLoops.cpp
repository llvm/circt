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
//===----------------------------------------------------------------------===//

#include "PassDetails.h"
#include "circt/Dialect/FIRRTL/FIRRTLInstanceGraph.h"
#include "circt/Dialect/FIRRTL/FIRRTLUtils.h"
#include "circt/Dialect/FIRRTL/FIRRTLVisitors.h"
#include "circt/Dialect/FIRRTL/Passes.h"
#include "llvm/ADT/DepthFirstIterator.h"
#include "llvm/ADT/PostOrderIterator.h"
#include "llvm/ADT/SCCIterator.h"
#include "llvm/ADT/SmallSet.h"
#include <variant>

#define DEBUG_TYPE "check-comb-loops"

using namespace circt;
using namespace firrtl;
struct Node {
  Value val = nullptr;
  size_t fieldId = 0;
  Node() = default;
  Node(Value val, size_t fieldId) : val(val), fieldId(fieldId) {}
  Node(const Node &rhs) = default;
  Node &operator=(const Node &rhs) = default;
  bool operator==(const Node &rhs) const {
    return (val == rhs.val && fieldId == rhs.fieldId);
  }
  bool operator!=(const Node &rhs) const { return !(*this == rhs); }
  bool isValid() { return val != nullptr; }
  FModuleOp getParentModule() {
    assert(val && "Must be a valid node");
    if (val.isa<BlockArgument>())
      return cast<FModuleOp>(val.getParentBlock()->getParentOp());
    return val.getDefiningOp()->getParentOfType<FModuleOp>();
  }
};

class ModuleDFS {
  using ValueOpPair = std::pair<Value, Operation *>;

public:
  ModuleDFS(
      FModuleOp module, InstanceGraph &instanceGraph,
      llvm::SmallDenseMap<Value, llvm::SmallDenseMap<size_t, SmallVector<Node>>>
          &portPaths)
      : module(module), instanceGraph(instanceGraph), portPaths(portPaths) {}

  LogicalResult dfsFromNode(Value node, Node inputArg) {

    LLVM_DEBUG(llvm::dbgs() << "\n visiting node :" << node);
    if (!node.getType().isa<FIRRTLBaseType>())
      return success();
    if (visited.contains(node))
      return success();
    // visiting: Set of nodes that have a path to this current node.
    if (visiting.contains(node)) {
      // A comb cycle exists, print an error.
      FieldRef f = getFieldRef(node);
      auto lastSignalName = getFieldName(f).first;
      auto errorDiag = mlir::emitError(
          module.getLoc(),
          "detected combinational cycle in a FIRRTL module, sample path: ");
      if (!lastSignalName.empty())
        errorDiag << module.getName() << "." << lastSignalName << " <- ";
      size_t pathSize = 1;

      for (auto n : llvm::reverse(currentPath)) {
        FieldRef f = getFieldRefFromValue(n);
        auto signalName = getFieldName(f).first;
        if (!signalName.empty() && lastSignalName != signalName) {
          errorDiag << module.getName() << "." << signalName << " <- ";
          ++pathSize;
        }
        lastSignalName = signalName;
        if (n == node)
          break;
      }
      if (pathSize == 1)
        errorDiag << module.getName() << "." << lastSignalName << " <- ";
      currentPath.push_back(node);
      return failure();
    }
    auto tmpVisiting = visiting;
    visiting.insert(node);
    SmallVector<Value> valuesForSameStorageLoc = {node};

    SmallVector<ValueOpPair> children;
    // Now get all the Values, that refer to the same storage location as
    // `node`.
    auto storageLocIter = valToStorageLocIdMap.find(node);
    if (storageLocIter != valToStorageLocIdMap.end())
      // A Value can refer to multiple Storage locations, (for example, result
      // of subaccess op).
      for (auto storageLocId : storageLocIter->getSecond())
        // All the values that refer to `storageLocId` are a valid child of
        // `node`.
        for (auto aliasVal : storageLocIdToValuesMap[storageLocId])
          // The following check ensures, Values referring to the same Location
          // Id are added only once.
          if (aliasVal != node) {
            valuesForSameStorageLoc.push_back(aliasVal);
            visiting.insert(aliasVal);
          }
    currentPath.push_back(node);
    for (auto visitingNode : valuesForSameStorageLoc) {
      // Record the current value being traversed, used for printing the
      // discovered cycle.
      currentPath.push_back(visitingNode);
      // Get the Values that have an edge from this node. The ValueOpPair
      // records the Operation that is the owner of the corresponding child
      // value. The operation will only be used for error reporting.
      getChildren(visitingNode, inputArg, children);
      // Now continue the DFS recursively.
      for (auto childNode : children)
        if (dfsFromNode(childNode.first, inputArg).failed()) {
          printError(visitingNode, childNode);
          return failure();
        }
      currentPath.pop_back();
    }
    currentPath.pop_back();

    visiting = std::move(tmpVisiting);
    // Finished discovering all the reachable nodes from node.
    visited.insert(node);
    return success();
  }

  /// Customized implementation of getFieldRefFromValue. If `field` is a
  /// subfield, then this will return the base aggregate Value and the
  /// corresponding fieldId. This will return nullptr if thie field is a result
  /// of a register and can be ignored.
  FieldRef getFieldRef(Value field) {
    if (!isa_and_nonnull<SubfieldOp, SubaccessOp, SubindexOp>(
            field.getDefiningOp()))
      return FieldRef(field, 0);
    auto iter = leafToBaseValMap.find(field);
    if (iter == leafToBaseValMap.end())
      return FieldRef(nullptr, 0);
    auto baseVal = iter->second;
    auto storIter = valToStorageLocIdMap.find(baseVal);
    if (storIter == valToStorageLocIdMap.end())
      return FieldRef(nullptr, 0);
    auto baseStorId = storIter->getSecond().front();
    auto fieldStorIter = valToStorageLocIdMap.find(field);
    if (fieldStorIter == valToStorageLocIdMap.end())
      return FieldRef(nullptr, 0);
    auto fieldStorId = fieldStorIter->getSecond().front();

    return FieldRef(baseVal, fieldStorId - baseStorId);
  }

  // Add all the Values to `children` that have a valid dataflow from `node`.
  // The `inputArg` is valid, only if it is one of the reaching definitions to
  // `node`.
  void getChildren(Value node, Node &inputArg,
                   SmallVector<ValueOpPair> &children) {

    // Handle the following cases
    // Case: Not a BlockArgument
    if (!node.isa<BlockArgument>()) {
      // Any Value that is derived from a Register must be ignored. `skip` is
      // true, only if this node is the result of some Register.
      bool skip =
          TypeSwitch<Operation *, bool>(node.getDefiningOp())
              // Case: Instance port: Add the port fields that have a
              // combinational path in the referenced module, to the children.
              .Case<InstanceOp>([&](InstanceOp ins) {
                // Get the port index for `node`.
                size_t portNum = node.cast<mlir::OpResult>().getResultNumber();
                FModuleOp referencedModule = dyn_cast<FModuleOp>(
                    *instanceGraph.getReferencedModule(ins));
                if (referencedModule) {
                  auto port =
                      referencedModule.getBodyBlock()->getArgument(portNum);
                  // Get all the ports that have a comb path from `port`.
                  for (const auto &portNode : portPaths[port])
                    for (auto outputPort : portNode.getSecond()) {
                      Value instResult = ins.getResult(
                          outputPort.val.cast<BlockArgument>().getArgNumber());
                      if (!instResult.getType()
                               .cast<FIRRTLBaseType>()
                               .isGround()) {

                        size_t outPortId;
                        if (!getStorageLocIdFromVal(instResult, outPortId))
                          continue;
                        if (!getValueFromStorageLocId(
                                outPortId + outputPort.fieldId, instResult))
                          continue;
                      }
                      children.push_back({instResult, ins});
                    }
                }
                return false;
              })
              .Case<RegOp, RegResetOp>([&](auto) {
                children.clear();
                return true;
              })
              .Case<SubfieldOp, SubindexOp, SubaccessOp>([&](auto sub) {
                // If the subfield result is not recorded then it must be
                // ignored. Only aggregate port and wire ops are recorded.
                if (valToStorageLocIdMap.find(node) ==
                    valToStorageLocIdMap.end()) {
                  children.clear();
                  // This subfield is derived from some Register.
                  return true;
                }
                return false;
              })
              .Default([&](auto) { return false; });
      if (skip)
        return;
    }

    // Now, handle the ops which create an explicit comb path from `node` to
    // another Value.
    for (auto &child : node.getUses()) {
      Operation *owner = child.getOwner();
      LLVM_DEBUG(llvm::dbgs() << "\n owner: " << *owner);
      auto childVal =
          TypeSwitch<Operation *, Value>(owner)
              .Case<FConnectLike>([&](FConnectLike connect) -> Value {
                // Ignore connects, where the child is the destination.
                if (child.getOperandNumber() == 0)
                  return {};
                // Get the aggregate base type Value, if the dest is a subfield
                // of it. Returns an invalid FieldRef, if this Value can be
                // ignored.
                auto fRef = getFieldRef(connect.getDest());

                if (!fRef.getValue())
                  return {};
                if (inputArg.isValid())
                  if (auto sinkOut = fRef.getValue().dyn_cast<BlockArgument>())
                    // Record the combinational path that was discovered between
                    // the ports of this module.
                    portPaths[inputArg.val][inputArg.fieldId].push_back(
                        Node(sinkOut, fRef.getFieldID()));

                return connect.getDest();
              })
              .Case<RegOp, RegResetOp, SubaccessOp>(
                  [&](auto reg) -> Value { return {}; })
              .Default([&](Operation *op) -> Value {
                if (op->getNumResults() == 1)
                  return op->getResult(0);

                return {};
              });
      if (childVal) {
        children.push_back({childVal, owner});
      }
    }
  }

  /// Record all the leaf ground type values for an aggregate type value. The
  /// Optional `fieldIdsToAdd` can be used to only add field ids that exist in
  /// the set and ignore all other fields. This is used for MemOp ports, and it
  /// will only contain 3 elements, so the copy shouldn't be a concern.
  void gatherAggregateLeafs(Value val,
                            SmallVector<std::pair<Value, Node>> &rootValues,
                            llvm::SmallDenseSet<size_t> fieldIdsToAdd = {}) {
    auto valType = val.getType().dyn_cast<FIRRTLBaseType>();
    // Ignore RefType and GroundType Values.
    if (!valType || valType.isGround())
      return;
    Node inputArg = Node();
    if (val.isa<BlockArgument>())
      inputArg = Node(val, 0);
    SmallVector<Node> worklist;
    worklist.push_back(Node(val, 0));
    // Assign a unique storage location for the `val`, which is the aggregate
    // base Value.
    auto baseStorageLocId = maxStorageLocId;
    valToStorageLocIdMap[val] = {baseStorageLocId};
    storageLocIdToValuesMap[baseStorageLocId] = {val};
    // Reserve unique storage location ids for all the subfields of `val`.
    maxStorageLocId += valType.getMaxFieldID() + 1;
    auto handleSub = [&](Value res, size_t fieldId) {
      if (res.getType().cast<FIRRTLBaseType>().isGround()) {
        if (!fieldIdsToAdd.empty())
          if (!fieldIdsToAdd.contains(fieldId))
            return;
        // Compute the storage location id based on the fieldIndex.
        auto storageLocId = baseStorageLocId + fieldId;
        // The results of subaccess op can refer to multiple storage locations.
        valToStorageLocIdMap[res].push_back(storageLocId);
        storageLocIdToValuesMap[storageLocId].push_back(res);
        if (!leafToBaseValMap.count(res)) {
          leafToBaseValMap[res] = val;
          if (inputArg.isValid())
            inputArg.fieldId = fieldId;
          rootValues.push_back({res, inputArg});
        }
      } else
        worklist.push_back(Node(res, fieldId));
    };
    // Traverse the IR, to get all the lowered fields of the aggregate type.
    // This representation is equivalent to the LowerTypes transformation.
    while (!worklist.empty()) {
      Node node = worklist.pop_back_val();
      for (auto &use : node.val.getUses()) {
        Operation *op = use.getOwner();
        TypeSwitch<Operation *>(op)
            .Case<SubfieldOp>([&](SubfieldOp sub) {
              auto bundleType = sub.getInput().getType();
              auto fId = bundleType.getFieldID(sub.getFieldIndex());
              auto res = sub.getResult();
              handleSub(res, node.fieldId + fId);
            })
            .Case<SubindexOp>([&](SubindexOp sub) {
              auto vecType = sub.getInput().getType().cast<FVectorType>();
              auto res = sub.getResult();
              auto fId = vecType.getFieldID(sub.getIndex());
              handleSub(res, node.fieldId + fId);
            })
            .Case<SubaccessOp>([&](SubaccessOp sub) {
              auto vecType = sub.getInput().getType().cast<FVectorType>();
              auto res = sub.getResult();
              // The result of a subaccess can refer to multiple storage
              // locations corresponding to all the possible indices.
              for (size_t index = 0; index < vecType.getNumElements(); ++index)
                handleSub(
                    res,
                    node.fieldId + 1 +
                        index * (vecType.getElementType().getMaxFieldID() + 1));
            })
            .Default([&](auto op) {});
      }
    }
  }

  bool getValueFromStorageLocId(size_t storageLocId, Value &val) {
    auto iter = storageLocIdToValuesMap.find(storageLocId);
    if (iter == storageLocIdToValuesMap.end())
      return false;
    if (iter->getSecond().empty())
      return false;
    val = iter->getSecond().front();
    return true;
  }

  bool getStorageLocIdFromVal(Value val, size_t &storageLocId) {
    auto iter = valToStorageLocIdMap.find(val);
    if (iter == valToStorageLocIdMap.end())
      return false;
    if (iter->getSecond().empty())
      return false;
    storageLocId = iter->getSecond().front();
    return true;
  }

  LogicalResult processModule() {
    // The list of Values, from which the DFS traversal can start, and the
    // second element of the pair indicates if the Value is an input port.
    SmallVector<std::pair<Value, Node>> rootValues;
    LLVM_DEBUG(llvm::dbgs() << "\n processing module :" << module.getName());
    // Given any Value `val`, add it to `rootValues`, if it is Ground Type. Else
    // traverse the IR, to discover all the ground type leaf sub fields of
    // `val`.
    auto addRoots = [&](Value val) {
      auto type = val.getType().dyn_cast<FIRRTLBaseType>();
      if (!type)
        return;
      if (type.isGround()) {
        Node inputArg = Node();
        if (val.isa<BlockArgument>())
          inputArg = Node(val, 0);
        rootValues.push_back({val, inputArg});
        return;
      }
      // Discover all the ground type leaf subfields.
      gatherAggregateLeafs(val, rootValues);
    };
    // Gather all the Ground type Values which can serve as the root for a DFS
    // traversal. First use ports as the seed.
    for (auto arg : module.getArguments())
      addRoots(arg);

    // Then use Wires, Instances and Memory results as the seed.
    for (auto &op : module.getOps()) {
      TypeSwitch<Operation *>(&op)
          .Case<WireOp>([&](WireOp wire) { addRoots(wire.getResult()); })
          .Case<InstanceOp>([&](InstanceOp ins) {
            FModuleOp referencedModule =
                dyn_cast<FModuleOp>(*instanceGraph.getReferencedModule(ins));
            for (auto instPort : op.getResults())
              addRoots(instPort);
            for (const auto &instPort : llvm::enumerate(op.getResults())) {
              auto portType =
                  instPort.value().getType().dyn_cast<FIRRTLBaseType>();
              if (!portType || portType.isGround())
                continue;
              size_t inPortBaseId;
              if (!getStorageLocIdFromVal(instPort.value(), inPortBaseId))
                continue;
              if (referencedModule) {
                auto port = referencedModule.getBodyBlock()->getArgument(
                    instPort.index());
                // Get all the ports that have a comb path from `port`.
                for (const auto &portNode : portPaths[port]) {
                  for (auto outputPort : portNode.getSecond()) {
                    auto instResult = ins.getResult(
                        outputPort.val.cast<BlockArgument>().getArgNumber());
                    size_t outPortId;
                    if (!getStorageLocIdFromVal(instResult, outPortId))
                      continue;
                    Value outLeafVal;
                    if (!getValueFromStorageLocId(
                            outPortId + outputPort.fieldId, outLeafVal))
                      continue;
                    storageLocIdToValuesMap[inPortBaseId + portNode.first]
                        .push_back(outLeafVal);
                  }
                }
              }
            }
          })
          .Case<MemOp>([&](MemOp mem) {
            if (!(mem.getReadLatency() == 0)) {
              return;
            }
            for (auto memPort : mem.getResults()) {
              if (!memPort.getType().isa<FIRRTLBaseType>())
                continue;
              auto type = memPort.getType().cast<BundleType>();
              auto enableFieldId =
                  type.getFieldID((unsigned)ReadPortSubfield::en);
              auto dataFieldId =
                  type.getFieldID((unsigned)ReadPortSubfield::data);
              auto addressFieldId =
                  type.getFieldID((unsigned)ReadPortSubfield::addr);
              llvm::SmallDenseSet<size_t> fieldIdsToAdd = {
                  enableFieldId, dataFieldId, addressFieldId};
              gatherAggregateLeafs(memPort, rootValues, fieldIdsToAdd);

              size_t baseId;
              if (!getStorageLocIdFromVal(memPort, baseId))
                continue;
              Value dataField;
              if (!getValueFromStorageLocId(baseId + dataFieldId, dataField))
                continue;
              storageLocIdToValuesMap[baseId + enableFieldId].push_back(
                  dataField);
              storageLocIdToValuesMap[baseId + addressFieldId].push_back(
                  dataField);
            }
          })
          .Default([&](auto) {});
    }
    LLVM_DEBUG({
      for (auto leaf : storageLocIdToValuesMap) {
        llvm::dbgs() << "\n Valueindex :" << leaf.first;
        for (auto val : leaf.second)
          llvm::dbgs() << "\n leaf:" << val;
      }
      for (const auto &i1 : valToStorageLocIdMap) {
        llvm::dbgs() << "\n val : " << i1.first;
        for (const auto &i2 : i1.second) {
          llvm::dbgs() << "\n indices :" << i2;
        }
      }
    });

    auto traverseDFSFrom = [&](Value root, Node inputArg) -> LogicalResult {
      if (visited.contains(root))
        return success();
      return dfsFromNode(root, inputArg);
    };
    for (auto root : rootValues) {
      if (root.second.isValid())
        visited.clear();
      if (traverseDFSFrom(root.first, root.second).failed())
        return failure();
    }

    LLVM_DEBUG({
      for (const auto &i1 : portPaths)
        for (const auto &i2 : i1.second)
          for (const auto &i3 : i2.second)
            llvm::dbgs() << "\n node :" << i1.first << "," << i2.first
                         << " is connected to " << i3.val << "," << i3.fieldId;
    });
    return success();
  }

  void printError(Value node, ValueOpPair &childNode) {
    if (!printCycle || currentPath.back() == node) {
      printCycle = false;
      return;
    }
    if (childNode.second)
      TypeSwitch<Operation *>(childNode.second)
          .Case<InstanceOp>([&](InstanceOp ins) {
            size_t portNum = 0, childPortNum = 0;
            // Get the port index for `node.val`.
            for (const auto &p : llvm::enumerate(ins->getOpResults())) {
              if (p.value() == childNode.first)
                childPortNum = p.index();
              if (p.value() == node)
                portNum = p.index();
            }
            auto remark = ins.emitRemark("instance is part of a combinational "
                                         "cycle, instance port number '")
                          << childPortNum << "' has a path from port number '"
                          << portNum << "'";
            FieldRef fOut = getFieldRef(node);
            FieldRef fIn = getFieldRef(childNode.first);
            auto inName = getFieldName(fIn).first;
            auto outName = getFieldName(fOut).first;
            remark << ", " << inName << " <- " << outName;
          })
          .Case<MemOp>([&](MemOp mem) {
            mem.emitRemark("memory is part of a combinational cycle");
          })
          .Case<SubfieldOp, SubindexOp, SubaccessOp>([&](auto op) {})
          .Default([&](auto owner) {
            owner->emitRemark(
                "this operation is part of the combinational cycle");
          });
  }

  FModuleOp module;
  InstanceGraph &instanceGraph;
  // Set of visiting and visited nodes.
  llvm::SmallDenseSet<Value> visiting, visited;
  // Combinational paths between the ports of a module.
  llvm::SmallDenseMap<Value, llvm::SmallDenseMap<size_t, SmallVector<Node>>>
      &portPaths;
  // Map of each unique storage location id and the list of SSA values that
  // refer to it.
  llvm::DenseMap<size_t, SmallVector<Value>> storageLocIdToValuesMap;
  // Map of an SSA value to the list of storage location ids, that it can refer
  // to.
  llvm::DenseMap<Value, SmallVector<size_t>> valToStorageLocIdMap;
  // For each leaf value, record the aggregate base value that it corresponds
  // to.
  llvm::DenseMap<Value, Value> leafToBaseValMap;
  size_t maxStorageLocId = 0;
  SmallVector<Value> currentPath;
  bool printCycle = true;
  bool detectedCycle = false;
};

/// This pass constructs a local graph for each module to detect combinational
/// cycles. To capture the cross-module combinational cycles, this pass inlines
/// the combinational paths between IOs of its subinstances into a subgraph and
/// encodes them in a `combPathsMap`.
class CheckCombLoopsPass : public CheckCombLoopsBase<CheckCombLoopsPass> {
public:
  void runOnOperation() override {
    auto &instanceGraph = getAnalysis<InstanceGraph>();
    llvm::SmallDenseMap<Value, llvm::SmallDenseMap<size_t, SmallVector<Node>>>
        portPaths;
    // Traverse modules in a post order to make sure the combinational paths
    // between IOs of a module have been detected and recorded in `portPaths`
    // before we handle its parent modules.
    for (auto *igNode : llvm::post_order<InstanceGraph *>(&instanceGraph)) {
      if (auto module = dyn_cast<FModuleOp>(*igNode->getModule())) {
        ModuleDFS m(module, instanceGraph, portPaths);
        if (m.processModule().failed())
          return signalPassFailure();
      }
    }
    markAllAnalysesPreserved();
  }
};

std::unique_ptr<mlir::Pass> circt::firrtl::createCheckCombLoopsPass() {
  return std::make_unique<CheckCombLoopsPass>();
}
