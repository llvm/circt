//===- CheckCombLoops.cpp - FIRRTL check combinational cycles --*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements the FIRRTL combintational cycles detection pass.
//
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
//
//===----------------------------------------------------------------------===//
// DenseMapInfo on Node
//===----------------------------------------------------------------------===//

namespace llvm {
template <>
struct DenseMapInfo<Node> {
  static Node getEmptyKey() {
    auto *pointer = llvm::DenseMapInfo<void *>::getEmptyKey();
    return Node(Value::getFromOpaquePointer(pointer), 0);
  }
  static Node getTombstoneKey() {
    auto *pointer = llvm::DenseMapInfo<void *>::getTombstoneKey();
    return Node(Value::getFromOpaquePointer(pointer), 0);
  }
  static unsigned getHashValue(const Node &node) {
    return detail::combineHashValue(mlir::hash_value(node.val),
                                    llvm::hash_value(node.fieldId));
  }
  static bool isEqual(const Node &lhs, const Node &rhs) { return lhs == rhs; }
};
} // namespace llvm

class ModuleDFS {
  using NodeOpPair = std::pair<Node, Operation *>;

public:
  ModuleDFS(
      FModuleOp module, InstanceGraph &instanceGraph,
      llvm::SmallDenseMap<Value, llvm::SmallDenseMap<size_t, SmallVector<Node>>>
          &portPaths)
      : module(module), instanceGraph(instanceGraph), portPaths(portPaths) {}

  LogicalResult dfsFromNode(Node &node, Node inputArg) {

    LLVM_DEBUG(llvm::dbgs()
               << "\n visiting node :" << node.val << "," << node.fieldId);
    if (!node.val.getType().isa<FIRRTLBaseType>())
      return success();
    if (visited.contains(node))
      return success();
    // visiting: Set of nodes that have a path to this current node.
    if (visiting.contains(node)) {
      FieldRef f = getFieldRefFromValue(node.val);
      auto lastSignalName = getFieldName(f);
      auto errorDiag = mlir::emitError(
          module.getLoc(),
          "detected combinational cycle in a FIRRTL module, sample path: ");
      if (!lastSignalName.empty()) {
        errorDiag << module.getName() << "." << lastSignalName << " <- ";
      }
      for (auto n : llvm::reverse(currentPath)) {
        FieldRef f = getFieldRefFromValue(n.val);
        auto signalName = getFieldName(f);
        if (!signalName.empty() && lastSignalName != signalName)
          errorDiag << module.getName() << "." << signalName << " <- ";
        lastSignalName = signalName;
        if (n == node)
          break;
      }
      currentPath.push_back(node);
      return failure();
    }
    llvm::SmallDenseSet<Node> tmpCopy = visiting;
    visiting.insert(node);
    currentPath.push_back(node);
    SmallVector<NodeOpPair> children;
    auto groupIter = valToGroupIdMap.find(node.val);
    if (groupIter != valToGroupIdMap.end())
      for (auto groupId : groupIter->getSecond()) {
        for (auto childVal : groupIdToValuesMap[groupId]) {
          if (childVal != node.val) {
            LLVM_DEBUG(llvm::dbgs() << "\n adding child :" << childVal);
            auto childNode = Node(childVal, 0);
            if (!visiting.contains(childNode))
              children.push_back({childNode, childVal.getDefiningOp()});
          }
        }
      }
    getChildren(node, inputArg, children);
    for (auto childNode : children)
      if (dfsFromNode(childNode.first, inputArg).failed()) {
        printError(node, childNode);
        return failure();
      }

    visiting = std::move(tmpCopy);
    currentPath.pop_back();
    // Finished discovering all the reachable nodes from node.
    visited.insert(node);
    return success();
  }

  void getChildren(Node &node, Node &inputArg,
                   SmallVector<NodeOpPair> &children) {

    for (auto &child : node.val.getUses()) {
      Operation *owner = child.getOwner();
      LLVM_DEBUG(llvm::dbgs() << "\n owner: " << *owner);
      auto childVal =
          TypeSwitch<Operation *, Value>(owner)
              .Case<FConnectLike>([&](FConnectLike connect) {
                if (child.getOperandNumber() == 1) {
                  auto fRef = getFieldRefFromValue(connect.getDest());
                  if (inputArg.isValid()) {
                    if (auto sinkOut =
                            fRef.getValue().dyn_cast<BlockArgument>())
                      portPaths[inputArg.val][inputArg.fieldId].push_back(
                          Node(sinkOut, fRef.getFieldID()));
                  }
                  return connect.getDest();
                }
                return (Value) nullptr;
              })
              .Default([&](Operation *op) {
                if (op->getNumResults() == 1) {
                  return (Value)op->getResult(0);
                }
                return (Value) nullptr;
              });
      if (childVal != nullptr) {
        children.push_back({Node(childVal, 0), owner});
      }
    }
    if (node.val.isa<BlockArgument>())
      return;
    return TypeSwitch<Operation *>(node.val.getDefiningOp())
        .Case<InstanceOp>([&](InstanceOp ins) {
          size_t portNum = 0;
          // Get the port index for `node.val`.
          for (const auto &p : llvm::enumerate(ins->getOpResults()))
            if (p.value() == node.val) {
              portNum = p.index();
              break;
            }
          FModuleOp referencedModule =
              dyn_cast<FModuleOp>(*instanceGraph.getReferencedModule(ins));
          if (referencedModule) {
            auto port = referencedModule.getBodyBlock()->getArgument(portNum);
            // Get all the ports that have a comb path from `port`.
            for (const auto &portNode : portPaths[port])
              for (auto outputPort : portNode.getSecond()) {
                auto instResult = ins.getResult(
                    outputPort.val.cast<BlockArgument>().getArgNumber());
                children.push_back({Node(instResult, 0), ins});
              }
          }
          return;
        })
        .Case<RegOp, RegResetOp>([&](auto) { children.clear(); })
        .Case<SubfieldOp, SubindexOp, SubaccessOp>([&](auto sub) {
          if (valToGroupIdMap.find(node.val) == valToGroupIdMap.end())
            children.clear();
        })
        .Case<MemOp>([&](MemOp mem) {
          auto type = node.val.getType().cast<BundleType>();
          auto enableFieldId = type.getFieldID((unsigned)ReadPortSubfield::en);
          auto dataFieldId = type.getFieldID((unsigned)ReadPortSubfield::data);
          size_t addressFieldId =
              type.getFieldID((unsigned)ReadPortSubfield::addr);
          // If this is the enable or addr field of the memory read port.
          if (!(mem.getReadLatency() == 0 && (node.fieldId == addressFieldId ||
                                              node.fieldId == enableFieldId ||
                                              node.fieldId == dataFieldId))) {
            // For any other kind of memory, terminate the DFS.
            children.clear();
            return;
          }
          for (const auto &memPort : llvm::enumerate(mem.getResults())) {
            // If this is a read port.
            if (memPort.value() != node.val ||
                mem.getPortKind(memPort.index()) != MemOp::PortKind::Read)
              continue;
            // If its a data field, then continue DFS from the data
            // fields.
            if (node.fieldId == dataFieldId)
              return;
            // Check for possible combinational paths from 0-latency
            // memory reads to the input address and enable.
            auto dataNode = Node(
                node.val, type.getFieldID((unsigned)ReadPortSubfield::data));
            children.push_back({dataNode, mem});
          }
        })
        .Default([&](auto) { return; });
  }

  /// Record all the leaf ground type values for an aggregate type value.
  void gatherAggregateLeafs(Value val,
                            llvm::SmallDenseSet<size_t> fieldIdsToAdd = {}) {
    auto valType = val.getType().dyn_cast<FIRRTLBaseType>();
    if (!valType || valType.isGround())
      return;
    SmallVector<Node> worklist;
    worklist.push_back(Node(val, 0));
    auto baseGroupId = aliasGroupNum;
    aliasGroupNum += valType.getMaxFieldID() + 1;
    valToGroupIdMap[val] = {baseGroupId};
    groupIdToValuesMap[baseGroupId] = {val};
    auto handleSub = [&](Value res, size_t index, bool isGround) {
      if (isGround) {
        if (!fieldIdsToAdd.empty())
          if (!fieldIdsToAdd.contains(index))
            return;
        auto groupId = baseGroupId + index;
        valToGroupIdMap[res].push_back(groupId);
        groupIdToValuesMap[groupId].push_back(res);
        leafToBaseValMap[res] = val;
      } else
        worklist.push_back(Node(res, index));
    };
    while (!worklist.empty()) {
      Node node = worklist.pop_back_val();
      for (auto &use : node.val.getUses()) {
        Operation *op = use.getOwner();
        TypeSwitch<Operation *>(op)
            .Case<SubfieldOp>([&](SubfieldOp sub) {
              auto bundleType = sub.getInput().getType().cast<BundleType>();
              auto fId = bundleType.getFieldID(sub.getFieldIndex());
              auto res = sub.getResult();
              handleSub(res, node.fieldId + fId, res.getType().isGround());
            })
            .Case<SubindexOp>([&](SubindexOp sub) {
              auto vecType = sub.getInput().getType().cast<FVectorType>();
              auto res = sub.getResult();
              auto fId = vecType.getFieldID(sub.getIndex());
              handleSub(res, node.fieldId + fId, res.getType().isGround());
            })
            .Case<SubaccessOp>([&](SubaccessOp sub) {
              auto vecType = sub.getInput().getType().cast<FVectorType>();
              auto res = sub.getResult();
              for (size_t index = 0; index < vecType.getNumElements(); ++index)
                handleSub(
                    res,
                    node.fieldId + 1 +
                        index * (vecType.getElementType().getMaxFieldID() + 1),
                    vecType.getElementType().isGround());
            })
            .Default([&](auto op) {});
      }
    }
  }

  LogicalResult processModule() {
    SmallVector<Value> rootValues;
    LLVM_DEBUG(llvm::dbgs() << "\n processing module :" << module);
    auto addRoots = [&](Value val, Value inArg = nullptr) {
      auto type = val.getType().dyn_cast<FIRRTLBaseType>();
      if (!type)
        return;
      if (type.isGround()) {
        rootValues.push_back(val);
        return;
      }
      gatherAggregateLeafs(val);
    };
    // First discover all the combinational paths originating from the ports.
    for (auto arg : module.getArguments()) {
      addRoots(arg, arg);
    }
    for (auto &op : module.getOps()) {
      // Discover combinational paths originating at a wire.
      if (isa<WireOp>(op))
        addRoots(op.getResult(0));
      else if (auto ins = dyn_cast<InstanceOp>(op)) {
        FModuleOp referencedModule =
            dyn_cast<FModuleOp>(*instanceGraph.getReferencedModule(ins));
        for (auto instPort : op.getResults())
          addRoots(instPort);
        for (auto instPort : llvm::enumerate(op.getResults())) {
          auto portType = instPort.value().getType().dyn_cast<FIRRTLBaseType>();
          if (!portType || portType.isGround())
            continue;
          auto inPortBaseId = valToGroupIdMap[instPort.value()].front();
          if (referencedModule) {
            auto port =
                referencedModule.getBodyBlock()->getArgument(instPort.index());
            // Get all the ports that have a comb path from `port`.
            for (const auto &portNode : portPaths[port]) {
              for (auto outputPort : portNode.getSecond()) {
                auto instResult = ins.getResult(
                    outputPort.val.cast<BlockArgument>().getArgNumber());
                auto outPortId = valToGroupIdMap[instResult].front();
                groupIdToValuesMap[inPortBaseId + portNode.first].push_back(
                    groupIdToValuesMap[outPortId + outputPort.fieldId].front());
              }
            }
          }
        }
      } else if (auto mem = dyn_cast<MemOp>(op)) {
        if (!(mem.getReadLatency() == 0)) {
          continue;
        }
        for (auto memPort : mem.getResults()) {
          auto type = memPort.getType().cast<BundleType>();
          auto enableFieldId = type.getFieldID((unsigned)ReadPortSubfield::en);
          auto dataFieldId = type.getFieldID((unsigned)ReadPortSubfield::data);
          auto addressFieldId =
              type.getFieldID((unsigned)ReadPortSubfield::addr);
          llvm::SmallDenseSet<size_t> fieldIdsToAdd = {
              enableFieldId, dataFieldId, addressFieldId};
          gatherAggregateLeafs(memPort, fieldIdsToAdd);
          auto baseId = valToGroupIdMap[memPort].front();
          auto dataField = groupIdToValuesMap[baseId + dataFieldId].front();
          groupIdToValuesMap[baseId + enableFieldId].push_back(dataField);
          groupIdToValuesMap[baseId + addressFieldId].push_back(dataField);
        }
      }
    }
    LLVM_DEBUG({
      for (auto leaf : groupIdToValuesMap) {
        llvm::dbgs() << "\n Valueindex :" << leaf.first;
        for (auto val : leaf.second)
          llvm::dbgs() << "\n leaf:" << val;
      }
    });

    auto traverseDFSFrom = [&](Value root, Node inputArg) -> LogicalResult {
      auto node = Node(root, 0);
      if (visited.contains(node))
        return success();
      return dfsFromNode(node, inputArg);
    };
    for (auto root : rootValues) {
      Node inputArg = Node();
      if (root.isa<BlockArgument>())
        inputArg = Node(root, 0);
      if (traverseDFSFrom(root, inputArg).failed())
        return failure();
    }
    for (auto groupIter : groupIdToValuesMap) {
      auto id = groupIter.first;
      auto leaf = groupIter.getSecond().front();
      if (!leaf.getType().cast<FIRRTLBaseType>().isGround())
        continue;
      auto baseVal = leafToBaseValMap[leaf];
      auto baseId = valToGroupIdMap[baseVal].front();
      auto fieldId = id - baseId;
      Node inputArg = Node();
      if (baseVal.isa<BlockArgument>())
        inputArg = Node(baseVal, fieldId);
      if (traverseDFSFrom(leaf, inputArg).failed())
        return failure();
    }

    LLVM_DEBUG({
      for (const auto &i1 : portPaths)
        for (const auto &i2 : i1.second)
          for (const auto &i3 : i2.second)
            llvm::dbgs() << "\n node :" << i1.first << "," << i2.first
                         << " is connected to ," << i3.val << "," << i3.fieldId;
    });
    return success();
  }

  void printError(Node &node, NodeOpPair &childNode) {
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
              if (p.value() == childNode.first.val)
                childPortNum = p.index();
              if (p.value() == node.val)
                portNum = p.index();
            }
            auto remark = ins.emitRemark("instance is part of a combinational "
                                         "cycle, instance port number '")
                          << childPortNum << "' has a path from port number '"
                          << portNum << "'";
            FieldRef fOut(node.val, node.fieldId);
            FieldRef fIn(childNode.first.val, childNode.first.fieldId);
            auto inName = getFieldName(fIn);
            auto outName = getFieldName(fOut);
            remark << ", " << inName << " <- " << outName;
          })
          .Case<MemOp>([&](MemOp mem) {
            mem.emitRemark("memory is part of a combinational cycle ");
          })
          .Case<SubfieldOp, SubindexOp, SubaccessOp>([&](auto op) {})
          .Default([&](auto owner) {
            owner->emitRemark(
                "this operation is part of the combinational cycle");
          });
  }

  using ValueField = std::pair<Value, size_t>;
  DenseMap<Value, SmallVector<ValueField>> valueBaseFieldMap;
  FModuleOp module;
  InstanceGraph &instanceGraph;
  // Set of visiting and visited nodes.
  llvm::SmallDenseSet<Node> visiting, visited;
  // Combinational paths between the ports of a module.
  llvm::SmallDenseMap<Value, llvm::SmallDenseMap<size_t, SmallVector<Node>>>
      &portPaths;
  // Map of all the leaf ground type values for a corresponding aggregate value.
  llvm::DenseMap<size_t, SmallVector<Value>> groupIdToValuesMap;
  llvm::DenseMap<Value, SmallVector<size_t>> valToGroupIdMap;
  llvm::DenseMap<Value, Value> leafToBaseValMap;
  llvm::DenseMap<Node, SmallVector<Value>> nodeToAliasGroupMap;
  llvm::DenseMap<Value, SmallVector<Node>> valueToNodes;
  llvm::DenseMap<Value, llvm::SmallDenseSet<size_t>> aggregateToLeafIds;
  size_t aliasGroupNum = 0;
  SmallVector<Node> currentPath;
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
    // between IOs of a module have been detected and recorded in `combPathsMap`
    // before we handle its parent modules.
    for (auto *igNode : llvm::post_order<InstanceGraph *>(&instanceGraph)) {
      if (auto module = dyn_cast<FModuleOp>(*igNode->getModule())) {
        ModuleDFS m(module, instanceGraph, portPaths);
        if (m.processModule().failed())
          return signalPassFailure();
      }
    }
  }
};

std::unique_ptr<mlir::Pass> circt::firrtl::createCheckCombLoopsPass() {
  return std::make_unique<CheckCombLoopsPass>();
}
