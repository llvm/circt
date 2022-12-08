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
    return (val == rhs.val &&
           fieldId == rhs.fieldId);
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
  ModuleDFS(FModuleOp module, InstanceGraph &instanceGraph,
            llvm::MapVector<Node, SmallVector<Node>> &portPaths)
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
      FieldRef f(node.val, node.fieldId);
      auto signalName = getFieldName(f);
      auto errorDiag = mlir::emitError(
          module.getLoc(),
          "detected combinational cycle in a FIRRTL module, sample path: ");
      if (!signalName.empty()) {
        errorDiag << module.getName() << "." << signalName << " <- ";
      }
      for (auto n : llvm::reverse(currentPath)) {
        FieldRef f(n.val, n.fieldId);
        auto signalName = getFieldName(f);
        if (!signalName.empty())
          errorDiag << module.getName() << "." << signalName << " <- ";
        if (n == node)
          break;
      }
      currentPath.push_back(node);
      return failure();
    }
    visiting.insert(node);
    currentPath.push_back(node);
    SmallVector<NodeOpPair> children;
    getChildren(node, inputArg, children);
    for (auto childNode : children)
      if (dfsFromNode(childNode.first, inputArg).failed()) {
        printError(node, childNode);
        return failure();
      }

    visiting.erase(node);
    currentPath.pop_back();
    // Finished discovering all the reachable nodes from node.
    visited.insert(node);
    return success();
  }

  /// Traverse all the reachable paths from InstanceOp, MemOp, RegOp,
  /// RegResetOps. Return true only if the DFS can be terminated at the `node`.
  /// Traversal can be terminated after all the valid children of the `node` are
  /// visited. Ignore register ops, for InstanceOps get the cominational paths
  /// between the module ports, for MemOps check for any combinational paths
  /// between the input addres or enable fields and the output data field for
  /// latency 0 read ports.
  void getChildren(Node &node, Node &inputArg,
                   SmallVector<NodeOpPair> &children) {
    if (!node.val.getType().cast<FIRRTLBaseType>().isGround()) {
      auto baseVal = node.val;
      size_t fieldId = node.fieldId;
      if (node.fieldId == 0) {
        // Get the base aggregate value from the leaf.
        auto fRef = getFieldRefFromValue(node.val);
        baseVal = fRef.getValue();
        fieldId = fRef.getFieldID();
      }
      if (!valLeafOps.count(baseVal))
        gatherAggregateLeafs(baseVal);
      // Get all the leaf values, that correspond to the `fieldId` leaf of the
      // `baseVal`. This is required, because, mulitple subfield ops can refer
      // to the same field.
      for (auto &leaf : valLeafOps[baseVal][fieldId]) {
        children.push_back(
            std::make_pair<Node, Operation *>(Node(leaf, 0), leaf.getDefiningOp()));
      }
    } else
      for (auto &child : node.val.getUses()) {
        Operation *owner = child.getOwner();
        LLVM_DEBUG(llvm::dbgs() << "\n owner: " << *owner);
        auto childNode =
            TypeSwitch<Operation *, Node>(owner)
                .Case<FConnectLike>([&](FConnectLike connect) {
                  if (child.getOperandNumber() == 1) {
                    auto fRef = getFieldRefFromValue(connect.getDest());
                    if (inputArg.isValid()) {
                      if (auto sinkOut =
                              fRef.getValue().dyn_cast<BlockArgument>())
                        portPaths[Node(inputArg.val, inputArg.fieldId)]
                            .push_back(Node(sinkOut, fRef.getFieldID()));
                    }
                    return Node(fRef.getValue(), fRef.getFieldID());
                  }
                  return Node();
                })
                .Default([&](Operation *op) {
                  if (op->getNumResults() == 1) {
                    auto res = op->getResult(0);
                    auto fRef = getFieldRefFromValue(res);
                    return Node(fRef.getValue(), fRef.getFieldID());
                  }
                  return Node();
                });
        if (childNode.isValid())
          children.push_back({childNode, owner});
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
            for (const auto &portNode : portPaths[Node(port, node.fieldId)]) {
              auto instResult = ins.getResult(
                  portNode.val.cast<BlockArgument>().getArgNumber());
              Node instNode(instResult, portNode.fieldId);
              children.push_back({instNode, ins});
            }
          }
          return;
        })
        .Case<RegOp, RegResetOp>([&](auto) { children.clear(); })
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
  void gatherAggregateLeafs(Value val) {
    if (val.getType().dyn_cast<FIRRTLBaseType>().isGround())
      return;
    SmallVector<Node> worklist;
    worklist.push_back(Node(val, 0));
    auto handleSub = [&](Value res, size_t index, bool isGround) {
      if (isGround)
        valLeafOps[val][index].push_back(res);
      else
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
              for (size_t index = 0 ; index < vecType.getNumElements() ; ++index)
                handleSub(res, node.fieldId  + 1 + index *(vecType.getElementType().getMaxFieldID()+1), vecType.getElementType().isGround());
                })
            .Default([&](auto op) {});
      }
    }
    LLVM_DEBUG({
      for (auto leafs : valLeafOps) {
        if (leafs.first != val)
          continue;
        for (auto sec : leafs.second) {
          llvm::dbgs() << "\n node :" << leafs.first << "," << sec.first;
          for (auto leaf : sec.second) {
            llvm::dbgs() << "\n leaf:" << leaf;
            auto fRef = getFieldRefFromValue(leaf);
            llvm::dbgs() << "\n getFieldRefFromValue:" << fRef.getValue()
                         << "::" << fRef.getFieldID();
          }
        }
      }
    });
  }

  LogicalResult processModule() {
    auto visitVal = [&](Value val, bool isArg = true) -> LogicalResult {
      auto valType = val.getType().dyn_cast<FIRRTLBaseType>();
      if (!valType)
        return success();
      // List of root nodes to begin the DFS from.
      SmallVector<Node> rootNodes;

      if (valType.isGround())
        rootNodes.push_back(Node(val, 0));
      else {
        // If aggregate type value, then each ground type leaf value is a root
        // for traversal.
        gatherAggregateLeafs(val);
        for (auto &leafIter : valLeafOps[val])
          if (!leafIter.second.empty()) {
            rootNodes.push_back(Node(val, leafIter.first));
          }
      }
      while (!rootNodes.empty()) {
        auto node = rootNodes.pop_back_val();

        if (visited.contains(node))
          continue;
        if (dfsFromNode(node, isArg ? node : Node()).failed()) {
          if (isArg) {
            FieldRef f(node.val, node.fieldId);
            auto argName = getFieldName(f);
            return node.getParentModule().emitRemark(
                       "this operation is part of the combinational cycle, "
                       "module argument '")
                   << argName << "'";
          }
          return node.val.getDefiningOp()->emitRemark(
              "this operation is part of the combinational cycle");
        }
      }
      return success();
    };
    LLVM_DEBUG(llvm::dbgs() << "\n processing module :" << module);
    // First discover all the combinational paths originating from the ports.
    for (auto arg : module.getArguments()) {
      if (visitVal(arg).failed())
        return failure();
    }
    for (auto &op : module.getOps())
      // Discover combinational paths originating at a wire.
      if (isa<WireOp>(op))
        if (visitVal(op.getResult(0), false).failed())
          return failure();

    LLVM_DEBUG({
      for (const auto &i1 : portPaths)
        for (const auto &i2 : i1.second)
          llvm::dbgs() << "\n node :" << i1.first.val << "," << i1.first.fieldId
                       << " is connected to ," << i2.val << "," << i2.fieldId;
    });
    return success();
  }

  void printError(Node &node, NodeOpPair &childNode) {
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
          .Default([&](auto owner) {
            owner->emitRemark(
                "this operation is part of the combinational cycle");
          });
    if (!printCycle || currentPath.back() == node) {
      printCycle = false;
      return;
    }
  }

  FModuleOp module;
  InstanceGraph &instanceGraph;
  // Set of visiting and visited nodes.
  llvm::SmallDenseSet<Node> visiting, visited;
  // Combinational paths between the ports of a module.
  llvm::MapVector<Node, SmallVector<Node>> &portPaths;
  // Map of all the leaf ground type values for a corresponding aggregate value.
  llvm::MapVector<Value, llvm::MapVector<size_t, SmallVector<Value>>> valLeafOps;
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
    llvm::MapVector<Node, SmallVector<Node>> portPaths;
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
