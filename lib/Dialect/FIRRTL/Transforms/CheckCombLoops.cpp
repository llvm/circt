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
  Value leafVal = nullptr;
  Node() = default;
  Node(Value val, size_t fieldId, Value leafVal = nullptr)
      : val(val), fieldId(fieldId), leafVal(leafVal) {}
  Node(const Node &rhs) = default;
  Node &operator=(const Node &rhs) = default;
  bool operator==(const Node &rhs) const {
    return val == rhs.val &&
           (fieldId == rhs.fieldId || fieldId == 0 || rhs.fieldId == 0);
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
public:
  ModuleDFS(FModuleOp module, InstanceGraph &instanceGraph,
            llvm::SmallDenseMap<Node, SmallVector<Node>> &portPaths)
      : module(module), instanceGraph(instanceGraph), portPaths(portPaths) {}

  LogicalResult dfsFromNode(Node &node, Node inputArg) {

    LLVM_DEBUG(llvm::dbgs()
               << "\n visiting node :" << node.val << "," << node.fieldId);
    if (!node.val.getType().isa<FIRRTLBaseType>())
      return success();
    if (visited.contains(node))
      return success();
    // visiting: Set of nodes that have a path to this current node.
    if (visiting.contains(node))
      return node.getParentModule().emitError(
          "detected combinational cycle in a FIRRTL module '");

    visiting.insert(node);
    // Case 1: Instance and Mem ops.
    if (!handleDefOps(node, inputArg)) {
      // Case 2: Handle aggregate values.
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
          auto leafNode = Node(leaf, 0);
          if (dfsFromNode(leafNode, inputArg).failed()) {
            if (auto nodeType = baseVal.getType().dyn_cast<BundleType>()) {
              Operation *errOp;
              if (baseVal.isa<BlockArgument>())
                errOp = module;
              else
                errOp = baseVal.getDefiningOp();
              if (!isa<MemOp>(errOp))
                errOp->emitRemark()
                    << "this operation's field '"
                    << nodeType.getElementName(
                           nodeType.getIndexForFieldID(node.fieldId))
                    << "' is part of the combinational cycle";
            }
            return failure();
          }
        }
      } // Case 3: Handle ground type values and iterate over all the users.
      else if (visitChildren(node, inputArg).failed())
        return failure();
    }
    visiting.erase(node);
    // Finished discovering all the reachable nodes from node.
    visited.insert(node);
    return success();
  }

  /// Traverse all the reachable paths from InstanceOp and MemOps.
  bool handleDefOps(Node &node, Node &inputArg) {
    if (node.val.isa<BlockArgument>())
      return false;

    return TypeSwitch<Operation *, bool>(node.val.getDefiningOp())
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
            // Get all the ports that have a combination path from `port`.
            for (const auto &portNode : portPaths[Node(port, node.fieldId)]) {
              auto instResult = ins.getResult(
                  portNode.val.cast<BlockArgument>().getArgNumber());
              Node instNode(instResult, portNode.fieldId);
              if (dfsFromNode(instNode, inputArg).failed()) {
                auto remark =
                    ins.emitRemark("instance is part of a combinational "
                                   "cycle, instance port number '")
                    << portNode.val.cast<BlockArgument>().getArgNumber()
                    << "' has a path to port number '" << portNum << "'"
                    << " " << ins.getName() << "."
                    << ins.getPortName(
                              portNode.val.cast<BlockArgument>().getArgNumber())
                           .getValue();
                if (auto nodeType =
                        portNode.val.getType().dyn_cast<BundleType>()) {

                  remark << "."
                         << nodeType.getElementName(
                                nodeType.getIndexForFieldID(portNode.fieldId));
                }
                remark << " --> " << ins.getName() << "."
                       << ins.getPortName(portNum).getValue();
                if (auto nodeType = node.val.getType().dyn_cast<BundleType>()) {
                  remark << "."
                         << nodeType.getElementName(
                                nodeType.getIndexForFieldID(node.fieldId));
                }
                return true;
              }
            }
          }
          return false;
        })
        .Case<RegOp, RegResetOp>([&](auto) { return true; })
        .Case<MemOp>([&](MemOp mem) {
          auto type = node.val.getType().cast<BundleType>();
          auto enableFieldId = type.getFieldID(1);
          auto dataFieldId = type.getFieldID(3);
          size_t addressFieldId = 1;
          // If this is the enable or addr field of the memory read port.
          if (mem.getReadLatency() == 0 &&
              (node.fieldId == addressFieldId ||
               node.fieldId == enableFieldId || node.fieldId == dataFieldId))
            for (const auto &memPort : llvm::enumerate(mem.getResults())) {
              // If this is a read port.
              if (memPort.value() != node.val ||
                  mem.getPortKind(memPort.index()) != MemOp::PortKind::Read)
                continue;
              // If its a data field, then continue DFS from the data
              // fields.
              if (node.fieldId == dataFieldId)
                return false;
              // Check for possible combinational paths from 0-latency
              // memory reads to the input address and enable.
              auto dataNode = Node(node.val, type.getFieldID(3));

              if (dfsFromNode(dataNode, inputArg).failed()) {
                mem.emitRemark("memory is part of a combinational cycle "
                               "between the data and ")
                    << (node.fieldId == 1 ? "address" : "enable") << " port";
                return true;
              }
            }
          // For any other kind of memory, terminate the DFS.
          return true;
        })
        .Default([&](auto) { return false; });
  }

  LogicalResult visitChildren(Node &node, Node &inputArg) {
    for (auto &child : node.val.getUses()) {
      Operation *owner = child.getOwner();
      LLVM_DEBUG(llvm::dbgs() << "\n owner: " << *owner);
      Node childNode =
          TypeSwitch<Operation *, Node>(owner)
              .Case<FConnectLike>([&](FConnectLike connect) {
                if (child.getOperandNumber() == 1) {
                  auto fRef = getFieldRefFromValue(connect.getDest());
                  if (inputArg.isValid()) {
                    if (auto sinkOut =
                            fRef.getValue().dyn_cast<BlockArgument>())
                      portPaths[Node(inputArg.val, inputArg.fieldId)].push_back(
                          Node(sinkOut, fRef.getFieldID()));
                  }
                  return Node(fRef.getValue(), fRef.getFieldID(),
                              connect.getDest());
                }
                return Node();
              })
              .Default([&](Operation *op) {
                if (op->getNumResults() == 1) {
                  auto res = op->getResult(0);
                  auto fRef = getFieldRefFromValue(res);
                  return Node(fRef.getValue(), fRef.getFieldID(), res);
                }
                return Node();
              });
      if (childNode.isValid())
        if (dfsFromNode(childNode, inputArg).failed()) {
          owner->emitRemark(
              "this operation is part of the combinational cycle");
          return failure();
        }
    }
    return success();
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
            .Default([&](auto op) {});
      }
    }
    LLVM_DEBUG({
      for (auto leafs : valLeafOps) {
        if (leafs.first != val)
          continue;
        for (auto sec : leafs.second) {
          llvm::dbgs() << "\n node :" << leafs.first << "," << sec.getFirst();
          for (auto leaf : sec.getSecond()) {
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
        rootNodes.emplace_back(Node(val, 0));
      else {
        // If aggregate type value, then each ground type leaf value is a root
        // for traversal.
        gatherAggregateLeafs(val);
        for (auto leafIter : valLeafOps[val])
          if (!leafIter.getSecond().empty())
            rootNodes.emplace_back(
                Node(val, leafIter.getFirst(), leafIter.getSecond().front()));
      }
      while (!rootNodes.empty()) {
        auto node = rootNodes.pop_back_val();

        if (visited.contains(node))
          continue;
        if (dfsFromNode(node, isArg ? node : Node()).failed()) {
          if (isArg)
            return node.getParentModule().emitRemark(
                       "this operation is part of the combinational cycle, "
                       "module argument ")
                   << node.val.cast<BlockArgument>().getArgNumber();
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
                       << "," << i1.first.leafVal << " is connected to ,"
                       << i2.val << "," << i2.fieldId;
    });
    return success();
  }

  FModuleOp module;
  InstanceGraph &instanceGraph;
  // Set of visiting and visited nodes.
  llvm::SmallDenseSet<Node> visiting, visited;
  // Combinational paths between the ports of a module.
  llvm::SmallDenseMap<Node, SmallVector<Node>> &portPaths;
  // Map of all the leaf ground type values for a corresponding aggregate value.
  DenseMap<Value, DenseMap<size_t, SmallVector<Value>>> valLeafOps;
};

/// This pass constructs a local graph for each module to detect combinational
/// cycles. To capture the cross-module combinational cycles, this pass inlines
/// the combinational paths between IOs of its subinstances into a subgraph and
/// encodes them in a `combPathsMap`.
class CheckCombLoopsPass : public CheckCombLoopsBase<CheckCombLoopsPass> {
public:
  void runOnOperation() override {
    auto &instanceGraph = getAnalysis<InstanceGraph>();
    llvm::SmallDenseMap<Node, SmallVector<Node>> portPaths;
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
