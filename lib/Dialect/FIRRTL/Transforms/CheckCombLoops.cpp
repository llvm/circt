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
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DepthFirstIterator.h"
#include "llvm/ADT/PostOrderIterator.h"
#include "llvm/ADT/SCCIterator.h"
#include "llvm/ADT/SmallSet.h"
#include <variant>

#define DEBUG_TYPE "check-comb-loops"

using namespace circt;
using namespace firrtl;

using SetOfNodes = std::set<FieldRef>;
using Node = FieldRef;
class TarjansSCC {

public:
  TarjansSCC(FModuleOp &module, InstanceGraph &instanceGraph,
             DenseMap<Node, SetOfNodes> &portPaths,
             DenseMap<BlockArgument, std::set<unsigned>> &blockArgToFields)
      : module(module), instanceGraph(instanceGraph), portPaths(portPaths),
        blockArgToFields(blockArgToFields) {}

  LogicalResult processModule() {
    LLVM_DEBUG(llvm::dbgs() << "\n processing module :" << module);
    SmallVector<Value> worklist;
    // Traverse over ports and ops, to populate the worklist and get the
    // FieldRef corresponding to every Value. Also process the InstanceOps and
    // get the paths that exist between the ports of the referenced module.
    preprocess(worklist);

    llvm::DenseSet<Value> visited;
    llvm::DenseSet<Node> visiting;
    SmallVector<Value> visitingStack;
    // Record the discover time and lowLike as per Tarjan's algorithm.
    DenseMap<Value, unsigned> discoveryTime, lowLink;

    // Used to record the time any node is visited for the first time.
    unsigned visitTime = 0;
    // worklist is the list of roots, to begin the traversal from.
    for (auto root : worklist) {
      SmallVector<Value> dfsStack;
      dfsStack.push_back(root);
      SmallVector<Node> inputArgFields;
      LLVM_DEBUG(llvm::dbgs() << "\n Starting traversal from root :"
                              << getFieldName(Node(root, 0)).first);
      if (auto inArg = root.dyn_cast<BlockArgument>()) {
        if (module.getPortDirection(inArg.getArgNumber()) == Direction::In)
          visited.clear();
      }
      while (!dfsStack.empty()) {
        auto dfsVal = dfsStack.back();
        unsigned dfsSize = dfsStack.size();

        LLVM_DEBUG(llvm::dbgs() << "\n Stack pop :"
                                << getFieldName(Node(dfsVal, 0)).first << ","
                                << dfsVal << "::" << dfsStack.size();
                   for (auto visi
                        : visiting) llvm::dbgs()
                   << "\n visiting :" << getFieldName(visi).first << "="
                   << visi.getValue();

        );

        visitingStack.push_back(dfsVal);
        // Record all the children of `dfsVal`.
        SmallVector<Value> children;
        SmallVector<Node> inputArgFieldsTemp;
        DenseSet<Value> aliasingValues = {dfsVal};
        // If `dfsVal` is a subfield, then get all the FieldRefs that it refers
        // to and add them to the visiting set.
        forallRefersTo(dfsVal, [&](Node ref) {
          visiting.insert(ref);
          // If this subfield refers to instance/mem results(input port), then
          // add the output port FieldRefs that exist in the referenced module
          // comb paths to the children.
          handlePorts(ref, children);
          auto ftovIter = fieldToVals.find(ref);
          if (ftovIter != fieldToVals.end())
            for (auto alias : ftovIter->second)
              aliasingValues.insert(alias);
          if (auto arg = ref.getValue().dyn_cast<BlockArgument>()) {
            if (module.getPortDirection(arg.getArgNumber()) == Direction::In)
              inputArgFieldsTemp.push_back(ref);
          }
          return success();
        });
        if (!inputArgFieldsTemp.empty())
          inputArgFields = std::move(inputArgFieldsTemp);

        auto timeIter = discoveryTime.find(dfsVal);
        if (timeIter == discoveryTime.end()) {
          // Node visited for first time, record the discovery time.
          discoveryTime[dfsVal] = visitTime;
          lowLink[dfsVal] = visitTime;
          ++visitTime;
        }
        // Add the Value to `children`, to which a path exists from `dfsVal`.
        for (auto dfsFromVal : aliasingValues) {
          visited.insert(dfsFromVal);

          for (auto &use : dfsFromVal.getUses()) {
            auto owner = use.getOwner();
            Value childVal;
            if (owner->getNumResults() == 1)
              childVal = owner->getResult(0);
            else if (auto connect = dyn_cast<FConnectLike>(owner))
              if (use.getOperandNumber() == 1) {
                childVal = connect.getDest();
                handleConnects(childVal, inputArgFields);
              }
            if (childVal)
              children.push_back(childVal);
          }
        }
        for (auto childVal : children) {
          // This childVal can be ignored, if
          // > It is a Register or a subfield of a register.
          if (childVal.getDefiningOp() &&
              (isa<RegOp, RegResetOp>(childVal.getDefiningOp()) ||
               (isa<SubfieldOp, SubaccessOp, SubindexOp>(
                    childVal.getDefiningOp()) &&
                valRefersTo.find(childVal) == valRefersTo.end())))
            continue;
          if (!visited.contains(childVal))
            dfsStack.push_back(childVal);
          Node childNode;
          bool isBackEdge = false;
          // If the childVal is a sub, then check if it aliases with any of the
          // predecessors (the visiting set).
          forallRefersTo(childVal, [&](Node ref) {
            if (visiting.contains(ref)) {
              childNode = ref;
              isBackEdge = true;
              return failure();
            }
            return success();
          });

          if (isBackEdge) {
            // Comb Cycle Detected !!
            lowLink[dfsVal] = std::min(lowLink[childVal], lowLink[dfsVal]);
            reportLoopFound(childVal, childNode, visiting, visitingStack);
            return failure();
          }
        }
        // child nodes added, continue the DFS
        if (dfsSize != dfsStack.size())
          continue;
        // Node is an SCC root node, pop the visiting stack to remove the nodes
        // that are no longer active predecessors, that is their sub-tree is
        // already explored.
        if (lowLink[dfsVal] == discoveryTime[dfsVal]) {
          // All the Values reachable from `dfsVal` have been explored, remove
          // it and its children from the visiting stack.
          while (visiting.contains(Node(dfsVal, 0))) {
            auto visitingVal = visitingStack.pop_back_val();
            forallRefersTo(visitingVal, [&](Node visitingNode) {
              LLVM_DEBUG(llvm::dbgs() << "\n Erasing visiting : "
                                      << getFieldName(visitingNode).first);
              visiting.erase(visitingNode);
              return success();
            });
          }
        }
        auto popped = dfsStack.pop_back_val();
        LLVM_DEBUG(llvm::dbgs()
                   << "\n dfs popped :" << getFieldName(Node(popped, 0)).first);
      }
    }
    LLVM_DEBUG(
        for (auto vI
             : discoveryTime) {
          LLVM_DEBUG(llvm::dbgs()
                     << "\n val :" << vI.first << ":"
                     << getFieldName(Node(vI.first, 0)).first << "="
                     << vI.second << "," << lowLink[vI.first]);
        } for (auto pp
               : portPaths) {
          llvm::dbgs() << "\n Path from :" << getFieldName(pp.first).first;
          for (auto outport : pp.second)
            llvm::dbgs() << " output :" << getFieldName(outport).first;
        });

    return success();
  }

  // Preprocess the module ops to get the
  // 1. FieldRef corresponding to any Value,
  // 2. roots for DFS traversal,
  void preprocess(SmallVector<Value> &worklist) {
    // All the input ports are added to the worklist.
    for (BlockArgument arg : module.getArguments()) {
      auto argType = arg.getType();
      if (argType.isa<RefType>())
        continue;
      worklist.push_back(arg);
      if (!argType.cast<FIRRTLBaseType>().isGround())
        setValRefsTo(arg, Node(arg, 0));
    }
    DenseSet<Value> memPorts;

    for (auto &op : module.getOps()) {
      TypeSwitch<Operation *>(&op)
          // Wire is added to the worklist
          .Case<WireOp>([&](WireOp wire) {
            worklist.push_back(wire.getResult());
            if (!wire.getType().cast<FIRRTLBaseType>().isGround())
              setValRefsTo(wire.getResult(), Node(wire.getResult(), 0));
          })
          // All sub elements are added to the worklist.
          .Case<SubfieldOp>([&](SubfieldOp sub) {
            auto res = sub.getResult();
            bool isValid = false;
            auto fieldIndex = sub.getFieldIndex() + 1;
            if (memPorts.contains(sub.getInput())) {
              auto memPort = sub.getInput();
              auto type = memPort.getType().cast<BundleType>();
              auto enableFieldId =
                  type.getFieldID((unsigned)ReadPortSubfield::en);
              auto dataFieldId =
                  type.getFieldID((unsigned)ReadPortSubfield::data);
              auto addressFieldId =
                  type.getFieldID((unsigned)ReadPortSubfield::addr);
              memPorts.insert(memPort);
              if (fieldIndex == enableFieldId || fieldIndex == dataFieldId ||
                  fieldIndex == addressFieldId) {
                setValRefsTo(memPort, Node(memPort, 0));
              } else
                return;
            }
            forallRefersTo(
                sub.getInput(),
                [&](Node subBase) {
                  isValid = true;
                  setValRefsTo(res, Node(subBase.getValue(),
                                         subBase.getFieldID() + fieldIndex));
                  return success();
                },
                false);
            if (isValid)
              worklist.push_back(res);
          })
          .Case<SubindexOp>([&](SubindexOp sub) {
            auto res = sub.getResult();
            bool isValid = false;
            auto index = sub.getIndex();
            forallRefersTo(
                sub.getInput(),
                [&](Node subBase) {
                  isValid = true;
                  setValRefsTo(res, Node(subBase.getValue(),
                                         subBase.getFieldID() + index + 1));
                  return success();
                },
                false);
            if (isValid)
              worklist.push_back(res);
          })
          .Case<SubaccessOp>([&](SubaccessOp sub) {
            auto vecType = sub.getInput().getType().cast<FVectorType>();
            auto res = sub.getResult();
            bool isValid = false;
            forallRefersTo(
                sub.getInput(),
                [&](Node subBase) {
                  isValid = true;
                  // The result of a subaccess can refer to multiple storage
                  // locations corresponding to all the possible indices.
                  for (size_t index = 0; index < vecType.getNumElements();
                       ++index)
                    setValRefsTo(
                        res,
                        Node(subBase.getValue(),
                             subBase.getFieldID() + 1 +
                                 index *
                                     (vecType.getElementType().getMaxFieldID() +
                                      1)));
                  return success();
                },
                false);
            if (isValid)
              worklist.push_back(res);
          })
          .Case<InstanceOp>(
              [&](InstanceOp ins) { handleInstanceOp(ins, worklist); })
          .Case<MemOp>([&](MemOp mem) {
            if (!(mem.getReadLatency() == 0)) {
              return;
            }
            for (auto memPort : mem.getResults()) {
              if (!memPort.getType().isa<FIRRTLBaseType>())
                continue;
              memPorts.insert(memPort);
            }
          })
          .Default([&](auto) {});
    }
    dump();
  }

  void handleInstanceOp(InstanceOp ins, SmallVector<Value> &worklist) {
    FModuleOp refModule =
        dyn_cast<FModuleOp>(instanceGraph.getReferencedModule(ins));
    if (!refModule)
      return;
    for (auto port : ins.getResults()) {
      worklist.push_back(port);
      if (auto type = port.getType().dyn_cast<FIRRTLBaseType>())
        if (!type.isGround())
          setValRefsTo(port, Node(port, 0));
    }
  }

  void handlePorts(Node ref, SmallVector<Value> &children) {
    if (auto inst = dyn_cast<InstanceOp>(ref.getDefiningOp())) {
      auto res = ref.getValue().cast<OpResult>();
      auto portNum = res.getResultNumber();
      auto refMod =
          dyn_cast<FModuleOp>(instanceGraph.getReferencedModule(inst));
      if (!refMod)
        return;
      Node modArg(refMod.getArgument(portNum), ref.getFieldID());
      auto pathIter = portPaths.find(modArg);
      if (pathIter == portPaths.end())
        return;
      for (auto modOutPort : pathIter->second) {
        auto outPortNum =
            modOutPort.getValue().cast<BlockArgument>().getArgNumber();
        if (modOutPort.getFieldID() == 0) {
          children.push_back(inst.getResult(outPortNum));
          return;
        }
        Node instanceOutPort(inst.getResult(outPortNum),
                             modOutPort.getFieldID());
        for (auto r : fieldToVals[instanceOutPort])
          children.push_back(r);
      }
    } else if (auto mem = dyn_cast<MemOp>(ref.getDefiningOp())) {
      if (mem.getReadLatency() > 0)
        return;
      auto memPort = ref.getValue();
      auto type = memPort.getType().cast<BundleType>();
      auto enableFieldId = type.getFieldID((unsigned)ReadPortSubfield::en);
      auto dataFieldId = type.getFieldID((unsigned)ReadPortSubfield::data);
      auto addressFieldId = type.getFieldID((unsigned)ReadPortSubfield::addr);
      if (ref.getFieldID() == enableFieldId ||
          ref.getFieldID() == addressFieldId) {
        for (auto dataField : fieldToVals[Node(memPort, dataFieldId)])
          children.push_back(dataField);
      }
    }
  }

  void reportLoopFound(Value childVal, Node childNode,
                       DenseSet<Node> visitingSet,
                       SmallVector<Value> &visitingStack) {
    auto getName = [&](Node node) {
      if (isa<SubfieldOp, SubindexOp, SubaccessOp>(node.getDefiningOp())) {
        assert(!valRefersTo[node.getValue()].empty());
        return getFieldName(*valRefersTo[node.getValue()].begin()).first;
      }
      return getFieldName(node).first;
    };
    auto lastSignalName = getName(childNode);
    auto errorDiag = mlir::emitError(
        module.getLoc(),
        "detected combinational cycle in a FIRRTL module, sample path: ");
    if (!lastSignalName.empty())
      errorDiag << module.getName() << "." << lastSignalName << " <- ";
    while (visitingSet.contains(childNode)) {
      auto visitingVal = visitingStack.pop_back_val();
      auto signalName = getName(Node(visitingVal, 0));
      if (!signalName.empty())
        errorDiag << module.getName() << "." << signalName << " <- ";
      forallRefersTo(visitingVal, [&](Node visitingNode) {
        LLVM_DEBUG(llvm::dbgs()
                   << "\n Erasing visiting : " << getName(visitingNode));
        visitingSet.erase(visitingNode);
        return success();
      });
      // if (auto op = visitingVal.getDefiningOp())
      //   op->emitRemark(" is part of a combinational cycle");
    }
  }

  void handleConnects(Value dst, SmallVector<Node> &inputArgFields) {
    auto pathsToOutPort = [&](Node dstNode) {
      if (!dstNode.getValue().isa<BlockArgument>())
        return success();
      for (auto inArg : inputArgFields) {
        portPaths[inArg].insert(dstNode);
      }
      return success();
    };
    forallRefersTo(dst, pathsToOutPort);
  }

  void setValRefsTo(Value val, Node ref) {
    if (!val || !ref)
      return;
    fieldToVals[ref].insert(val);
    valRefersTo[val].insert(ref);
    if (auto arg = ref.getValue().dyn_cast<BlockArgument>())
      blockArgToFields[arg].insert(ref.getFieldID());
  }

  void forallRefersTo(Value val, std::function<LogicalResult(Node refNode)> f,
                      bool baseCase = true) {
    if (baseCase && f(Node(val, 0)).failed())
      return;
    auto refersToIter = valRefersTo.find(val);
    if (refersToIter != valRefersTo.end())
      for (auto ref : refersToIter->second)
        if (f(ref).failed())
          return;
  }

  void dump() {
    LLVM_DEBUG({
      for (auto valRef : valRefersTo) {
        llvm::dbgs() << "\n val :" << valRef.first;
        for (auto node : valRef.second)
          llvm::dbgs() << "\n Refers to :" << getFieldName(node).first;
      }
      for (auto dtv : fieldToVals) {
        llvm::dbgs() << "\n Field :" << getFieldName(dtv.first).first
                     << " ::" << dtv.first.getValue();
        for (auto val : dtv.second)
          llvm::dbgs() << "\n val :" << val;
      }
      for (auto p : portPaths) {
        llvm::dbgs() << "\n Output port : " << getFieldName(p.first).first
                     << " has comb path from :";
        for (auto src : p.second)
          llvm::dbgs() << "\n Input port : " << getFieldName(src).first;
      }
      for (auto b : blockArgToFields) {
        llvm::dbgs() << "\n Block argument:"
                     << getFieldName(Node(b.first, 0)).first;
        for (auto fi : b.second)
          llvm::dbgs() << "\n Block argument field :"
                       << getFieldName(Node(b.first, fi)).first;
      }
    });
  }

  FModuleOp &module;
  InstanceGraph &instanceGraph;
  // Map of a Value to all the FieldRefs that it refers to.
  DenseMap<Value, SetOfNodes> valRefersTo;

  DenseMap<Node, DenseSet<Value>> fieldToVals;
  // Comb paths that exist between module ports. This is maintained across
  // modules.
  DenseMap<Node, SetOfNodes> &portPaths;
  // Map of the fields of an aggregate BlockArgument/Ports that are used within
  // a module. This is maintained across modules.
  DenseMap<BlockArgument, std::set<unsigned>> &blockArgToFields;
};

/// This pass constructs a local graph for each module to detect combinational
/// cycles. To capture the cross-module combinational cycles, this pass inlines
/// the combinational paths between IOs of its subinstances into a subgraph and
/// encodes them in a `combPathsMap`.
class CheckCombLoopsPass : public CheckCombLoopsBase<CheckCombLoopsPass> {
public:
  void runOnOperation() override {
    auto &instanceGraph = getAnalysis<InstanceGraph>();
    DenseMap<Node, SetOfNodes> portPaths;
    DenseMap<BlockArgument, std::set<unsigned>> blockArgToFields;
    // Traverse modules in a post order to make sure the combinational paths
    // between IOs of a module have been detected and recorded in `portPaths`
    // before we handle its parent modules.
    for (auto *igNode : llvm::post_order<InstanceGraph *>(&instanceGraph)) {
      if (auto module = dyn_cast<FModuleOp>(*igNode->getModule())) {
        TarjansSCC rdf(module, instanceGraph, portPaths, blockArgToFields);
        if (rdf.processModule().failed()) {
          rdf.dump();
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
