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

using SetOfNodes = DenseSet<FieldRef>;
using Node = FieldRef;

/// A value is in VisitingSet if its subtree is still being traversed. That is,
/// all its children have not yet been visited. If any Value is visited while
/// its still in the `VisitingSet`, that implies a back edge and a cycle.
struct VisitingSet {
private:
  /// The stack is maintained to keep track of the cycle, if one is found. This
  /// is required for an iterative DFS traversal, its implicitly recorded for a
  /// recursive version of this algorithm. Each entry in the stack is a list of
  /// aliasing Values, which were visited at the same time.
  SmallVector<SmallVector<Value, 2>> visitingStack;
  /// This map of the Visiting values, is for faster query, to check if a Value
  /// is in VisitingSet. It also records the corresponding index into the
  /// visitingStack, for faster pop until the Value.
  DenseMap<Value, unsigned> valToStackMap;

public:
  void appendEmpty() { visitingStack.push_back({}); }
  void appendToEnd(SmallVector<Value> &values) {
    auto stackSize = visitingStack.size() - 1;
    visitingStack.back().append(values.begin(), values.end());
    // Record the stack location where this Value is pushed.
    llvm::for_each(values, [&](Value v) { valToStackMap[v] = stackSize; });
  }
  bool contains(Value v) {
    return valToStackMap.find(v) != valToStackMap.end();
  }
  // Pop all the Values which were visited after v. Then invoke f (if present)
  // on a popped value for each index.
  void popUntillVal(Value v,
                    const llvm::function_ref<void(Value poppedVal)> f = {}) {
    auto valPos = valToStackMap[v];
    while (visitingStack.size() != valPos) {
      auto poppedVals = visitingStack.pop_back_val();
      Value poppedVal;
      llvm::for_each(poppedVals, [&](Value pv) {
        if (!poppedVal)
          poppedVal = pv;
        valToStackMap.erase(pv);
      });
      if (f && poppedVal)
        f(poppedVal);
    }
  }
};

class DiscoverLoops {

public:
  DiscoverLoops(FModuleOp module, InstanceGraph &instanceGraph,
                DenseMap<Node, SetOfNodes> &portPaths)
      : module(module), instanceGraph(instanceGraph), portPaths(portPaths) {}

  LogicalResult processModule() {
    LLVM_DEBUG(llvm::dbgs() << "\n processing module :" << module.getName());
    SmallVector<Value> worklist;
    // Traverse over ports and ops, to populate the worklist and get the
    // FieldRef corresponding to every Value. Also process the InstanceOps and
    // get the paths that exist between the ports of the referenced module.
    preprocess(worklist);

    llvm::DenseSet<Value> visited;
    VisitingSet visiting;

    // worklist is the list of roots, to begin the traversal from.
    for (auto root : worklist) {
      SmallVector<Value> dfsStack;
      dfsStack.push_back(root);
      SmallVector<Node> inputArgFields;
      LLVM_DEBUG(llvm::dbgs() << "\n Starting traversal from root :"
                              << getFieldName(Node(root, 0)).first);
      if (auto inArg = dyn_cast<BlockArgument>(root)) {
        if (module.getPortDirection(inArg.getArgNumber()) == Direction::In)
          // This is required, such that paths to output port can be discovered.
          // If there is an overlapping path from two input ports to an output
          // port, then the already visited nodes must be re-visited to discover
          // the comb paths to the output port.
          visited.clear();
      }
      while (!dfsStack.empty()) {
        auto dfsVal = dfsStack.back();
        if (!visiting.contains(dfsVal)) {
          unsigned dfsSize = dfsStack.size();

          LLVM_DEBUG(llvm::dbgs() << "\n Stack pop :"
                                  << getFieldName(Node(dfsVal, 0)).first << ","
                                  << dfsVal;);

          // Visiting set will contain all the values which alias with the
          // dfsVal, this is required to detect back edges to aliasing Values.
          // That is fieldRefs that can refer to the same memory location.
          visiting.appendEmpty();
          // Record all the children of `dfsVal`.
          SmallVector<Value, 8> children;
          // If this is an input port field, then record it. This is used to
          // discover paths from input to output ports. Only the last input port
          // that is visited on the DFS traversal is recorded.
          SmallVector<Node, 2> inputArgFieldsTemp;
          // All the Values that refer to the same FieldRef are added to the
          // aliasingValues.
          SmallVector<Value> aliasingValues = {dfsVal};
          // If `dfsVal` is a subfield, then get all the FieldRefs that it
          // refers to and then get all the values that alias with it.
          forallRefersTo(dfsVal, [&](Node ref) {
            // If this subfield refers to instance/mem results(input port), then
            // add the output port FieldRefs that exist in the referenced module
            // comb paths to the children.
            handlePorts(ref, children);
            // Get all the values that refer to this FieldRef, and add them to
            // the aliasing values.
            auto ftovIter = fieldToVals.find(ref);
            if (ftovIter != fieldToVals.end())
              for (auto alias : ftovIter->second)
                aliasingValues.push_back(alias);
            if (auto arg = dyn_cast<BlockArgument>(ref.getValue()))
              if (module.getPortDirection(arg.getArgNumber()) == Direction::In)
                inputArgFieldsTemp.push_back(ref);

            return success();
          });
          if (!inputArgFieldsTemp.empty())
            inputArgFields = std::move(inputArgFieldsTemp);
          auto type = dfsVal.getType().dyn_cast<FIRRTLBaseType>();

          aliasingValues.erase(
              std::unique(aliasingValues.begin(), aliasingValues.end()),
              aliasingValues.end());
          visiting.appendToEnd(aliasingValues);
          visited.insert(aliasingValues.begin(), aliasingValues.end());
          // Add the Value to `children`, to which a path exists from `dfsVal`.
          for (auto dfsFromVal : aliasingValues) {

            for (auto &use : dfsFromVal.getUses()) {
              Operation *owner = use.getOwner();
              Value childVal;
              if (owner->getNumResults() == 1 && !type.isa<ClockType>())
                childVal = owner->getResult(0);
              else if (auto connect = dyn_cast<FConnectLike>(owner))
                if (use.getOperandNumber() == 1) {
                  auto dst = connect.getDest();
                  if (handleConnects(dst, inputArgFields).succeeded())
                    childVal = dst;
                }
              if (childVal && childVal.getType().isa<FIRRTLBaseType>())
                children.push_back(childVal);
            }
          }
          for (auto childVal : children) {
            // This childVal can be ignored, if
            // It is a Register or a subfield of a register.
            if (!visited.contains(childVal))
              dfsStack.push_back(childVal);
            // If the childVal is a sub, then check if it aliases with any of
            // the predecessors (the visiting set).
            if (visiting.contains(childVal)) {
              // Comb Cycle Detected !!
              reportLoopFound(childVal, visiting);
              return failure();
            }
          }
          // child nodes added, continue the DFS
          if (dfsSize != dfsStack.size())
            continue;
        }
        // Node is an SCC root node, pop the visiting stack to remove the nodes
        // that are no longer active predecessors, that is their sub-tree is
        // already explored.
        // All the Values reachable from `dfsVal` have been explored, remove
        // it and its children from the visiting stack.
        visiting.popUntillVal(dfsVal);

        auto popped = dfsStack.pop_back_val();
        LLVM_DEBUG(llvm::dbgs()
                   << "\n dfs popped :" << getFieldName(Node(popped, 0)).first);
        dump();
      }
    }
    LLVM_DEBUG({
      for (auto pp : portPaths) {
        llvm::dbgs() << "\n Path from :" << getFieldName(pp.first).first;
        for (auto outport : pp.second)
          llvm::dbgs() << " output :" << getFieldName(outport).first;
      }
    });

    return success();
  }

  // Preprocess the module ops to get the
  // 1. FieldRef corresponding to any Value,
  // 2. roots for DFS traversal,
  void preprocess(SmallVector<Value> &worklist) {
    // All the input ports are added to the worklist.
    size_t ops = 0;
    for (BlockArgument arg : module.getArguments()) {
      ops++;
      auto argType = arg.getType();
      if (argType.isa<RefType>())
        continue;
      if (module.getPortDirection(arg.getArgNumber()) == Direction::In)
        worklist.push_back(arg);
      if (!argType.cast<FIRRTLBaseType>().isGround())
        setValRefsTo(arg, Node(arg, 0));
    }
    DenseSet<Value> memPorts;

    for (auto &op : module.getOps()) {
      ops++;
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
            SmallVector<Node, 4> fields;
            forallRefersTo(
                sub.getInput(),
                [&](Node subBase) {
                  isValid = true;
                  fields.push_back(Node(subBase.getValue(),
                                        subBase.getFieldID() + fieldIndex));
                  return success();
                },
                false);
            if (isValid) {
              for (auto f : fields)
                setValRefsTo(res, f);
            }
          })
          .Case<SubindexOp>([&](SubindexOp sub) {
            auto res = sub.getResult();
            bool isValid = false;
            auto index = sub.getIndex();
            SmallVector<Node, 4> fields;
            forallRefersTo(
                sub.getInput(),
                [&](Node subBase) {
                  isValid = true;
                  fields.push_back(Node(subBase.getValue(),
                                        subBase.getFieldID() + index + 1));
                  return success();
                },
                false);
            if (isValid) {
              for (auto f : fields)
                setValRefsTo(res, f);
            }
          })
          .Case<SubaccessOp>([&](SubaccessOp sub) {
            auto vecType = sub.getInput().getType().cast<FVectorType>();
            auto res = sub.getResult();
            bool isValid = false;
            SmallVector<Node, 4> fields;
            forallRefersTo(
                sub.getInput(),
                [&](Node subBase) {
                  isValid = true;
                  // The result of a subaccess can refer to multiple storage
                  // locations corresponding to all the possible indices.
                  for (size_t index = 0; index < vecType.getNumElements();
                       ++index)
                    fields.push_back(Node(
                        subBase.getValue(),
                        subBase.getFieldID() + 1 +
                            index * (vecType.getElementType().getMaxFieldID() +
                                     1)));
                  return success();
                },
                false);
            if (isValid) {
              for (auto f : fields)
                setValRefsTo(res, f);
            }
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
  }

  void handleInstanceOp(InstanceOp ins, SmallVector<Value> &worklist) {
    for (auto port : ins.getResults()) {
      if (auto type = port.getType().dyn_cast<FIRRTLBaseType>()) {
        worklist.push_back(port);
        if (!type.isGround())
          setValRefsTo(port, Node(port, 0));
      }
    }
  }

  void handlePorts(Node ref, SmallVectorImpl<Value> &children) {
    if (auto inst = dyn_cast_or_null<InstanceOp>(ref.getDefiningOp())) {
      auto res = ref.getValue().cast<OpResult>();
      auto portNum = res.getResultNumber();
      auto refMod =
          dyn_cast_or_null<FModuleOp>(*instanceGraph.getReferencedModule(inst));
      if (!refMod)
        return;
      Node modArg(refMod.getArgument(portNum), ref.getFieldID());
      auto pathIter = portPaths.find(modArg);
      if (pathIter == portPaths.end())
        return;
      for (auto modOutPort : pathIter->second) {
        auto outPortNum =
            cast<BlockArgument>(modOutPort.getValue()).getArgNumber();
        if (modOutPort.getFieldID() == 0) {
          children.push_back(inst.getResult(outPortNum));
          return;
        }
        Node instanceOutPort(inst.getResult(outPortNum),
                             modOutPort.getFieldID());
        llvm::append_range(children, fieldToVals[instanceOutPort]);
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

  void reportLoopFound(Value childVal, VisitingSet visiting) {
    auto getName = [&](Node node) {
      if (isa<SubfieldOp, SubindexOp, SubaccessOp>(node.getDefiningOp())) {
        assert(!valRefersTo[node.getValue()].empty());
        return getFieldName(*valRefersTo[node.getValue()].begin()).first;
      }
      return getFieldName(node).first;
    };
    Node childNode(childVal, 0);
    auto lastSignalName = getName(childNode);
    auto errorDiag = mlir::emitError(
        module.getLoc(),
        "detected combinational cycle in a FIRRTL module, sample path: ");
    if (!lastSignalName.empty())
      errorDiag << module.getName() << "." << lastSignalName << " <- ";
    visiting.popUntillVal(childVal, [&](Value visitingVal) {
      auto signalName = getName(Node(visitingVal, 0));
      if (!signalName.empty())
        errorDiag << module.getName() << "." << signalName << " <- ";
    });
    if (!lastSignalName.empty())
      errorDiag << module.getName() << "." << lastSignalName;
  }

  LogicalResult handleConnects(Value dst, SmallVector<Node> &inputArgFields) {

    bool onlyFieldZero = true;
    auto pathsToOutPort = [&](Node dstNode) {
      if (dstNode.getFieldID() != 0)
        onlyFieldZero = false;
      if (!dstNode.getValue().isa<BlockArgument>()) {
        return failure();
      }
      onlyFieldZero = false;
      for (auto inArg : inputArgFields) {
        portPaths[inArg].insert(dstNode);
      }
      return success();
    };
    forallRefersTo(dst, pathsToOutPort);

    if (onlyFieldZero) {
      if (isa<RegOp, RegResetOp, SubfieldOp, SubaccessOp, SubindexOp>(
              dst.getDefiningOp()))
        return failure();
    }
    return success();
  }

  void setValRefsTo(Value val, Node ref) {
    if (!val || !ref)
      return;
    fieldToVals[ref].insert(val);
    valRefersTo[val].insert(ref);
  }

  void forallRefersTo(Value val,
                      const llvm::function_ref<LogicalResult(Node &refNode)> f,
                      bool baseCase = true) {
    auto refersToIter = valRefersTo.find(val);
    if (refersToIter != valRefersTo.end()) {
      for (auto ref : refersToIter->second)
        if (f(ref).failed())
          return;
    } else if (baseCase) {
      Node base(val, 0);
      if (f(base).failed())
        return;
    }
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
    });
  }

  FModuleOp module;
  InstanceGraph &instanceGraph;
  /// Map of a Value to all the FieldRefs that it refers to.
  DenseMap<Value, SetOfNodes> valRefersTo;

  DenseMap<Node, DenseSet<Value>> fieldToVals;
  /// Comb paths that exist between module ports. This is maintained across
  /// modules.
  DenseMap<Node, SetOfNodes> &portPaths;
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
    // Traverse modules in a post order to make sure the combinational paths
    // between IOs of a module have been detected and recorded in `portPaths`
    // before we handle its parent modules.
    for (auto *igNode : llvm::post_order<InstanceGraph *>(&instanceGraph)) {
      if (auto module = dyn_cast<FModuleOp>(*igNode->getModule())) {
        DiscoverLoops rdf(module, instanceGraph, portPaths);
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
