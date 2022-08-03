//===- LowerXMR.cpp - FIRRTL Lower to XMR -----------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements FIRRTL XMR Lowering.
//
//===----------------------------------------------------------------------===//

#include "PassDetails.h"
#include "circt/Dialect/FIRRTL/FIRRTLInstanceGraph.h"
#include "circt/Dialect/FIRRTL/FIRRTLUtils.h"
#include "circt/Dialect/FIRRTL/Passes.h"
#include "mlir/IR/ImplicitLocOpBuilder.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/PostOrderIterator.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "firrtl-lower-xmr"

using namespace circt;
using namespace firrtl;
using hw::InnerRefAttr;

// The LowerXMRPass, replaces every RefResolveOp with a verbatim expr op,
// with the corresponding XMR. And also removes every Ref port from the modules
// and corresponding instances.
// This is a dataflow analysis over a very constrained RefType.
// Domain of the dataflow analysis is the set of all RefSendOps
// (and RefRecvOp). Essentially every RefType value must be mapped to one and
// only one RefSendOp.
// The analysis propagates the dataflow from every RefSendOp to the module
// port it is connected to. And subsequently to the modules that instantiate it
// through the instance port connections. The RefResolveOp is the final leaf
// into which the dataflow must flow into.
//
// Algorithm:
// Due to the downward only reference constraint on XMRs, the port order
//  traversal ensures that the RefSendOp (and RefRecvOp) will be encountered
// before any RefResolveOp.
// For every RefSendOp
//  0. RefSendOp has two ops, the result op and the reference port op.
//  1. create a new entry into the remoteRefToXMRsymVec.
//  2. Get the InnerRef to the result of RefSendOp and initialize the
//     remoteRefToXMRsymVec with it. (This is the remote op that the remote
//     xmr needs to refer to)
//  3. Get the module ref port for the RefSendOp and record it in the
//     refPortToXmrMap. This map tracks the dataflow from the original
//     RefSendOp to the corresponding ref ports.
// For every InstanceOp
//  1. For every RefType port of the InstanceOp, get the remote RefSendOp
//     that flows into the corresponding port of the Referenced module.
//     Because of the order of traversal and the constraints on the ref
//     ports, the Referenced module ref ports must already be resolved.
// 3. Update the remoteRefToXMRsymVec for the corresponding RefSendOp,
//    with an InnerRef to this InstanceOp. This denotes that the final
//    XMR must include this InstanceOp.
// 2. Forall users of the Instance ref port
//    1. If it is a connect op, it must be connected to a module ref port.
//       Record the corresponding remote RefSendOp to the module ref port.
//       Update the refPortToXmrMap.
//    2. If this is a RefResolveOp,
//       Replace the op result with a VerbatimExpr, representing the XMR.
//       The InnerRef sequence of symbols from remoteRefToXMRsymVec is
//       used to construct the symbol list for the verbatim.
//
class LowerXMRPass : public LowerXMRBase<LowerXMRPass> {

  /// This function forwards the dataflow information from the instance port to
  /// its users.
  void handleRefPort(Value instanceResult, Operation *remoteOp) {
    // Given the instance port `instanceResult` and the corresponding remote
    // RefSendOp `remoteOp` from which the instance port carries the dataflow
    // from, forward the dataflow infomration into the users of the instance
    // port. The instance ref port can either be connected to the module ref
    // port or to a RefResolveOp. Replace the RefResolveOp with the
    // corresponding verbatim Expr. Forward the data flow information regarding
    // the remote RefSendOp from the instance op to the module port to which its
    // connected.
    for (Operation *user : instanceResult.getUsers()) {
      if (auto connect = dyn_cast<FConnectLike>(user)) {
        Value driver;
        if (connect.getDest() == instanceResult)
          driver = connect.getSrc();
        else if (connect.getSrc() == instanceResult)
          driver = connect.getDest();
        // Downward reference constraint ensures that it must be a block
        // argument.
        auto blockArg = driver.cast<BlockArgument>();
        // Record the remote op from which the dataflow is established to this
        // module port.
        refPortToXmrMap[std::make_pair(
            instanceResult.getDefiningOp()->getParentOfType<FModuleOp>(),
            blockArg.getArgNumber())] = remoteOp;
      } else if (auto getOp = dyn_cast<RefResolveOp>(user)) {
        // The source of the dataflow for this RefResolveOp is established. So
        // replace the RefResolveOp with the coresponding VerbatimExpr to
        // generate the XMR.
        SmallVector<Attribute> xmrHierPath(
            remoteRefToXMRsymVec[remoteOp].rbegin(),
            remoteRefToXMRsymVec[remoteOp].rend());
        SmallString<128> xmrString;
        for (unsigned id = 0, e = xmrHierPath.size(); id < e; ++id) {
          ("{{" + Twine(id) + "}}").toVector(xmrString);
          if (id < e - 1)
            xmrString += '.';
        }
        ImplicitLocOpBuilder builder(getOp.getLoc(), getOp);
        auto xmrVerbatim = builder.create<VerbatimExprOp>(
            getOp.getType().cast<FIRRTLType>(), xmrString, ValueRange{},
            xmrHierPath);
        getOp.getResult().replaceAllUsesWith(xmrVerbatim);
        LLVM_DEBUG(llvm::dbgs() << "\n The get op:" << getOp
                                << "\n is connected to :" << remoteOp;
                   for (auto attr
                        : remoteRefToXMRsymVec[remoteOp]) llvm::dbgs()
                   << ":" << attr;);
      } else
        user->emitError("unhandled user of reference port");
      // This reference port will be removed, so remove all its users also.
      opsToRemove.push_back(user);
    }
  }

  bool handleRemoteReference(Operation *op, FModuleOp module) {
    Value xmrPort, xmrResult;
    if (auto xmrEnd = dyn_cast<RefSendOp>(op)) {
      xmrPort = xmrEnd.getRef();
      xmrResult = xmrEnd.getResult();
    } else // Handle RefRecvOp when its implemented.
      return false;
    LLVM_DEBUG(llvm::errs()
               << "\n xmr get/end drives :"
               << "\n for xmr get/end op :" << op << "\n xmr port" << xmrPort);
    // Get a reference to the actual signal to which the XMR will be generated.
    auto xmrDefOp = getInnerRefTo(xmrResult.getDefiningOp());
    remoteRefToXMRsymVec[op] = {xmrDefOp};

    // The downward reference constraint ensures that xmrPort must be a
    // BlockArg.
    auto blockArg = xmrPort.cast<BlockArgument>();
    // Record the remote reference op, that this module port refers to.
    refPortToXmrMap[std::make_pair(module, blockArg.getArgNumber())] = op;
    opsToRemove.push_back(op);
    return true;
  }

  bool handleInstanceOp(Operation *op, FModuleOp module) {
    auto inst = dyn_cast<InstanceOp>(op);
    if (!inst)
      return false;
    auto refMod = inst.getReferencedModule();
    for (size_t portNum = 0, e = cast<FModuleLike>(refMod).getNumPorts();
         portNum < e; ++portNum) {
      auto instanceResult = inst.getResult(portNum);
      if (!instanceResult.getType().isa<RefType>())
        continue;
      // Reference ports must be removed.
      refPortsToRemoveMap[inst].push_back(portNum);
      auto portModPair = std::make_pair(refMod, portNum);
      // Get the remote RefSendOp, that flows through the module ports.
      auto iter = refPortToXmrMap.find(portModPair);
      if (iter == refPortToXmrMap.end()) {
        inst.emitOpError("Ref port connection cannot be traced back to the "
                         "remote read/write op");
        return false;
      }
      auto remoteOp = iter->getSecond();
      // This instance op must participate in the XMR hierarchical path, so
      // record the innerRef to it.
      remoteRefToXMRsymVec[remoteOp].push_back(getInnerRefTo(inst));
      handleRefPort(instanceResult, remoteOp);
    }
    return true;
  }

  void runOnOperation() override {
    auto &instanceGraph = getAnalysis<InstanceGraph>();
    // Get the post order traversal of the modules in the InstanceGraph.
    SmallVector<FModuleOp, 0> modules(
        llvm::map_range(llvm::post_order(&instanceGraph), [](auto *node) {
          return cast<FModuleOp>(*node->getModule());
        }));
    // Traverse the modules in post order.
    for (auto module : modules) {
      LLVM_DEBUG(llvm::dbgs()
                 << "\n Traversing module:" << module.moduleNameAttr());
      for (Operation &op :
           llvm::make_early_inc_range(module.getBodyBlock()->getOperations()))
        // This only handles the RefSendOp (and RefRecvOp when its implemented)
        if (handleRemoteReference(&op, module))
          continue;
        else if (handleInstanceOp(&op, module))
          continue;

      for (size_t portNum = 0, e = module.getNumPorts(); portNum < e; ++portNum)
        if (module.getPortType(portNum).isa<RefType>()) {
          refPortsToRemoveMap[module].push_back(portNum);
        }
    }
    for (auto op : opsToRemove)
      op->erase();
    for (auto iter : refPortsToRemoveMap)
      if (auto mod = dyn_cast<FModuleOp>(iter.getFirst()))
        mod.erasePorts(iter.getSecond());
      else if (auto inst = dyn_cast<InstanceOp>(iter.getFirst())) {
        ImplicitLocOpBuilder b(inst.getLoc(), inst);
        inst.erasePorts(b, iter.getSecond());
        inst.erase();
      }
  }

  /// Get the cached namespace for a module.
  ModuleNamespace &getModuleNamespace(FModuleLike module) {
    auto it = moduleNamespaces.find(module);
    if (it != moduleNamespaces.end())
      return it->second;
    return moduleNamespaces.insert({module, ModuleNamespace(module)})
        .first->second;
  }

  InnerRefAttr getInnerRefTo(Operation *op) {
    return ::getInnerRefTo(op, "xmr_sym",
                           [&](FModuleOp mod) -> ModuleNamespace & {
                             return getModuleNamespace(mod);
                           });
  }
  /// Cached module namespaces.
  DenseMap<Operation *, ModuleNamespace> moduleNamespaces;

  /// Map of a ref port to the RefSend op. Each ref port can be satically
  /// resolved to a single remote send op, according to the constraints on the
  /// ref ports. The pair represents the module and the port number.
  DenseMap<std::pair<Operation *, size_t>, Operation *> refPortToXmrMap;
  /// Map of the remote RefSend op, to the sequence of the InnerRef symbols that
  /// represents the Final XMR. Each of the intermediary symbol is an instance
  /// op, and the final symbol represents the leaf operation corresponding to
  /// the XMR. The symbol sequence can represent a hierarchical path to the
  /// operation, for which the XMR needs to be constructed.
  DenseMap<Operation *, SmallVector<Attribute>> remoteRefToXMRsymVec;
  SmallVector<Operation *> opsToRemove;
  // Instance and module ref ports that needs to be removed.
  DenseMap<Operation *, SmallVector<unsigned>> refPortsToRemoveMap;
};

std::unique_ptr<mlir::Pass> circt::firrtl::createLowerXMRPass() {
  return std::make_unique<LowerXMRPass>();
}
