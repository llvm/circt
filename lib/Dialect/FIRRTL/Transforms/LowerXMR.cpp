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
// with the corresponding XMR. This also removes every RefType port from
// the modules and corresponding instances.
// This is a dataflow analysis over a very constrained RefType.
// Domain of the dataflow analysis is the set of all RefSendOps.
// Essentially every RefType value must be mapped to one and
// only one RefSendOp.
// The analysis propagates the dataflow from every RefSendOp to every value of
// RefType across modules.
// The RefResolveOp is the final leaf into which the dataflow must flow into.
//
// Algorithm:
// Due to the downward only reference constraint on XMRs, the post order
//  traversal ensures that the RefSendOp will be encountered before any
//  RefResolveOp.
// For every RefSendOp
//  0. The BaseType input is the value to which the final XMR should refer to.
//  1. create a new entry into the `remoteRefToXMRsymVec`.
//  2. Get the InnerRef to the BaseType input of RefSendOp and initialize the
//     remoteRefToXMRsymVec with it. (This is the remote op that the
//     xmr needs to refer to)
//  3. Set the `refSendAt` for the result RefType to this op. This map tracks
//  the dataflow from the original
//     RefSendOp to the corresponding ref ports.
// For every InstanceOp
//  1. For every RefType port of the InstanceOp, get the remote RefSendOp
//     that flows into the corresponding port of the Referenced module.
//     Because of the order of traversal and the constraints on the ref
//     ports, the Referenced module ref ports must already be resolved.
//  2. Update the `remoteRefToXMRsymVec` for the corresponding RefSendOp,
//    with an InnerRef to this InstanceOp. This denotes that the final
//    XMR must include this InstanceOp.
// For every ConnectLike op
//  1. Copy the dataflow of the src to the dest.
// For every RefResolveOp,
//  1. Replace the op result with a VerbatimExpr, representing the XMR.
//       The InnerRef sequence of symbols from remoteRefToXMRsymVec is
//       used to construct the symbol list for the verbatim.
//
class LowerXMRPass : public LowerXMRBase<LowerXMRPass> {

  void runOnOperation() override {
    auto &instanceGraph = getAnalysis<InstanceGraph>();
    // Get the post order traversal of the modules in the InstanceGraph.
    SmallVector<FModuleOp> modules(
        llvm::map_range(llvm::post_order(&instanceGraph), [](auto *node) {
          return cast<FModuleOp>(*node->getModule());
        }));
    // The dataflow function, that propagates the reachable RefSendOp across
    // RefType Ops.
    auto transferFunc = [&](Operation &op) -> LogicalResult {
      return TypeSwitch<Operation *, LogicalResult>(&op)
          .Case<RefSendOp>([&](auto send) {
            // Get a reference to the actual signal to which the XMR will be
            // generated.
            auto xmrDefOp = getInnerRefTo(send.getBase().getDefiningOp());
            remoteRefToXMRsymVec[send] = {xmrDefOp};
            // Record the remote reference op, that this ref value refers to.
            refSendAt[send.getResult()] = send;
            opsToRemove.push_back(send);
            return success();
          })
          .Case<InstanceOp>([&](auto inst) { return handleInstanceOp(inst); })
          .Case<FConnectLike>([&](FConnectLike connect) {
            // Ignore BaseType.
            if (!connect.getSrc().getType().isa<RefType>())
              return success();
            // Get the dataflow value into the src.
            if (auto remoteOp = getRemoteRefSend(connect.getSrc(), connect))
              refSendAt[connect.getDest()] = remoteOp;
            else
              return failure();
            opsToRemove.push_back(connect);
            return success();
          })
          .Case<RefResolveOp>(
              [&](auto resolve) { return handleRefResolve(resolve); })
          .Default([&](auto) { return success(); });
    };
    // Traverse the modules in post order.
    for (auto module : modules) {
      LLVM_DEBUG(llvm::dbgs()
                 << "\n Traversing module:" << module.moduleNameAttr());

      for (Operation &op : module.getBodyBlock()->getOperations())
        if (transferFunc(op).failed())
          return signalPassFailure();

      // Record all the RefType ports to be removed later.
      for (size_t portNum = 0, e = module.getNumPorts(); portNum < e; ++portNum)
        if (module.getPortType(portNum).isa<RefType>())
          refPortsToRemoveMap[module].push_back(portNum);
    }
    // Now erase all the Ops and ports of RefType.
    // This needs to be done as the last step to ensure uses are erased before
    // the def is erased.
    for (auto op : llvm::reverse(opsToRemove))
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

  Operation *getRemoteRefSend(Value val, Operation *op) {
    auto iter = refSendAt.find(val);
    if (iter != refSendAt.end())
      return iter->getSecond();
    // The referenced module must have already been analyzed, error out if the
    // dataflow at the child module is not resolved.
    op->emitOpError(
        "reference dataflow cannot be traced back to the remote read op");
    return nullptr;
  }
  //
  // Replace the RefResolveOp with verbatim op representing the XMR.
  LogicalResult handleRefResolve(RefResolveOp resolve) {
    opsToRemove.push_back(resolve);
    auto remoteOp = getRemoteRefSend(resolve.getRef(), resolve);
    if (!remoteOp)
      return failure();

    // The source of the dataflow for this RefResolveOp is established. So
    // replace the RefResolveOp with the coresponding VerbatimExpr to
    // generate the XMR.
    SmallVector<Attribute> xmrHierPath(remoteRefToXMRsymVec[remoteOp].rbegin(),
                                       remoteRefToXMRsymVec[remoteOp].rend());
    SmallString<128> xmrString;
    for (unsigned id = 0, e = xmrHierPath.size(); id < e; ++id) {
      ("{{" + Twine(id) + "}}").toVector(xmrString);
      if (id < e - 1)
        xmrString += '.';
    }
    ImplicitLocOpBuilder builder(resolve.getLoc(), resolve);
    auto xmrVerbatim =
        builder.create<VerbatimExprOp>(resolve.getType().cast<FIRRTLType>(),
                                       xmrString, ValueRange{}, xmrHierPath);
    resolve.getResult().replaceAllUsesWith(xmrVerbatim);
    LLVM_DEBUG({
      llvm::dbgs() << "\n The get op:" << resolve
                   << "\n is connected to :" << remoteOp;
      for (auto attr : remoteRefToXMRsymVec[remoteOp])
        llvm::dbgs() << ":" << attr;
    });
    return success();
  }

  // Propagate the reachable RefSendOp across modules.
  LogicalResult handleInstanceOp(InstanceOp inst) {
    auto refMod = dyn_cast<FModuleOp>(inst.getReferencedModule());
    for (size_t portNum = 0, e = inst.getNumResults(); portNum < e; ++portNum) {
      auto instanceResult = inst.getResult(portNum);
      if (!instanceResult.getType().isa<RefType>())
        continue;
      if (!refMod) {
        return inst.emitOpError("cannot lower ext modules with RefType ports");
      }
      // Reference ports must be removed.
      refPortsToRemoveMap[inst].push_back(portNum);
      // Drop dead instance ports.
      if (instanceResult.use_empty())
        continue;
      auto refModuleArg = refMod.getArgument(portNum);
      // Get the remote RefSendOp, that flows through the module ports.
      if (auto remoteOp = getRemoteRefSend(refModuleArg, inst)) {
        // This instance op must participate in the XMR hierarchical path, so
        // record the innerRef to it.
        remoteRefToXMRsymVec[remoteOp].push_back(getInnerRefTo(inst));
        refSendAt[instanceResult] = remoteOp;
      } else
        return failure();
    }
    return success();
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

  /// Map of a reference value to the RefSend op. Each ref value can be
  /// satically resolved to a single remote send op, according to the
  /// constraints on the RefType.
  DenseMap<Value, Operation *> refSendAt;
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
