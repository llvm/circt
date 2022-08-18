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

// The LowerXMRPass will replace every RefResolveOp with an XMR encoded within a
// verbatim expr op. This also removes every RefType port from the modules and
// corresponding instances. This is a dataflow analysis over a very constrained
// RefType. Domain of the dataflow analysis is the set of all RefSendOps. It
// computes an interprocedural reaching definitions (of RefSendOp) analysis.
// Essentially every RefType value must be mapped to one and only one RefSendOp.
// The analysis propagates the dataflow from every RefSendOp to every value of
// RefType across modules. The RefResolveOp is the final leaf into which the
// dataflow must reach.
//
// Due to the downward only reference constraint on XMRs, the post order
// traversal ensures that the RefSendOp will be encountered before any
// RefResolveOp. But since there can be multiple readers, multiple
// RefResolveOps can be reachable from a single RefSendOp. To support multiply
// instantiated modules and multiple readers, it is essential to track the path
// to the RefSendOp, other than just the RefSendOp. For example, if there exists
// a wire `xmr_wire` in module `L2`, the algorithm needs to support generating
// Top.L1.L2.xmr_wire and Top.L2.xmr_wire and Top.L3.L2.xmr_wire for different
// instance paths that exist in the circuit.
// Algorithm:
//
// For every RefSendOp
//  1. The BaseType input is the value to which the final XMR should refer to.
//  2. Set the `reachingRefSendAt` for the result RefType to an InnerRef to this
//  op. This map tracks the dataflow path from the original RefSendOp to the
//  corresponding ref values.
// For every InstanceOp
//  1. For every RefType port of the InstanceOp, get the remote RefSendOp that
//  flows into the corresponding port of the Referenced module. Because of the
//  order of traversal and the constraints on the ref ports, the Referenced
//  module ref ports must already be resolved.
//  2. Update the `reachingRefSendAt` for the corresponding RefSendOp, to append
//  an InnerRef to this InstanceOp. This denotes that the final XMR must include
//  this InstanceOp.
// For every ConnectLike op
//  1. Copy the dataflow of the src to the dest.
// For every RefResolveOp,
//  1. Replace the op result with a VerbatimExpr, representing the XMR. The
//  InnerRef sequence of symbols from `reachingRefSendAt` is used to construct
//  the symbol list for the verbatim.
//
class LowerXMRPass : public LowerXMRBase<LowerXMRPass> {

  void runOnOperation() override {
    auto &instanceGraph = getAnalysis<InstanceGraph>();
    // The dataflow function, that propagates the reachable RefSendOp across
    // RefType Ops.
    auto transferFunc = [&](Operation &op) -> LogicalResult {
      return TypeSwitch<Operation *, LogicalResult>(&op)
          .Case<RefSendOp>([&](auto send) {
            // Get a reference to the actual signal to which the XMR will be
            // generated.
            auto xmrDefOp = getInnerRefTo(send.getBase());
            // Record the remote reference op, that this ref value refers to.
            reachingRefSendAt[send.getResult()] =
                ArrayAttr::get(send.getContext(), {xmrDefOp});
            opsToRemove.push_back(send);
            return success();
          })
          .Case<InstanceOp>([&](auto inst) { return handleInstanceOp(inst); })
          .Case<FConnectLike>([&](FConnectLike connect) {
            // Ignore BaseType.
            if (!connect.getSrc().getType().isa<RefType>())
              return success();
            // Get the dataflow value into the src.
            if (auto remoteOpPath = getRemoteRefSend(connect.getSrc()))
              reachingRefSendAt[connect.getDest()] = remoteOpPath;
            else
              return failure();
            opsToRemove.push_back(connect);
            return success();
          })
          .Case<RefResolveOp>(
              [&](auto resolve) { return handleRefResolve(resolve); })
          .Default([&](auto) { return success(); });
    };

    DenseSet<Operation *> visitedModules;
    // Traverse the modules in post order.
    for (auto node : llvm::post_order(&instanceGraph)) {
      auto module = dyn_cast<FModuleOp>(*node->getModule());
      if (!module)
        continue;
      if (!visitedModules.insert(module).second)
        continue;

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

  ArrayAttr getRemoteRefSend(Value val) {
    auto iter = reachingRefSendAt.find(val);
    if (iter != reachingRefSendAt.end())
      return iter->getSecond();
    // The referenced module must have already been analyzed, error out if the
    // dataflow at the child module is not resolved.
    if (BlockArgument arg = val.dyn_cast<BlockArgument>())
      arg.getOwner()->getParentOp()->emitError(
          "reference dataflow cannot be traced back to the remote read op for "
          "module port '")
          << dyn_cast<FModuleOp>(arg.getOwner()->getParentOp())
                 .getPortName(arg.getArgNumber())
          << "'";
    else
      val.getDefiningOp()->emitOpError(
          "reference dataflow cannot be traced back to the remote read op");
    return {};
  }

  // Replace the RefResolveOp with verbatim op representing the XMR.
  LogicalResult handleRefResolve(RefResolveOp resolve) {
    opsToRemove.push_back(resolve);
    auto remoteOpPath = getRemoteRefSend(resolve.getRef());
    if (!remoteOpPath)
      return failure();
    auto xmrSize = remoteOpPath.size();
    SmallVector<Attribute> xmrHierPath(xmrSize);
    SmallString<128> xmrString;
    for (auto instanceRef : llvm::enumerate(remoteOpPath)) {
      xmrHierPath[xmrSize - instanceRef.index() - 1] = instanceRef.value();
      ("{{" + Twine(instanceRef.index()) + "}}").toVector(xmrString);
      if (instanceRef.index() < xmrSize - 1)
        xmrString += '.';
    }

    // The source of the dataflow for this RefResolveOp is established. So
    // replace the RefResolveOp with the coresponding VerbatimExpr to
    // generate the XMR.
    ImplicitLocOpBuilder builder(resolve.getLoc(), resolve);
    auto xmrVerbatim =
        builder.create<VerbatimExprOp>(resolve.getType().cast<FIRRTLType>(),
                                       xmrString, ValueRange{}, xmrHierPath);
    resolve.getResult().replaceAllUsesWith(xmrVerbatim);
    return success();
  }

  // Propagate the reachable RefSendOp across modules.
  LogicalResult handleInstanceOp(InstanceOp inst) {
    auto refMod = dyn_cast<FModuleOp>(inst.getReferencedModule());
    for (size_t portNum = 0, e = inst.getNumResults(); portNum < e; ++portNum) {
      auto instanceResult = inst.getResult(portNum);
      if (!instanceResult.getType().isa<RefType>())
        continue;
      if (!refMod)
        return inst.emitOpError("cannot lower ext modules with RefType ports");
      // Reference ports must be removed.
      refPortsToRemoveMap[inst].push_back(portNum);
      // Drop dead instance ports.
      if (instanceResult.use_empty())
        continue;
      auto refModuleArg = refMod.getArgument(portNum);
      // Get the remote RefSendOp, that flows through the module ports.
      auto remoteOpPath = getRemoteRefSend(refModuleArg);
      if (!remoteOpPath)
        return failure();

      auto pathToRefSend = remoteOpPath.getValue().vec();
      pathToRefSend.push_back(getInnerRefTo(inst));
      reachingRefSendAt[instanceResult] =
          ArrayAttr::get(inst.getContext(), pathToRefSend);
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

  InnerRefAttr getInnerRefTo(Value val) {
    if (auto arg = val.dyn_cast<BlockArgument>())
      return ::getInnerRefTo(
          cast<FModuleLike>(arg.getParentBlock()->getParentOp()),
          arg.getArgNumber(), "xmr_sym",
          [&](FModuleLike mod) -> ModuleNamespace & {
            return getModuleNamespace(mod);
          });
    else
      return getInnerRefTo(val.getDefiningOp());
  }

  InnerRefAttr getInnerRefTo(Operation *op) {
    return ::getInnerRefTo(op, "xmr_sym",
                           [&](FModuleOp mod) -> ModuleNamespace & {
                             return getModuleNamespace(mod);
                           });
  }

  /// Cached module namespaces.
  DenseMap<Operation *, ModuleNamespace> moduleNamespaces;

  /// Map of a reference value to the path to the RefSend op. The path to
  /// RefSend is represented as an ArrayAttr, which is an array of the InnerRef
  /// to InstanceOps. The path is required since there can be multiple paths to
  /// the RefSend and we need to identify a unique path. Each ref value can be
  /// satically resolved to a single remote send op, according to the
  /// constraints on the RefType. The ArrayAttr ensures that only unique copies
  /// of the path exist.
  DenseMap<Value, ArrayAttr> reachingRefSendAt;

  /// RefResolve, RefSend, and Connects involving them that will be removed.
  SmallVector<Operation *> opsToRemove;

  // Instance and module ref ports that needs to be removed.
  DenseMap<Operation *, SmallVector<unsigned>> refPortsToRemoveMap;
};

std::unique_ptr<mlir::Pass> circt::firrtl::createLowerXMRPass() {
  return std::make_unique<LowerXMRPass>();
}
