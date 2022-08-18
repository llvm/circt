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

/// The LowerXMRPass will replace every RefResolveOp with an XMR encoded within
/// a verbatim expr op. This also removes every RefType port from the modules
/// and corresponding instances. This is a dataflow analysis over a very
/// constrained RefType. Domain of the dataflow analysis is the set of all
/// RefSendOps. It computes an interprocedural reaching definitions (of
/// RefSendOp) analysis. Essentially every RefType value must be mapped to one
/// and only one RefSendOp. The analysis propagates the dataflow from every
/// RefSendOp to every value of RefType across modules. The RefResolveOp is the
/// final leaf into which the dataflow must reach.
///
/// Since there can be multiple readers, multiple RefResolveOps can be reachable
/// from a single RefSendOp. To support multiply instantiated modules and
/// multiple readers, it is essential to track the path to the RefSendOp, other
/// than just the RefSendOp. For example, if there exists a wire `xmr_wire` in
/// module `Foo`, the algorithm needs to support generating Top.Bar.Foo.xmr_wire
/// and Top.Foo.xmr_wire and Top.Zoo.Foo.xmr_wire for different instance paths
/// that exist in the circuit.

class LowerXMRPass : public LowerXMRBase<LowerXMRPass> {

  void runOnOperation() override {
    auto &instanceGraph = getAnalysis<InstanceGraph>();
    SmallVector<RefResolveOp> resolveOps;
    // The dataflow function, that propagates the reachable RefSendOp across
    // RefType Ops.
    auto transferFunc = [&](Operation &op) -> LogicalResult {
      return TypeSwitch<Operation *, LogicalResult>(&op)
          .Case<RefSendOp>([&](RefSendOp send) {
            // Get a reference to the actual signal to which the XMR will be
            // generated.
            Value xmrDef = send.getBase();
            if (!xmrDef.isa<BlockArgument>()) {
              Operation *xmrDefOp = xmrDef.getDefiningOp();
              if (!isa<InnerSymbolOpInterface>(xmrDefOp)) {
                // Add a node, for non-innerSym ops. Otherwise the sym will be
                // dropped after LowerToHW.
                ImplicitLocOpBuilder b(xmrDefOp->getLoc(), xmrDefOp);
                b.setInsertionPointAfter(xmrDefOp);
                StringRef opName;
                auto nameKind = NameKindEnum::DroppableName;
                if (auto name = xmrDefOp->getAttrOfType<StringAttr>("name")) {
                  opName = name.getValue();
                  nameKind = NameKindEnum::InterestingName;
                }
                xmrDef = b.create<NodeOp>(xmrDef.getType(), xmrDef, opName,
                                          nameKind);
              }
            }
            // Create a new entry for this RefSendOp. The path is currently
            // local.
            addReachingSendsEntry(send.getResult(), getInnerRefTo(xmrDef));
            markForRemoval(send);
            return success();
          })
          .Case<InstanceOp>([&](auto inst) { return handleInstanceOp(inst); })
          .Case<FConnectLike>([&](FConnectLike connect) {
            // Ignore BaseType.
            if (!connect.getSrc().getType().isa<RefType>())
              return success();
            markForRemoval(connect);
            mergeDataflow(connect.getSrc(), connect.getDest());
            return success();
          })
          .Case<RefResolveOp>([&](RefResolveOp resolve) {
            dataflowAt[resolve.getResult()] =
                getRemoteRefSend(resolve.getRef()).value();
            resolveOps.push_back(resolve);
            markForRemoval(resolve);
            return success();
          })
          .Default([&](auto) { return success(); });
    };

    // Traverse the modules in post order.
    for (auto node : llvm::post_order(&instanceGraph)) {
      auto module = dyn_cast<FModuleOp>(*node->getModule());
      if (!module)
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

    for (auto refResolve : resolveOps)
      if (handleRefResolve(refResolve).failed())
        return signalPassFailure();
    garbageCollect();
  }

  // Replace the RefResolveOp with verbatim op representing the XMR.
  LogicalResult handleRefResolve(RefResolveOp resolve) {
    auto remoteOpPath = getRemoteRefSend(resolve.getRef(), false);
    if (!remoteOpPath)
      return failure();
    auto xmrSize = refSendList[remoteOpPath.value()].size();
    SmallVector<Attribute> xmrHierPath(xmrSize);
    SmallString<128> xmrString;
    for (const auto &instanceRef :
         llvm::enumerate(refSendList[remoteOpPath.value()])) {
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
    bool multiplyInstantiated = !visitedModules.insert(refMod).second;
    for (size_t portNum = 0, e = inst.getNumResults(); portNum < e; ++portNum) {
      auto instanceResult = inst.getResult(portNum);
      if (!instanceResult.getType().isa<RefType>())
        continue;
      if (!refMod)
        return inst.emitOpError("cannot lower ext modules with RefType ports");
      // Reference ports must be removed.
      refPortsToRemoveMap[inst].push_back(portNum);
      // Drop the dead-instance-ports.
      if (instanceResult.use_empty())
        continue;
      auto refModuleArg = refMod.getArgument(portNum);
      if (inst.getPortDirection(portNum) == Direction::Out) {
        // For output instance ports, the dataflow is into this module.
        // Get the remote RefSendOp, that flows through the module ports.
        // If dataflow at remote module argument does not exist, error out.
        auto remoteOpPath = getRemoteRefSend(refModuleArg, false);
        if (!remoteOpPath)
          return failure();
        // Get the path to reaching refSend at the referenced module argument.
        // Now append this instance to the path to the reaching refSend.
        addReachingSendsEntry(instanceResult, getInnerRefTo(inst),
                              remoteOpPath.value());
      } else {
        // For input instance ports, the dataflow is into the referenced module.
        // Input RefType port implies, generating an upward scoped XMR.
        // No need to add the instance context, since downward reference must be
        // through single instantiated modules.
        if (multiplyInstantiated)
          return refMod.emitOpError(
                     "multiply instantiated module with input RefType port '")
                 << portNum << "'";

        auto dataFlowAtRefArg = getRemoteRefSend(refModuleArg).value();

        dataflowAt[instanceResult] = dataFlowAtRefArg;
      }
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

  void markForRemoval(Operation *op) { opsToRemove.push_back(op); }

  Optional<size_t> getRemoteRefSend(Value val, bool mayNotExist = true) {
    auto iter = dataflowAt.find(val);
    if (iter != dataflowAt.end())
      if (mayNotExist || !refSendList[iter->getSecond()].empty())
        return iter->getSecond();
    if (!mayNotExist) {
      // The referenced module must have already been analyzed, error out if the
      // dataflow at the child module is not resolved.
      if (BlockArgument arg = val.dyn_cast<BlockArgument>())
        arg.getOwner()->getParentOp()->emitError(
            "reference dataflow cannot be traced back to the remote read op "
            "for module port '")
            << dyn_cast<FModuleOp>(arg.getOwner()->getParentOp())
                   .getPortName(arg.getArgNumber())
            << "'";
      else
        val.getDefiningOp()->emitOpError(
            "reference dataflow cannot be traced back to the remote read op");
      signalPassFailure();
      return None;
    }
    // The Reaching def at this value is not yet known. So create an empty entry
    // and record it.
    return addReachingSendsEntry(val, {});
  }

  void mergeDataflow(Value src, Value dst) {
    auto flowAtSrc = dataflowAt.find(src);
    // If now flow at source, then just copy the info from dst. The dst might
    // also not exist yet, in that case a new entry will be created.
    if (flowAtSrc == dataflowAt.end()) {
      dataflowAt[src] = getRemoteRefSend(dst).value();
      return;
    }
    // dataflow at src exists, so merge it if an entry already exists at dest.
    auto flowAtDst = dataflowAt.find(dst);
    if (flowAtDst == dataflowAt.end()) {
      dataflowAt[dst] = flowAtSrc->getSecond();
      return;
    }
    auto oldEntry = flowAtDst->getSecond();
    if (!refSendList[oldEntry].empty()) {
      // If a valid dataflow already reaches the destination, emit remark.
      // This happens when multiple connects attached to an input RefType
      // port, or multiple instantiation (which is already an error).
      dst.getDefiningOp()->emitRemark("has multipl reaching definitions");
    }
    // Dest already has an entry, make sure to update all the entries in the
    // dataflow map to point to this new entry.
    for (auto entr : dataflowAt) {
      if (entr.getSecond() == oldEntry) {
        dataflowAt[entr.getFirst()] = flowAtSrc->getSecond();
      }
    }
  }

  size_t addReachingSendsEntry(Value atRefVal, Attribute newRef,
                               Optional<size_t> copyFrom = None) {
    auto index = refSendList.size();
    dataflowAt[atRefVal] = index;
    if (copyFrom) {
      refSendList.emplace_back(refSendList[copyFrom.value()]);
      refSendList[index].push_back(newRef);
    } else if (newRef)
      refSendList.push_back({newRef});
    else
      refSendList.push_back({});
    return index;
  }

  void garbageCollect() {
    // Now erase all the Ops and ports of RefType.
    // This needs to be done as the last step to ensure uses are erased before
    // the def is erased.
    for (Operation *op : llvm::reverse(opsToRemove))
      op->erase();
    for (auto iter : refPortsToRemoveMap)
      if (auto mod = dyn_cast<FModuleOp>(iter.getFirst()))
        mod.erasePorts(iter.getSecond());
      else if (auto inst = dyn_cast<InstanceOp>(iter.getFirst())) {
        ImplicitLocOpBuilder b(inst.getLoc(), inst);
        inst.erasePorts(b, iter.getSecond());
        inst.erase();
      }
    opsToRemove.clear();
    refPortsToRemoveMap.clear();
    dataflowAt.clear();
    refSendList.clear();
  }

  /// Cached module namespaces.
  DenseMap<Operation *, ModuleNamespace> moduleNamespaces;

  DenseSet<Operation *> visitedModules;
  /// Map of a reference value to an entry into refSendList. Each entry in
  /// refSendList represents the path to RefSend is as an ArrayAttr, which is an
  /// array of the InnerRef to InstanceOps. The path is required since there can
  /// be multiple paths to the RefSend and we need to identify a unique path.
  /// Each ref value can be satically resolved to a single remote send op,
  /// according to the constraints on the RefType. The ArrayAttr ensures that
  /// only unique copies of the path exist.
  DenseMap<Value, size_t> dataflowAt;

  /// This is the set of all unique paths from a RefSendOp to RefResolveOps.
  /// Each entry is a unique path to a RefSendOp. In case this becomes a
  /// performance issue, store only the unique instance paths from the root to
  /// each RefResolve, and index into them.
  using instancePath = SmallVector<Attribute>;
  SmallVector<instancePath> refSendList;

  // Instance and module ref ports that needs to be removed.
  DenseMap<Operation *, SmallVector<unsigned>> refPortsToRemoveMap;

  /// RefResolve, RefSend, and Connects involving them that will be removed.
  SmallVector<Operation *> opsToRemove;
};

std::unique_ptr<mlir::Pass> circt::firrtl::createLowerXMRPass() {
  return std::make_unique<LowerXMRPass>();
}
