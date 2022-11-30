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
#include "llvm/ADT/BitVector.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/EquivalenceClasses.h"
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
    dataFlowClasses = llvm::EquivalenceClasses<Value, ValueComparator>();
    InstanceGraph &instanceGraph = getAnalysis<InstanceGraph>();
    SmallVector<RefResolveOp> resolveOps;
    // The dataflow function, that propagates the reachable RefSendOp across
    // RefType Ops.
    auto transferFunc = [&](Operation &op) -> LogicalResult {
      return TypeSwitch<Operation *, LogicalResult>(&op)
          .Case<RefSendOp>([&](RefSendOp send) {
            // Get a reference to the actual signal to which the XMR will be
            // generated.
            Value xmrDef = send.getBase();
            if (isZeroWidth(send.getType().getType())) {
              markForRemoval(send);
              return success();
            }
            // Get an InnerRefAttr to the xmrDef op. If the operation does not
            // take any InnerSym (like firrtl.add, firrtl.or etc) then create a
            // NodeOp to add the InnerSym.
            if (!xmrDef.isa<BlockArgument>()) {
              Operation *xmrDefOp = xmrDef.getDefiningOp();
              if (auto verbExpr = dyn_cast<VerbatimExprOp>(xmrDefOp))
                if (verbExpr.getSymbolsAttr().empty() &&
                    xmrDefOp->hasOneUse()) {
                  // This represents the internal path into a module. For
                  // generating the correct XMR, no node can be created in this
                  // module. Create a null InnerRef and ensure the hierarchical
                  // path ends at the parent that instantiates this module.
                  auto inRef = InnerRefAttr();
                  auto ind = addReachingSendsEntry(send.getResult(), inRef);
                  xmrPathSuffix[ind] = verbExpr.getText();
                  markForRemoval(verbExpr);
                  markForRemoval(send);
                  return success();
                }
              if (!isa<InnerSymbolOpInterface>(xmrDefOp) ||
                  xmrDefOp->getResults().size() > 1) {
                // Add a node, for non-innerSym ops. Otherwise the sym will be
                // dropped after LowerToHW.
                // If the op has multiple results, we cannot add symbol to a
                // single result, so create a node from the result and add
                // symbol to the node.
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
          .Case<MemOp>([&](MemOp mem) {
            // MemOp can produce debug ports of RefType. Each debug port
            // represents the RefType for the corresponding register of the
            // memory. Since the memory is not yet generated the register name
            // is assumed to be "Memory". Note that MemOp creates RefType
            // without a RefSend.
            for (const auto &res : llvm::enumerate(mem.getResults()))
              if (mem.getResult(res.index())
                      .getType()
                      .cast<FIRRTLType>()
                      .isa<RefType>()) {
                auto inRef = getInnerRefTo(mem);
                auto ind = addReachingSendsEntry(res.value(), inRef);
                xmrPathSuffix[ind] = "Memory";
                // Just node that all the debug ports of memory must be removed.
                // So this does not record the port index.
                refPortsToRemoveMap[mem].resize(1);
              }
            return success();
          })
          .Case<InstanceOp>(
              [&](auto inst) { return handleInstanceOp(inst, instanceGraph); })
          .Case<FConnectLike>([&](FConnectLike connect) {
            // Ignore BaseType.
            if (!connect.getSrc().getType().isa<RefType>())
              return success();
            markForRemoval(connect);
            if (isZeroWidth(
                    connect.getSrc().getType().cast<RefType>().getType()))
              return success();
            // Merge the dataflow classes of destination into the source of the
            // Connect. This handles two cases:
            // 1. If the dataflow at the source is known, then the
            // destination is also inferred. By merging the dataflow class of
            // destination with source, every value reachable from the
            // destination automatically infers a reaching RefSend.
            // 2. If dataflow at source is unkown, then just record that both
            // source and destination will have the same dataflow information.
            // Later in the pass when the reaching RefSend is inferred at the
            // leader of the dataflowClass, then we automatically infer the
            // dataflow at this connect and every value reachable from the
            // destination.
            dataFlowClasses.unionSets(connect.getSrc(), connect.getDest());
            return success();
          })
          .Case<RefSubOp>([&](RefSubOp op) {
            markForRemoval(op);
            if (isZeroWidth(op.getType().getType()))
              return success();
            auto defMem = dyn_cast<MemOp>(op.getInput().getDefiningOp());
            if (!defMem) {
              defMem.emitOpError("can only lower RefSubOp of Memory");
              return failure();
            }
            auto inRef = getInnerRefTo(defMem);
            auto ind = addReachingSendsEntry(op.getResult(), inRef);
            xmrPathSuffix[ind] = ("Memory[" + Twine(op.getIndex()) + "]").str();

            return success();
          })
          .Case<RefResolveOp>([&](RefResolveOp resolve) {
            // Merge dataflow, under the same conditions as above for Connect.
            // 1. If dataflow at the resolve.getRef is known, propagate that to
            // the result. This is true for downward scoped XMRs, that is,
            // RefSendOp must be visited before the corresponding RefResolveOp
            // is visited.
            // 2. Else, just record that both result and ref should have the
            // same reaching RefSend. This condition is true for upward scoped
            // XMRs. That is, RefResolveOp can be visited before the
            // corresponding RefSendOp is recorded.

            markForRemoval(resolve);
            if (!isZeroWidth(resolve.getType()))
              dataFlowClasses.unionSets(resolve.getRef(), resolve.getResult());
            resolveOps.push_back(resolve);
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
                 << "Traversing module:" << module.moduleNameAttr() << "\n");
      for (Operation &op : module.getBodyBlock()->getOperations())
        if (transferFunc(op).failed())
          return signalPassFailure();

      // Record all the RefType ports to be removed later.
      size_t numPorts = module.getNumPorts();
      for (size_t portNum = 0; portNum < numPorts; ++portNum)
        if (module.getPortType(portNum).isa<RefType>()) {
          setPortToRemove(module, portNum, numPorts);
        }
    }

    LLVM_DEBUG({
      for (auto I = dataFlowClasses.begin(), E = dataFlowClasses.end(); I != E;
           ++I) { // Iterate over all of the equivalence sets.
        if (!I->isLeader())
          continue; // Ignore non-leader sets.
        // Print members in this set.
        llvm::interleave(llvm::make_range(dataFlowClasses.member_begin(I),
                                          dataFlowClasses.member_end()),
                         llvm::dbgs(), "\n");
        llvm::dbgs() << "\n dataflow at leader::" << I->getData() << "\n =>";
        auto iter = dataflowAt.find(I->getData());
        if (iter != dataflowAt.end()) {
          for (auto init = refSendPathList[iter->getSecond()]; init.second;
               init = refSendPathList[init.second.value()])
            llvm::dbgs() << "\n path ::" << init.first << "::" << init.second;
        }
        llvm::dbgs() << "\n Done\n"; // Finish set.
      }
    });
    for (auto refResolve : resolveOps)
      if (handleRefResolve(refResolve).failed())
        return signalPassFailure();
    garbageCollect();
  }

  // Replace the RefResolveOp with verbatim op representing the XMR.
  LogicalResult handleRefResolve(RefResolveOp resolve) {
    auto resWidth = getBitWidth(resolve.getType());
    if (resWidth.has_value() && resWidth.value() == 0) {
      // Donot emit 0 width XMRs, replace it with constant 0.
      ImplicitLocOpBuilder builder(resolve.getLoc(), resolve);
      auto zeroUintType = UIntType::get(builder.getContext(), 0);
      auto zeroC = builder.createOrFold<BitCastOp>(
          resolve.getType(), builder.create<ConstantOp>(
                                 zeroUintType, getIntZerosAttr(zeroUintType)));
      resolve.getResult().replaceAllUsesWith(zeroC);
      return success();
    }
    auto remoteOpPath = getRemoteRefSend(resolve.getRef());
    if (!remoteOpPath)
      return failure();
    SmallVector<Attribute> refSendPath;
    // Verbatim XMR begins with the Top level module.
    refSendPath.push_back(refSendPathList[remoteOpPath.value()]
                              .first.cast<InnerRefAttr>()
                              .getModuleRef());
    SmallString<128> xmrString;
    Attribute lastInnerRef;
    size_t lastIndex;
    unsigned index = 0;
    for (; remoteOpPath; ++index) {
      lastIndex = remoteOpPath.value();
      auto entr = refSendPathList[remoteOpPath.value()];
      refSendPath.push_back(entr.first);
      lastInnerRef = entr.first;
      remoteOpPath = entr.second;
      ("{{" + Twine(index) + "}}").toVector(xmrString);
      xmrString += '.';
    }
    ("{{" + Twine(index) + "}}").toVector(xmrString);
    auto iter = xmrPathSuffix.find(lastIndex);
    // If this xmr has a suffix string (internal path into a module, that is not
    // yet generated).
    if (iter != xmrPathSuffix.end())
      xmrString += ("." + iter->getSecond()).str();
    if (auto vec = resolve.getResult().getType().dyn_cast<FVectorType>()) {
      // If the RefType is a vector, then replace all its users with [i] suffix,
      // instead of creatign a temp wire to the vector xmr, and then followup
      // index into it.
      bool hasOtherUses = false;
      for (Operation *user : resolve.getResult().getUsers()) {
        if (auto sub = dyn_cast<SubindexOp>(user)) {
          auto index = sub.getIndex();
          ImplicitLocOpBuilder builder(sub.getLoc(), sub);
          auto xmrVerbatim = builder.create<VerbatimExprOp>(
              vec.getElementType(), xmrString + "[" + Twine(index) + "]",
              ValueRange{}, refSendPath);
          sub.getResult().replaceAllUsesWith(xmrVerbatim);
          opsToRemove.push_back(sub);
        } else
          hasOtherUses = true;
      }
      if (!hasOtherUses)
        return success();
    }

    // The source of the dataflow for this RefResolveOp is established. So
    // replace the RefResolveOp with the coresponding VerbatimExpr to
    // generate the XMR.
    ImplicitLocOpBuilder builder(resolve.getLoc(), resolve);
    auto xmrVerbatim =
        builder.create<VerbatimExprOp>(resolve.getType().cast<FIRRTLType>(),
                                       xmrString, ValueRange{}, refSendPath);
    resolve.getResult().replaceAllUsesWith(xmrVerbatim);
    return success();
  }

  void setPortToRemove(Operation *op, size_t index, size_t numPorts) {
    if (refPortsToRemoveMap[op].size() < numPorts)
      refPortsToRemoveMap[op].resize(numPorts);
    refPortsToRemoveMap[op].set(index);
  }

  // Propagate the reachable RefSendOp across modules.
  LogicalResult handleInstanceOp(InstanceOp inst,
                                 InstanceGraph &instanceGraph) {
    Operation *mod = instanceGraph.getReferencedModule(inst);
    if (auto extRefMod = dyn_cast<FExtModuleOp>(mod)) {
      // Extern modules can generate RefType ports, they have an attached
      // attribute which specifies the internal path into the extern module.
      // This string attribute will be used to generate the final xmr.
      auto internalPaths = extRefMod.getInternalPaths();
      // No internalPaths implies no RefType ports.
      if (internalPaths.empty())
        return success();
      size_t pathsIndex = 0;
      auto numPorts = inst.getNumResults();
      for (const auto &res : llvm::enumerate(inst.getResults())) {
        if (!inst.getResult(res.index())
                 .getType()
                 .cast<FIRRTLType>()
                 .isa<RefType>())
          continue;

        auto inRef = getInnerRefTo(inst);
        auto ind = addReachingSendsEntry(res.value(), inRef);

        xmrPathSuffix[ind] = internalPaths[pathsIndex].cast<StringAttr>().str();
        ++pathsIndex;
        // The instance result and module port must be marked for removal.
        setPortToRemove(inst, res.index(), numPorts);
        setPortToRemove(extRefMod, res.index(), numPorts);
      }
      return success();
    }
    auto refMod = dyn_cast<FModuleOp>(mod);
    bool multiplyInstantiated = !visitedModules.insert(refMod).second;
    for (size_t portNum = 0, numPorts = inst.getNumResults();
         portNum < numPorts; ++portNum) {
      auto instanceResult = inst.getResult(portNum);
      if (!instanceResult.getType().isa<RefType>())
        continue;
      if (!refMod)
        return inst.emitOpError("cannot lower ext modules with RefType ports");
      // Reference ports must be removed.
      setPortToRemove(inst, portNum, numPorts);
      // Drop the dead-instance-ports.
      if (instanceResult.use_empty() ||
          isZeroWidth(instanceResult.getType().cast<RefType>().getType()))
        continue;
      auto refModuleArg = refMod.getArgument(portNum);
      if (inst.getPortDirection(portNum) == Direction::Out) {
        // For output instance ports, the dataflow is into this module.
        // Get the remote RefSendOp, that flows through the module ports.
        // If dataflow at remote module argument does not exist, error out.
        auto remoteOpPath = getRemoteRefSend(refModuleArg);
        if (!remoteOpPath)
          return failure();
        // Get the path to reaching refSend at the referenced module argument.
        // Now append this instance to the path to the reaching refSend.
        addReachingSendsEntry(instanceResult, getInnerRefTo(inst),
                              remoteOpPath);
      } else {
        // For input instance ports, the dataflow is into the referenced module.
        // Input RefType port implies, generating an upward scoped XMR.
        // No need to add the instance context, since downward reference must be
        // through single instantiated modules.
        if (multiplyInstantiated)
          return refMod.emitOpError(
                     "multiply instantiated module with input RefType port '")
                 << refMod.getPortName(portNum) << "'";
        dataFlowClasses.unionSets(
            dataFlowClasses.getOrInsertLeaderValue(refModuleArg),
            dataFlowClasses.getOrInsertLeaderValue(instanceResult));
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

  Optional<size_t> getRemoteRefSend(Value val) {
    auto iter = dataflowAt.find(dataFlowClasses.getOrInsertLeaderValue(val));
    if (iter != dataflowAt.end())
      return iter->getSecond();
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

  size_t addReachingSendsEntry(Value atRefVal, Attribute newRef,
                               Optional<size_t> continueFrom = None) {
    auto leader = dataFlowClasses.getOrInsertLeaderValue(atRefVal);
    auto indx = refSendPathList.size();
    dataflowAt[leader] = indx;
    if (continueFrom.has_value()) {
      if (!refSendPathList[continueFrom.value()].first) {
        // This handles the case when the InnerRef is set to null at the
        // following path, that implies the path ends at this node, so copy the
        // xmrPathSuffix and end the path here.
        auto xmrIter = xmrPathSuffix.find(continueFrom.value());
        if (xmrIter != xmrPathSuffix.end()) {
          SmallString<128> xmrSuffix = xmrIter->getSecond();
          // The following assignment to the DenseMap can potentially reallocate
          // the map, that might invalidate the `xmrIter`. So, copy the result
          // to a temp, and then insert it back to the Map.
          xmrPathSuffix[indx] = xmrSuffix;
        }
        continueFrom = None;
      }
    }
    refSendPathList.push_back(std::make_pair(newRef, continueFrom));
    return indx;
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
      else if (auto mod = dyn_cast<FExtModuleOp>(iter.getFirst()))
        mod.erasePorts(iter.getSecond());
      else if (auto inst = dyn_cast<InstanceOp>(iter.getFirst())) {
        ImplicitLocOpBuilder b(inst.getLoc(), inst);
        inst.erasePorts(b, iter.getSecond());
        inst.erase();
      } else if (auto mem = dyn_cast<MemOp>(iter.getFirst())) {
        // Remove all debug ports of the memory.
        ImplicitLocOpBuilder builder(mem.getLoc(), mem);
        SmallVector<Attribute, 4> resultNames;
        SmallVector<Type, 4> resultTypes;
        SmallVector<Attribute, 4> portAnnotations;
        SmallVector<Value, 4> oldResults;
        for (const auto &res : llvm::enumerate(mem.getResults())) {
          if (mem.getResult(res.index())
                  .getType()
                  .cast<FIRRTLType>()
                  .isa<RefType>())
            continue;
          resultNames.push_back(mem.getPortName(res.index()));
          resultTypes.push_back(res.value().getType());
          portAnnotations.push_back(mem.getPortAnnotation(res.index()));
          oldResults.push_back(res.value());
        }
        auto newMem = builder.create<MemOp>(
            resultTypes, mem.getReadLatency(), mem.getWriteLatency(),
            mem.getDepth(), RUWAttr::Undefined,
            builder.getArrayAttr(resultNames), mem.getNameAttr(),
            mem.getNameKind(), mem.getAnnotations(),
            builder.getArrayAttr(portAnnotations), mem.getInnerSymAttr(),
            mem.getGroupIDAttr());
        for (const auto &res : llvm::enumerate(oldResults))
          res.value().replaceAllUsesWith(newMem.getResult(res.index()));
        mem.erase();
      }
    opsToRemove.clear();
    refPortsToRemoveMap.clear();
    dataflowAt.clear();
    refSendPathList.clear();
  }

  bool isZeroWidth(FIRRTLBaseType t) { return t.getBitWidthOrSentinel() == 0; }

  /// Cached module namespaces.
  DenseMap<Operation *, ModuleNamespace> moduleNamespaces;

  DenseSet<Operation *> visitedModules;
  /// Map of a reference value to an entry into refSendPathList. Each entry in
  /// refSendPathList represents the path to RefSend.
  /// The path is required since there can be multiple paths to the RefSend and
  /// we need to identify a unique path.
  DenseMap<Value, size_t> dataflowAt;

  /// refSendPathList is used to construct a path to the RefSendOp. Each entry
  /// is a node, with an InnerRefAttr and a pointer to the next node in the
  /// path. The InnerRefAttr can be to an InstanceOp or to the XMR defining
  /// op. All the nodes representing an InstanceOp must have a valid
  /// nextNodeOnPath. Only the node representing the final XMR defining op has
  /// no nextNodeOnPath, which denotes a leaf node on the path.
  using nextNodeOnPath = Optional<size_t>;
  using node = std::pair<Attribute, nextNodeOnPath>;
  SmallVector<node> refSendPathList;

  /// llvm::EquivalenceClasses wants comparable elements. This comparator uses
  /// uses pointer comparison on the Impl.
  struct ValueComparator {
    bool operator()(const Value &lhs, const Value &rhs) const {
      return lhs.getImpl() < rhs.getImpl();
    }
  };

  llvm::EquivalenceClasses<Value, ValueComparator> dataFlowClasses;
  // Instance and module ref ports that needs to be removed.
  DenseMap<Operation *, llvm::BitVector> refPortsToRemoveMap;

  /// RefResolve, RefSend, and Connects involving them that will be removed.
  SmallVector<Operation *> opsToRemove;

  /// Record the internal path to an external module or a memory.
  DenseMap<size_t, SmallString<128>> xmrPathSuffix;
};

std::unique_ptr<mlir::Pass> circt::firrtl::createLowerXMRPass() {
  return std::make_unique<LowerXMRPass>();
}
