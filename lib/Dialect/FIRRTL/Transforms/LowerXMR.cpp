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
#include "circt/Dialect/SV/SVOps.h"
#include "mlir/IR/BuiltinOps.h"
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
    // Populate a CircuitNamespace that can be used to generate unique
    // circuit-level symbols.
    auto ns = CircuitNamespace(getOperation());
    circuitNamespace = &ns;

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
              if (!isa<hw::InnerSymbolOpInterface>(xmrDefOp) ||
                  /* No innner symbols for results of instances */
                  isa<InstanceOp>(xmrDefOp) ||
                  /* Similarly, anything with multiple results isn't named by
                     the inner sym */
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
              if (mem.getResult(res.index()).getType().isa<RefType>()) {
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
            auto defMem =
                dyn_cast_or_null<MemOp>(op.getInput().getDefiningOp());
            if (!defMem) {
              op.emitError("can only lower RefSubOp of Memory")
                      .attachNote(op.getInput().getLoc())
                  << "input here";
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
               init = refSendPathList[*init.second])
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
    if (resWidth.has_value() && *resWidth == 0) {
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
    SmallString<128> xmrString;
    size_t lastIndex;
    while (remoteOpPath) {
      lastIndex = *remoteOpPath;
      auto entr = refSendPathList[*remoteOpPath];
      refSendPath.push_back(entr.first);
      remoteOpPath = entr.second;
    }
    auto iter = xmrPathSuffix.find(lastIndex);

    // If this xmr has a suffix string (internal path into a module, that is not
    // yet generated).
    if (iter != xmrPathSuffix.end())
      xmrString += ("." + iter->getSecond()).str();

    // Compute the reference given to the SVXMRRefOp.  If the path is size 1,
    // then this is just an InnerRefAttr (module--component pair).  Otehrwise,
    // we need to use the symbol of a HierPathOp that stores the path.
    ImplicitLocOpBuilder builder(resolve.getLoc(), resolve);
    Attribute ref;
    if (refSendPath.size() == 1)
      ref = refSendPath.front();
    else
      ref = FlatSymbolRefAttr::get(
          getOrCreatePath(builder.getArrayAttr(refSendPath), builder)
              .getSymNameAttr());

    // Create the XMR op and replace users of the result of the resolve with the
    // result of the XMR.
    auto xmr = builder.create<sv::XMRRefOp>(
        sv::InOutType::get(lowerType(resolve.getType())), ref, xmrString);
    auto conversion = builder.create<mlir::UnrealizedConversionCastOp>(
        resolve.getType(), xmr.getResult());
    resolve.getResult().replaceAllUsesWith(conversion.getResult(0));
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
        if (!isa<RefType>(inst.getResult(res.index()).getType()))
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

  std::optional<size_t> getRemoteRefSend(Value val) {
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
    return std::nullopt;
  }

  size_t
  addReachingSendsEntry(Value atRefVal, Attribute newRef,
                        std::optional<size_t> continueFrom = std::nullopt) {
    auto leader = dataFlowClasses.getOrInsertLeaderValue(atRefVal);
    auto indx = refSendPathList.size();
    dataflowAt[leader] = indx;
    if (continueFrom.has_value()) {
      if (!refSendPathList[*continueFrom].first) {
        // This handles the case when the InnerRef is set to null at the
        // following path, that implies the path ends at this node, so copy the
        // xmrPathSuffix and end the path here.
        auto xmrIter = xmrPathSuffix.find(*continueFrom);
        if (xmrIter != xmrPathSuffix.end()) {
          SmallString<128> xmrSuffix = xmrIter->getSecond();
          // The following assignment to the DenseMap can potentially reallocate
          // the map, that might invalidate the `xmrIter`. So, copy the result
          // to a temp, and then insert it back to the Map.
          xmrPathSuffix[indx] = xmrSuffix;
        }
        continueFrom = std::nullopt;
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
          if (isa<RefType>(mem.getResult(res.index()).getType()))
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
            mem.getGroupIDAttr(), mem.getInitAttr());
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
  using nextNodeOnPath = std::optional<size_t>;
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

  CircuitNamespace *circuitNamespace;

  /// A cache of already created HierPathOps.  This is used to avoid repeatedly
  /// creating the same HierPathOp.
  DenseMap<Attribute, hw::HierPathOp> pathCache;

  /// The insertion point where the pass inserts HierPathOps.
  OpBuilder::InsertPoint pathInsertPoint = {};

  /// Return a HierPathOp for the provided pathArray.  This will either return
  /// an existing HierPathOp or it will create and return a new one.
  hw::HierPathOp getOrCreatePath(ArrayAttr pathArray,
                                 ImplicitLocOpBuilder &builder) {
    // Return an existing HierPathOp if one exists with the same path.
    auto pathIter = pathCache.find(pathArray);
    if (pathIter != pathCache.end())
      return pathIter->second;

    // Reset the insertion point after this function returns.
    OpBuilder::InsertionGuard guard(builder);

    // Set the insertion point to either the known location where the pass
    // inserts HierPathOps or to the start of the circuit.
    if (pathInsertPoint.isSet())
      builder.restoreInsertionPoint(pathInsertPoint);
    else
      builder.setInsertionPointToStart(getOperation().getBodyBlock());

    // Create the new HierPathOp and insert it into the pathCache.
    hw::HierPathOp path =
        pathCache
            .insert({pathArray,
                     builder.create<hw::HierPathOp>(
                         circuitNamespace->newName("xmrPath"), pathArray)})
            .first->second;
    path.setVisibility(SymbolTable::Visibility::Private);

    // Save the insertion point so other unique HierPathOps will be created
    // after this one.
    pathInsertPoint = builder.saveInsertionPoint();

    // Return the new path.
    return path;
  }
};

std::unique_ptr<mlir::Pass> circt::firrtl::createLowerXMRPass() {
  return std::make_unique<LowerXMRPass>();
}
