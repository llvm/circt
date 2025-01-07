//===- TriggersToSV.cpp - Sim to SV lowering ------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// TODO
//
//===----------------------------------------------------------------------===//

#include "circt/Conversion/TriggersToSV.h"
#include "circt/Dialect/Comb/CombOps.h"
#include "circt/Dialect/HW/HWAttributes.h"
#include "circt/Dialect/HW/HWInstanceGraph.h"
#include "circt/Dialect/HW/HWOps.h"
#include "circt/Dialect/HW/InnerSymbolNamespace.h"
#include "circt/Dialect/SV/SVOps.h"
#include "circt/Dialect/Sim/SimDialect.h"
#include "circt/Dialect/Sim/SimOps.h"
#include "circt/Support/Namespace.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/ImplicitLocOpBuilder.h"
#include "mlir/IR/Threading.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"

#define DEBUG_TYPE "lower-triggers-to-sv"

namespace circt {
#define GEN_PASS_DEF_LOWERTRIGGERSTOSV
#include "circt/Conversion/Passes.h.inc"
} // namespace circt

using namespace circt;
using namespace sim;

namespace circt::sim::detail {
using IfDefGuardMap = llvm::SmallDenseMap<hw::HWModuleOp, Block *>;
}

static inline bool isTriggerType(const mlir::Type ty) {
  return isa<sim::InitTriggerType, sim::EdgeTriggerType>(ty);
}

static inline sv::EventControl convertEventControl(hw::EventControl eventCtrl) {
  switch (eventCtrl) {
  case hw::EventControl::AtEdge:
    return sv::EventControl::AtEdge;
  case hw::EventControl::AtPosEdge:
    return sv::EventControl::AtPosEdge;
  case hw::EventControl::AtNegEdge:
    return sv::EventControl::AtNegEdge;
  }
  assert(false && "Invalid event control attr");
}

static bool isBitToTriggerCast(mlir::UnrealizedConversionCastOp castOp) {
  if (!castOp || castOp.getNumResults() != 1 || castOp.getNumOperands() != 1)
    return false;
  auto ty = castOp.getResult(0).getType();
  if (!isTriggerType(ty))
    return false;
  if (auto inout = dyn_cast<sv::InOutType>(castOp.getOperand(0).getType())) {
    if (!isa<IntegerType>(inout.getElementType()))
      return false;
    return inout.getElementType().getIntOrFloatBitWidth() == 1;
  }
  return false;
}

static void collectTriggeredOps(Value root,
                                SmallVector<sim::TriggeredOp> &trigOps) {
  SmallVector<Value> worklist;
  worklist.push_back(root);
  while (!worklist.empty()) {
    auto workItem = worklist.pop_back_val();
    for (auto user : workItem.getUsers()) {
      if (auto trigOp = dyn_cast<sim::TriggeredOp>(user)) {
        trigOps.push_back(trigOp);
      } else if (auto trigOp = dyn_cast<sim::TriggerSequenceOp>(user)) {
        worklist.append(trigOp.getResults().begin(), trigOp.getResults().end());
      } else if (auto gateOp = dyn_cast<sim::TriggerGateOp>(user)) {
        worklist.push_back(gateOp.getResult());
      } else {
        user->emitWarning("Unexpected trigger user op.");
      }
    }
  }
}

static Value materializeInitValue(OpBuilder &builder, TypedAttr cst,
                                  Location loc) {
  if (llvm::isa<sv::SVDialect>(cst.getDialect()))
    return cst.getDialect()
        .materializeConstant(builder, cst, cst.getType(), loc)
        ->getResult(0);
  auto *hwDialect = builder.getContext()->getLoadedDialect<hw::HWDialect>();
  return hwDialect->materializeConstant(builder, cst, cst.getType(), loc)
      ->getResult(0);
}

static LogicalResult stripTriggerIOs(hw::HWModuleOp moduleOp,
                                     hw::InstanceGraph &igraph) {
  auto isTriggerType = [](Type ty) -> bool {
    return isa<sim::EdgeTriggerType, sim::InitTriggerType>(ty);
  };
  BitVector argsToRemove(moduleOp.getBody().getNumArguments());

  for (auto [i, arg] : llvm::enumerate(moduleOp.getBody().getArguments())) {
    if (isTriggerType(arg.getType())) {
      argsToRemove.set(i);
      if (!arg.use_empty()) {
        OpBuilder builder(moduleOp);
        builder.setInsertionPointToStart(moduleOp.getBodyBlock());
        auto loc = moduleOp.getPortLoc(moduleOp.getPortIdForInputId(i));
        auto never = builder.create<sim::NeverOp>(loc, arg.getType());
        arg.replaceAllUsesWith(never);
      }
    }
  }

  moduleOp.getBodyBlock()->eraseArguments(argsToRemove);
  auto outputOp = cast<hw::OutputOp>(moduleOp.getBodyBlock()->getTerminator());
  BitVector outputsToRemove(outputOp.getNumOperands());

  for (auto [i, outTy] : llvm::enumerate(outputOp.getOperandTypes()))
    if (isTriggerType(outTy))
      outputsToRemove.set(i);
  outputOp->eraseOperands(outputsToRemove);

  SmallVector<hw::ModulePort> newPorts;
  for (auto port : moduleOp.getModuleType().getPorts())
    if (!isTriggerType(port.type))
      newPorts.push_back(port);
  moduleOp.setModuleType(hw::ModuleType::get(moduleOp.getContext(), newPorts));

  auto *node = igraph.lookup(moduleOp);
  for (auto instUse : llvm::make_early_inc_range(node->uses())) {
    auto instanceOp = llvm::cast<hw::InstanceOp>(
        instUse->getInstance<hw::HWInstanceLike>().getOperation());

    SmallVector<Value> newOperands;
    SmallVector<Attribute> newNamesAttr;
    for (auto [operand, attr] :
         llvm::zip(instanceOp.getOperands(), instanceOp.getArgNamesAttr())) {
      if (!isTriggerType(operand.getType())) {
        newOperands.push_back(operand);
        newNamesAttr.push_back(attr);
      }
    }
    ImplicitLocOpBuilder builder(instanceOp.getLoc(), instanceOp);
    auto newInstance = builder.create<hw::InstanceOp>(
        outputOp.getOperandTypes(), instanceOp.getInstanceNameAttr(),
        instanceOp.getModuleNameAttr(), newOperands,
        builder.getArrayAttr(newNamesAttr), ArrayAttr{},
        instanceOp.getParametersAttr(), instanceOp.getInnerSymAttr(),
        instanceOp.getDoNotPrintAttr());
    newNamesAttr.clear();
    size_t newResIdx = 0;
    for (auto [idx, res] : llvm::enumerate(instanceOp.getResults())) {
      if (outputsToRemove[idx]) {
        auto never = builder.createOrFold<sim::NeverOp>(res.getType());
        res.replaceAllUsesWith(never);
      } else {
        res.replaceAllUsesWith(newInstance.getResult(newResIdx));
        newNamesAttr.push_back(instanceOp.getOutputNames()[idx]);
        newResIdx++;
      }
    }
    newInstance.setOutputNames(builder.getArrayAttr(newNamesAttr));
    instanceOp.erase();
  }
  return success();
}

namespace {

struct GlobalTriggersToSVPass {

  GlobalTriggersToSVPass(mlir::ModuleOp moduleOp,
                         detail::IfDefGuardMap &guardMap)
      : mlirModuleOp(moduleOp), ifdefGuardBlocks(guardMap) {
    for (auto hwModuleOp : mlirModuleOp.getOps<hw::HWModuleOp>())
      if (llvm::any_of(hwModuleOp.getPortTypes(), isTriggerType))
        crossTriggerModules.insert(hwModuleOp);
  };

  size_t getNumCrosstriggerModules() const {
    return crossTriggerModules.size();
  }
  LogicalResult run(hw::InstanceGraph &igraph) {
    if (crossTriggerModules.empty())
      return success();

    bool hasNonPrivateTriggersPorts = false;
    for (auto hwModuleOp : crossTriggerModules) {
      if (!hwModuleOp.isPrivate()) {
        hwModuleOp.emitError(
            "Trigger type in port list requires module to be private.");
        hasNonPrivateTriggersPorts = true;
        continue;
      }
    }
    if (hasNonPrivateTriggersPorts)
      return failure();

    bool hasFailed = false;
    for (auto crossTriggerMod : crossTriggerModules)
      hasFailed |= failed(lowerCrossModuleTrigger(crossTriggerMod, igraph));

    if (hasFailed)
      return failure();

    for (auto crossTriggerMod : crossTriggerModules)
      hasFailed |= failed(stripTriggerIOs(crossTriggerMod, igraph));

    if (hasFailed)
      return failure();
    return success();
  }

private:
  using ArgNumToSymMap =
      SmallDenseMap<std::pair<hw::ModulePort::Direction, size_t>,
                    hw::InnerRefAttr>;

  std::optional<Namespace> globalNamespaceCache;
  Namespace &getGlobalNamespace() {
    if (!globalNamespaceCache.has_value()) {
      globalNamespaceCache.emplace();
      SymbolCache cache;
      cache.addDefinitions(mlirModuleOp);
      globalNamespaceCache->add(cache);
    }
    return *globalNamespaceCache;
  }

  Block *getOrCreateGuardBlock(hw::HWModuleOp &moduleOp) {
    Block *guardBlock = ifdefGuardBlocks[moduleOp];
    if (!guardBlock) {
      OpBuilder builder(moduleOp.getBodyBlock()->getTerminator());
      auto ifDef = builder.create<sv::IfDefOp>(
          moduleOp.getLoc(), "SYNTHESIS", [] {}, [] {});
      guardBlock = ifDef.getElseBlock();
      ifdefGuardBlocks[moduleOp] = guardBlock;
    }
    return guardBlock;
  }

  LogicalResult lowerCrossModuleTrigger(hw::HWModuleOp moduleOp,
                                        hw::InstanceGraph &igraph);
  LogicalResult
  lowerCrossModuleTriggerParent(igraph::InstanceRecord *instRecord,
                                ArrayRef<hw::PortInfo> triggerPorts,
                                const ArgNumToSymMap &argNumToInnerNames);

  mlir::ModuleOp mlirModuleOp;
  llvm::SmallSetVector<hw::HWModuleOp, 0> crossTriggerModules;
  detail::IfDefGuardMap &ifdefGuardBlocks;
};

struct TriggersToSVPass
    : public circt::impl::LowerTriggersToSVBase<TriggersToSVPass> {

  using circt::impl::LowerTriggersToSVBase<
      TriggersToSVPass>::LowerTriggersToSVBase;

  static constexpr bool useDelayedInitProcesses = true;

  void runOnOperation() override {

    ifdefGuardBlocks.clear();
    auto globalLowering = std::make_unique<GlobalTriggersToSVPass>(
        getOperation(), ifdefGuardBlocks);
    bool hasCrosstriggers = globalLowering->getNumCrosstriggerModules() > 0;
    if (hasCrosstriggers) {
      auto &igraph = getAnalysis<hw::InstanceGraph>();
      if (failed(globalLowering->run(igraph))) {
        signalPassFailure();
        return;
      }
    } else {
      markAnalysesPreserved<hw::InstanceGraph>();
    }

    std::atomic<bool> anyChange = hasCrosstriggers;
    auto result = mlir::failableParallelForEach(
        &getContext(), getOperation().getOps<hw::HWModuleOp>(),
        [&](hw::HWModuleOp op) -> LogicalResult {
          size_t numRoots = 0;
          if (failed(lowerLocalTriggers(op, numRoots)))
            return failure();
          if (numRoots > 0)
            anyChange = true;
          return success();
        });

    if (failed(result))
      signalPassFailure();

    if (anyChange) {
      Operation *op = getOperation().lookupSymbol("SYNTHESIS");
      if (op) {
        if (!isa<sv::MacroDeclOp>(op)) {
          op->emitOpError("should be a macro declaration");
          return signalPassFailure();
        }
      } else {
        auto builder = ImplicitLocOpBuilder::atBlockBegin(
            UnknownLoc::get(&getContext()), getOperation().getBody());
        builder.create<sv::MacroDeclOp>("SYNTHESIS");
      }
    } else {
      markAllAnalysesPreserved();
    }
  }

private:
  detail::IfDefGuardMap ifdefGuardBlocks;
  LogicalResult lowerRootTrigger(Value root,
                                 SmallVectorImpl<TriggeredOp> &lowerdOps,
                                 Block *guardBlock);
  LogicalResult lowerLocalTriggers(hw::HWModuleOp moduleOp, size_t &numRoots);
  LogicalResult buildSVProcessForRoot(OpBuilder &builder, Operation *rootOp,
                                      Location loc);
};

LogicalResult GlobalTriggersToSVPass::lowerCrossModuleTriggerParent(
    igraph::InstanceRecord *instRecord, ArrayRef<hw::PortInfo> triggerPorts,
    const ArgNumToSymMap &argNumToInnerNames) {
  auto instanceOp = llvm::cast<hw::InstanceOp>(
      instRecord->getInstance<hw::HWInstanceLike>().getOperation());
  auto instParentMod = instRecord->getParent()->getModule<hw::HWModuleOp>();

  auto innerName = instanceOp.getInnerName();
  if (!innerName) {
    instanceOp.emitOpError("requires an inner name to lower cross trigger.");
    return failure();
  }

  auto guardBlock = getOrCreateGuardBlock(instParentMod);
  ImplicitLocOpBuilder builder(instanceOp.getLoc(), instanceOp);
  builder.setInsertionPointToEnd(guardBlock);

  for (auto trigPort : triggerPorts) {
    assert(isa<sim::EdgeTriggerType>(trigPort.type) ||
           isa<sim::InitTriggerType>(trigPort.type));

    auto &activeSigSym = argNumToInnerNames.at({trigPort.dir, trigPort.argNum});
    hw::HierPathOp triggerActivePathOp;
    {
      OpBuilder::InsertionGuard g(builder);
      builder.setInsertionPointToStart(mlirModuleOp.getBody());
      triggerActivePathOp = builder.create<hw::HierPathOp>(
          getGlobalNamespace().newName(
              "crosstrigger_path_" + instParentMod.getName() + "_" +
              *innerName + "_" + trigPort.getName() + "active"),
          builder.getArrayAttr({instanceOp.getInnerRef(), activeSigSym}));
    }

    Value triggerActiveXmr = builder.createOrFold<sv::XMRRefOp>(
        hw::InOutType::get(builder.getI1Type()),
        triggerActivePathOp.getSymNameAttr());

    if (trigPort.isInput()) {
      auto ssaValue = instanceOp.getInputs()[trigPort.argNum];
      assert(ssaValue.getType() == trigPort.type);

      instanceOp->setOperand(trigPort.argNum,
                             builder.createOrFold<sim::NeverOp>(trigPort.type));

      builder.create<sim::TriggeredOp>(ssaValue, triggerActiveXmr, [&]() {
        auto cst1 = builder.createOrFold<hw::ConstantOp>(
            IntegerAttr::get(builder.getI1Type(), 1));
        builder.create<sv::BPAssignOp>(
            builder.getInsertionBlock()->getArgument(0), cst1);
        auto readOp = builder.createOrFold<sv::ReadInOutOp>(
            builder.getInsertionBlock()->getArgument(0));
        auto notActiveOp =
            builder.createOrFold<comb::XorOp>(readOp, cst1, true);
        builder.create<sv::WaitOp>(notActiveOp);
      });
    } else {
      assert(trigPort.isOutput());
      auto convCast = builder.create<mlir::UnrealizedConversionCastOp>(
          TypeRange{trigPort.type}, triggerActiveXmr);
      auto res = instanceOp.getResult(trigPort.argNum);
      res.replaceAllUsesWith(convCast.getResult(0));
    }
  }

  return success();
}

LogicalResult
GlobalTriggersToSVPass::lowerCrossModuleTrigger(hw::HWModuleOp moduleOp,
                                                hw::InstanceGraph &igraph) {
  auto innerNamespace = hw::InnerSymbolNamespace(moduleOp);
  SmallVector<hw::PortInfo> triggerPorts;

  auto createActiveSigName = [&](StringRef portName) -> StringRef {
    return innerNamespace.newName("crosstrigger_" + portName, "active");
  };

  ArgNumToSymMap argNumToInnerNames;

  for (auto [idx, portInfo] : llvm::enumerate(moduleOp.getPortList())) {
    if (!isa<sim::EdgeTriggerType, sim::InitTriggerType>(portInfo.type))
      continue;
    if (portInfo.dir == hw::ModulePort::Direction::InOut)
      continue;

    triggerPorts.push_back(portInfo);

    auto guardBlock = getOrCreateGuardBlock(moduleOp);
    ImplicitLocOpBuilder builder(moduleOp.getPortLoc(idx), moduleOp);
    builder.setInsertionPointToStart(guardBlock);

    auto activeSigName =
        builder.getStringAttr(createActiveSigName(portInfo.name));
    auto activeSignalOp =
        builder.create<sv::BitOp>(builder.getI1Type(), activeSigName,
                                  hw::InnerSymAttr::get(activeSigName));
    argNumToInnerNames.insert(
        {{portInfo.dir, portInfo.argNum}, activeSignalOp.getInnerRef()});

    builder.setInsertionPointToEnd(guardBlock);
    if (portInfo.dir == hw::ModulePort::Direction::Input) {
      auto convCast = builder.create<mlir::UnrealizedConversionCastOp>(
          TypeRange{portInfo.type}, activeSignalOp.getResult());
      auto arg = moduleOp.getArgumentForPort(idx);
      arg.replaceAllUsesWith(convCast.getResult(0));
    } else { // portInfo.dir == hw::ModulePort::Direction::Output
      assert(portInfo.isOutput());
      auto outputOp =
          llvm::cast<hw::OutputOp>(moduleOp.getBodyBlock()->getTerminator());
      auto ssaValue = outputOp->getOperand(portInfo.argNum);
      outputOp.setOperand(portInfo.argNum, builder.createOrFold<sim::NeverOp>(
                                               ssaValue.getType()));

      builder.create<sim::TriggeredOp>(
          ssaValue, activeSignalOp.getResult(), [&]() {
            auto cst1 = builder.createOrFold<hw::ConstantOp>(
                IntegerAttr::get(builder.getI1Type(), 1));
            builder.create<sv::BPAssignOp>(
                builder.getInsertionBlock()->getArgument(0), cst1);
            auto readOp = builder.createOrFold<sv::ReadInOutOp>(
                builder.getInsertionBlock()->getArgument(0));
            auto notActiveOp =
                builder.createOrFold<comb::XorOp>(readOp, cst1, true);
            builder.create<sv::WaitOp>(notActiveOp);
          });
    }
  }

  auto *node = igraph.lookup(moduleOp);
  for (auto instUse : node->uses())
    if (failed(lowerCrossModuleTriggerParent(instUse, triggerPorts,
                                             argNumToInnerNames)))
      return failure();

  return success();
}

static void externalizeResults(
    TriggeredOp triggerdOp, OpBuilder &builder,
    SmallDenseMap<TriggeredOp, SmallVector<Value>> &resultToRegMap) {
  SmallVector<Value> newRegs;
  SmallVector<Value> reads;
  for (auto [res, tieoff] :
       llvm::zip(triggerdOp.getResults(), *triggerdOp.getTieoffs())) {
    auto cst = materializeInitValue(builder, cast<TypedAttr>(tieoff),
                                    triggerdOp.getLoc());
    auto reg = builder.create<sv::RegOp>(triggerdOp.getLoc(), res.getType(),
                                         StringAttr(), hw::InnerSymAttr(), cst);
    auto regRead = builder.createOrFold<sv::ReadInOutOp>(triggerdOp.getLoc(),
                                                         reg.getResult());
    newRegs.emplace_back(reg.getResult());
    reads.emplace_back(regRead);
  }
  triggerdOp.replaceAllUsesWith(reads);
  resultToRegMap.insert(std::pair<TriggeredOp, SmallVector<Value>>{
      triggerdOp, std::move(newRegs)});
}

LogicalResult TriggersToSVPass::buildSVProcessForRoot(OpBuilder &builder,
                                                      Operation *rootOp,
                                                      Location loc) {
  std::optional<sv::EventControl> clockEvent;
  Value clockSig;

  if (auto convCastOp = dyn_cast<mlir::UnrealizedConversionCastOp>(rootOp)) {
    if (!isBitToTriggerCast(convCastOp)) {
      convCastOp->emitError("Unsupported trigger root.");
      return failure();
    }
    clockSig = builder.createOrFold<sv::ReadInOutOp>(rootOp->getLoc(),
                                                     convCastOp->getOperand(0));
    clockEvent = sv::EventControl::AtPosEdge;
  } else if (auto edgeOp = dyn_cast<OnEdgeOp>(rootOp)) {
    auto clockConv = builder.create<mlir::UnrealizedConversionCastOp>(
        loc, TypeRange{builder.getI1Type()}, edgeOp.getClock());
    clockSig = clockConv.getResult(0);
    clockEvent =
        convertEventControl(edgeOp.getResult().getType().getEdgeEvent());
  } else if (!isa<sim::OnInitOp>(rootOp)) {
    rootOp->emitError("Unsupported trigger root.");
    return failure();
  }

  if (clockSig) {
    auto alwaysOp = builder.create<sv::AlwaysOp>(loc, *clockEvent, clockSig);
    builder.setInsertionPointToStart(alwaysOp.getBodyBlock());
  } else {
    auto initOp = builder.create<sv::InitialOp>(loc);
    builder.setInsertionPointToStart(initOp.getBodyBlock());
    if (useDelayedInitProcesses)
      builder.create<sv::DelayOp>(loc, 0);
  }

  if (isa<mlir::UnrealizedConversionCastOp>(rootOp)) {
    auto cst0 = builder.createOrFold<hw::ConstantOp>(
        rootOp->getLoc(), IntegerAttr::get(builder.getI1Type(), 0));
    auto assignOp = builder.create<sv::BPAssignOp>(rootOp->getLoc(),
                                                   rootOp->getOperand(0), cst0);
    builder.setInsertionPoint(assignOp);
  }

  return success();
}

LogicalResult TriggersToSVPass::lowerRootTrigger(
    Value root, SmallVectorImpl<TriggeredOp> &lowerdOps, Block *guardBlock) {
  auto rootDefOp = root.getDefiningOp();
  assert(!!rootDefOp);
  assert(isTriggerType(root.getType()));
  OpBuilder builder(rootDefOp);
  SmallVector<Location> locs;
  SmallDenseMap<TriggeredOp, SmallVector<Value>> resultToRegMap;
  SmallVector<TriggeredOp> procs;

  collectTriggeredOps(root, procs);

  if (procs.empty())
    return success();

  locs.reserve(procs.size() + 1);
  locs.emplace_back(rootDefOp->getLoc());
  for (auto proc : procs)
    locs.emplace_back(proc.getLoc());
  auto fusedLoc = FusedLoc::get(builder.getContext(), locs);

  if (!guardBlock) {
    auto ifDefOp = builder.create<sv::IfDefOp>(
        fusedLoc, "SYNTHESIS", [] {}, [] {});
    guardBlock = ifDefOp.getElseBlock();
  } else {
    builder.setInsertionPoint(guardBlock->getParentOp());
  }

  for (auto proc : procs)
    if (proc.getNumResults() > 0)
      externalizeResults(proc, builder, resultToRegMap);

  struct BuildStackEntry {
    PointerUnion<Value, Operation *> pv;
    OpBuilder::InsertPoint ip;
  };

  builder.setInsertionPointToEnd(guardBlock);

  auto res = buildSVProcessForRoot(builder, rootDefOp, fusedLoc);
  if (failed(res))
    return failure();

  SmallVector<BuildStackEntry> buildStack;
  buildStack.emplace_back(BuildStackEntry{root, builder.saveInsertionPoint()});
  while (!buildStack.empty()) {
    auto popVal = buildStack.pop_back_val();
    builder.restoreInsertionPoint(popVal.ip);

    if (auto trigVal = dyn_cast<Value>(popVal.pv)) {
      auto users = trigVal.getUsers();
      if (users.empty())
        continue;
      if (disallowForkJoin || trigVal.hasOneUse()) {
        for (auto user : users)
          buildStack.emplace_back(BuildStackEntry{user, popVal.ip});
        continue;
      }

      auto numUsers =
          std::distance(trigVal.getUsers().begin(), trigVal.getUsers().end());
      auto forkJoinOp =
          builder.create<sv::ForkJoinOp>(rootDefOp->getLoc(), numUsers);
      for (auto const &[user, region] :
           llvm::zip(users, forkJoinOp.getRegions())) {
        Block *block = builder.createBlock(&region);
        auto newIp = OpBuilder::InsertPoint(block, block->begin());
        buildStack.emplace_back(BuildStackEntry{user, newIp});
      }
      continue;
    }
    auto op = cast<Operation *>(popVal.pv);

    if (auto sequence = dyn_cast<TriggerSequenceOp>(op)) {
      for (auto res : llvm::reverse(sequence.getResults()))
        buildStack.emplace_back(BuildStackEntry{res, popVal.ip});
      continue;
    }

    if (auto gate = dyn_cast<TriggerGateOp>(op)) {
      auto ifOp = builder.create<sv::IfOp>(gate.getLoc(), gate.getEnable());
      auto newIp = OpBuilder::InsertPoint(ifOp.getThenBlock(),
                                          ifOp.getThenBlock()->begin());
      buildStack.emplace_back(BuildStackEntry{gate.getResult(), newIp});
      continue;
    }

    if (auto procedure = dyn_cast<TriggeredOp>(op)) {
      lowerdOps.push_back(procedure);

      auto yield =
          cast<YieldSeqOp>(procedure.getBody().front().getTerminator());
      auto &regs = resultToRegMap[procedure];
      mlir::IRRewriter rewriter(builder);

      OpBuilder::InsertPoint ip = builder.saveInsertionPoint();
      rewriter.inlineBlockBefore(&procedure.getBody().front(), ip.getBlock(),
                                 ip.getPoint(), procedure.getInputs());

      assert(regs.size() == yield.getNumOperands() &&
             "Failed to lookup materialized result registers");
      for (auto [reg, res] : llvm::zip(regs, yield.getOperands()))
        builder.create<sv::PAssignOp>(yield.getLoc(), reg, res);
      yield.erase();

      continue;
    }
    op->emitWarning("Unable to lower trigger user.");
  }

  return success();
}

static void cleanUpTriggerTree(ArrayRef<TriggeredOp> procs) {
  SmallVector<Operation *> cleanupList;
  SmallVector<Operation *> cleanupNextList;
  SmallPtrSet<Operation *, 8> erasedOps;
  for (auto proc : procs) {
    auto trigger = proc.getTrigger();
    cleanupNextList.emplace_back(trigger.getDefiningOp());
    erasedOps.insert(proc);
    proc.erase();
  }

  bool hasChanged = true;
  while (hasChanged && !cleanupNextList.empty()) {
    cleanupList = std::move(cleanupNextList);
    cleanupNextList.clear();
    hasChanged = false;
    for (auto op : cleanupList) {
      if (!op || erasedOps.contains(op))
        continue;
      if (auto seqOp = dyn_cast<TriggerSequenceOp>(op)) {
        if (seqOp.use_empty()) {
          cleanupNextList.push_back(seqOp.getParent().getDefiningOp());
          erasedOps.insert(seqOp);
          hasChanged = true;
          seqOp.erase();
        } else {
          cleanupNextList.push_back(seqOp);
        }
        continue;
      }
      if (auto gate = dyn_cast<TriggerGateOp>(op)) {
        if (gate.getResult().use_empty()) {
          erasedOps.insert(gate);
          hasChanged = true;
          cleanupNextList.push_back(gate.getInput().getDefiningOp());
          gate.erase();
        } else {
          cleanupNextList.push_back(gate);
        }
        continue;
      }
      if (isa<OnEdgeOp, OnInitOp, NeverOp>(op) ||
          isBitToTriggerCast(dyn_cast<mlir::UnrealizedConversionCastOp>(op))) {
        if (op->use_empty()) {
          erasedOps.insert(op);
          op->erase();
        } else {
          cleanupNextList.push_back(op);
        }
        continue;
      }
    }
  }
}

LogicalResult TriggersToSVPass::lowerLocalTriggers(hw::HWModuleOp moduleOp,
                                                   size_t &numRoots) {
  SmallVector<TriggeredOp> cleanupList;
  SmallVector<Value> trueRoots;

  SmallVector<Value> castRoots;

  moduleOp.walk([&](Operation *op) {
    TypeSwitch<Operation *>(op)
        .Case<OnEdgeOp, OnInitOp>([&](auto rootOp) {
          if (!rootOp.getResult().use_empty())
            trueRoots.push_back(rootOp.getResult());
          else
            rootOp->erase();
        })
        .Case<NeverOp>([&](auto neverOp) {
          if (!neverOp.getResult().use_empty())
            collectTriggeredOps(neverOp.getResult(), cleanupList);
          else
            neverOp->erase();
        })
        .Case<mlir::UnrealizedConversionCastOp>([&](auto castOp) {
          if (isBitToTriggerCast(castOp))
            castRoots.push_back(castOp.getResult(0));
        });
  });

  // Tie off unused casts
  for (auto castRoot : castRoots) {
    if (castRoot.use_empty()) {
      ImplicitLocOpBuilder builder(castRoot.getDefiningOp()->getLoc(),
                                   castRoot.getDefiningOp());
      builder.setInsertionPointToStart(moduleOp.getBodyBlock());
      builder.create<sim::TriggeredOp>(castRoot);
    }
  }
  SmallVector<Value> allRoots = std::move(castRoots);

  // Split true roots
  for (auto trueRoot : trueRoots) {
    if (trueRoot.hasOneUse()) {
      allRoots.push_back(trueRoot);
      continue;
    }
    for (auto &use : llvm::make_early_inc_range(trueRoot.getUses())) {
      ImplicitLocOpBuilder builder(trueRoot.getDefiningOp()->getLoc(),
                                   use.getOwner());
      auto clonedRoot = builder.clone(*trueRoot.getDefiningOp());
      use.assign(clonedRoot->getResult(0));
      allRoots.push_back(clonedRoot->getResult(0));
    }
    assert(trueRoot.use_empty());
    trueRoot.getDefiningOp()->erase();
  }

  // Lower roots
  bool hasFailed = false;
  numRoots = allRoots.size();
  for (auto root : allRoots)
    hasFailed |=
        failed(lowerRootTrigger(root, cleanupList, ifdefGuardBlocks[moduleOp]));

  if (hasFailed)
    return failure();

  cleanUpTriggerTree(cleanupList);
  return success();
}

} // anonymous namespace
