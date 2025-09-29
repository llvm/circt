//===- LowerMemory.cpp - Lower Memories -------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//===----------------------------------------------------------------------===//
//
// This file defines the LowerMemories pass.
//
//===----------------------------------------------------------------------===//

#include "circt/Analysis/FIRRTLInstanceInfo.h"
#include "circt/Dialect/FIRRTL/AnnotationDetails.h"
#include "circt/Dialect/FIRRTL/FIRRTLOps.h"
#include "circt/Dialect/FIRRTL/FIRRTLUtils.h"
#include "circt/Dialect/FIRRTL/Namespace.h"
#include "circt/Dialect/FIRRTL/Passes.h"
#include "circt/Dialect/HW/HWOps.h"
#include "circt/Dialect/HW/InnerSymbolNamespace.h"
#include "circt/Dialect/Seq/SeqAttributes.h"
#include "mlir/IR/Dominance.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/DepthFirstIterator.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/STLFunctionalExtras.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Parallel.h"
#include <optional>
#include <set>

namespace circt {
namespace firrtl {
#define GEN_PASS_DEF_LOWERMEMORY
#include "circt/Dialect/FIRRTL/Passes.h.inc"
} // namespace firrtl
} // namespace circt

using namespace circt;
using namespace firrtl;

// Extract all the relevant attributes from the MemOp and return the FirMemory.
FirMemory getSummary(MemOp op) {
  size_t numReadPorts = 0;
  size_t numWritePorts = 0;
  size_t numReadWritePorts = 0;
  llvm::SmallDenseMap<Value, unsigned> clockToLeader;
  SmallVector<int32_t> writeClockIDs;

  for (size_t i = 0, e = op.getNumResults(); i != e; ++i) {
    auto portKind = op.getPortKind(i);
    if (portKind == MemOp::PortKind::Read)
      ++numReadPorts;
    else if (portKind == MemOp::PortKind::Write) {
      for (auto *a : op.getResult(i).getUsers()) {
        auto subfield = dyn_cast<SubfieldOp>(a);
        if (!subfield || subfield.getFieldIndex() != 2)
          continue;
        auto clockPort = a->getResult(0);
        for (auto *b : clockPort.getUsers()) {
          if (auto connect = dyn_cast<FConnectLike>(b)) {
            if (connect.getDest() == clockPort) {
              auto result =
                  clockToLeader.insert({connect.getSrc(), numWritePorts});
              if (result.second) {
                writeClockIDs.push_back(numWritePorts);
              } else {
                writeClockIDs.push_back(result.first->second);
              }
            }
          }
        }
        break;
      }
      ++numWritePorts;
    } else
      ++numReadWritePorts;
  }

  auto width = op.getDataType().getBitWidthOrSentinel();
  if (width <= 0) {
    op.emitError("'firrtl.mem' should have simple type and known width");
    width = 0;
  }
  return {numReadPorts,
          numWritePorts,
          numReadWritePorts,
          (size_t)width,
          op.getDepth(),
          op.getReadLatency(),
          op.getWriteLatency(),
          op.getMaskBits(),
          *seq::symbolizeRUW(unsigned(op.getRuw())),
          seq::WUW::PortOrder,
          writeClockIDs,
          op.getNameAttr(),
          op.getMaskBits() > 1,
          op.getInitAttr(),
          op.getPrefixAttr(),
          op.getLoc()};
}

namespace {
struct LowerMemoryPass
    : public circt::firrtl::impl::LowerMemoryBase<LowerMemoryPass> {

  /// Get the cached namespace for a module.
  hw::InnerSymbolNamespace &getModuleNamespace(FModuleLike moduleOp) {
    return moduleNamespaces.try_emplace(moduleOp, moduleOp).first->second;
  }

  SmallVector<PortInfo> getMemoryModulePorts(const FirMemory &mem);
  FMemModuleOp emitMemoryModule(MemOp op, const FirMemory &summary,
                                const SmallVectorImpl<PortInfo> &ports);
  FMemModuleOp getOrCreateMemModule(MemOp op, const FirMemory &summary,
                                    const SmallVectorImpl<PortInfo> &ports,
                                    bool shouldDedup);
  FModuleOp createWrapperModule(MemOp op, const FirMemory &summary,
                                bool shouldDedup);
  InstanceOp emitMemoryInstance(MemOp op, FModuleOp moduleOp,
                                const FirMemory &summary);
  void lowerMemory(MemOp mem, const FirMemory &summary, bool shouldDedup);
  LogicalResult runOnModule(FModuleOp moduleOp, bool shouldDedup);
  void runOnOperation() override;

  /// Cached module namespaces.
  DenseMap<Operation *, hw::InnerSymbolNamespace> moduleNamespaces;
  CircuitNamespace circuitNamespace;
  SymbolTable *symbolTable;

  /// The set of all memories seen so far.  This is used to "deduplicate"
  /// memories by emitting modules one module for equivalent memories.
  std::map<FirMemory, FMemModuleOp> memories;

  /// A sequence of operations that should be erased later.
  SetVector<Operation *> operationsToErase;
};
} // end anonymous namespace

SmallVector<PortInfo>
LowerMemoryPass::getMemoryModulePorts(const FirMemory &mem) {
  auto *context = &getContext();

  // We don't need a single bit mask, it can be combined with enable. Create
  // an unmasked memory if maskBits = 1.
  FIRRTLType u1Type = UIntType::get(context, 1);
  FIRRTLType dataType = UIntType::get(context, mem.dataWidth);
  FIRRTLType maskType = UIntType::get(context, mem.maskBits);
  FIRRTLType addrType =
      UIntType::get(context, std::max(1U, llvm::Log2_64_Ceil(mem.depth)));
  FIRRTLType clockType = ClockType::get(context);
  Location loc = UnknownLoc::get(context);
  AnnotationSet annotations = AnnotationSet(context);

  SmallVector<PortInfo> ports;
  auto addPort = [&](const Twine &name, FIRRTLType type, Direction direction) {
    auto nameAttr = StringAttr::get(context, name);
    ports.push_back(
        {nameAttr, type, direction, hw::InnerSymAttr{}, loc, annotations, {}});
  };

  auto makePortCommon = [&](StringRef prefix, size_t idx, FIRRTLType addrType) {
    addPort(prefix + Twine(idx) + "_addr", addrType, Direction::In);
    addPort(prefix + Twine(idx) + "_en", u1Type, Direction::In);
    addPort(prefix + Twine(idx) + "_clk", clockType, Direction::In);
  };

  for (size_t i = 0, e = mem.numReadPorts; i != e; ++i) {
    makePortCommon("R", i, addrType);
    addPort("R" + Twine(i) + "_data", dataType, Direction::Out);
  }
  for (size_t i = 0, e = mem.numReadWritePorts; i != e; ++i) {
    makePortCommon("RW", i, addrType);
    addPort("RW" + Twine(i) + "_wmode", u1Type, Direction::In);
    addPort("RW" + Twine(i) + "_wdata", dataType, Direction::In);
    addPort("RW" + Twine(i) + "_rdata", dataType, Direction::Out);
    // Ignore mask port, if maskBits =1
    if (mem.isMasked)
      addPort("RW" + Twine(i) + "_wmask", maskType, Direction::In);
  }

  for (size_t i = 0, e = mem.numWritePorts; i != e; ++i) {
    makePortCommon("W", i, addrType);
    addPort("W" + Twine(i) + "_data", dataType, Direction::In);
    // Ignore mask port, if maskBits =1
    if (mem.isMasked)
      addPort("W" + Twine(i) + "_mask", maskType, Direction::In);
  }

  return ports;
}

static constexpr llvm::StringLiteral
    kExtMemoryEffectNLAAttrName("circt.layerSink.externalMemoryEffect");

FMemModuleOp
LowerMemoryPass::emitMemoryModule(MemOp op, const FirMemory &mem,
                                  const SmallVectorImpl<PortInfo> &ports) {
  // Get a non-colliding name for the memory module, and update the summary.
  StringRef prefix = "";
  if (mem.prefix)
    prefix = mem.prefix.getValue();
  auto newName =
      circuitNamespace.newName(prefix + mem.modName.getValue(), "ext");
  auto moduleName = StringAttr::get(&getContext(), newName);

  // Insert the memory module just above the current module.
  OpBuilder b(op->getParentOfType<FModuleOp>());
  ++numCreatedMemModules;
  auto moduleOp = FMemModuleOp::create(
      b, mem.loc, moduleName, ports, mem.numReadPorts, mem.numWritePorts,
      mem.numReadWritePorts, mem.dataWidth, mem.maskBits, mem.readLatency,
      mem.writeLatency, mem.depth,
      *symbolizeRUWBehavior(static_cast<uint32_t>(mem.readUnderWrite)));
  // Mark the generated external memory as effectful so downstream passes do not
  // treat it as pure and drop the wrapper wiring around it.  We create a
  // temporary HierPathOp, which LayerSink will consume and erase.
  auto circuit = op->getParentOfType<CircuitOp>();
  OpBuilder nlaBuilder(circuit);
  nlaBuilder.setInsertionPointToEnd(circuit.getBodyBlock());
  auto pathAttr = nlaBuilder.getArrayAttr(SmallVector<Attribute, 1>{
      FlatSymbolRefAttr::get(moduleOp.getModuleNameAttr())});
  auto nlaName = circuitNamespace.newName(
      (moduleOp.getModuleNameAttr().getValue() + "_effect_nla").str());
  auto nla = nlaBuilder.create<hw::HierPathOp>(
      mem.loc, StringAttr::get(&getContext(), nlaName), pathAttr);
  nla->setAttr(kExtMemoryEffectNLAAttrName,
               FlatSymbolRefAttr::get(moduleOp.getModuleNameAttr()));
  SymbolTable::setSymbolVisibility(nla, SymbolTable::Visibility::Private);
  SymbolTable::setSymbolVisibility(moduleOp, SymbolTable::Visibility::Private);
  return moduleOp;
}

FMemModuleOp
LowerMemoryPass::getOrCreateMemModule(MemOp op, const FirMemory &summary,
                                      const SmallVectorImpl<PortInfo> &ports,
                                      bool shouldDedup) {
  // Try to find a matching memory blackbox that we already created.  If
  // shouldDedup is true, we will just generate a new memory module.
  if (shouldDedup) {
    auto it = memories.find(summary);
    if (it != memories.end())
      return it->second;
  }

  // Create a new module for this memory. This can update the name recorded in
  // the memory's summary.
  auto moduleOp = emitMemoryModule(op, summary, ports);

  // Record the memory module.  We don't want to use this module for other
  // memories, then we don't add it to the table.
  if (shouldDedup)
    memories[summary] = moduleOp;

  return moduleOp;
}

void LowerMemoryPass::lowerMemory(MemOp mem, const FirMemory &summary,
                                  bool shouldDedup) {
  auto *context = &getContext();
  auto ports = getMemoryModulePorts(summary);

  // Get a non-colliding name for the memory module, and update the summary.
  StringRef prefix = "";
  if (summary.prefix)
    prefix = summary.prefix.getValue();
  auto newName = circuitNamespace.newName(prefix + mem.getName());

  auto wrapperName = StringAttr::get(&getContext(), newName);

  // Create the wrapper module, inserting it just before the current module.
  OpBuilder b(mem->getParentOfType<FModuleOp>());
  auto wrapper = FModuleOp::create(
      b, mem->getLoc(), wrapperName,
      ConventionAttr::get(context, Convention::Internal), ports);
  SymbolTable::setSymbolVisibility(wrapper, SymbolTable::Visibility::Private);

  // Create an instance of the external memory module. The instance has the
  // same name as the target module.
  auto memModule = getOrCreateMemModule(mem, summary, ports, shouldDedup);
  b.setInsertionPointToStart(wrapper.getBodyBlock());
  auto memInst = InstanceOp::create(
      b, mem->getLoc(), memModule, (mem.getName() + "_ext").str(),
      mem.getNameKind(), mem.getAnnotations().getValue());

  // Wire all the ports together.
  for (auto [dst, src] : llvm::zip(wrapper.getBodyBlock()->getArguments(),
                                   memInst.getResults())) {
    if (wrapper.getPortDirection(dst.getArgNumber()) == Direction::Out)
      MatchingConnectOp::create(b, mem->getLoc(), dst, src);
    else
      MatchingConnectOp::create(b, mem->getLoc(), src, dst);
  }

  // Create an instance of the wrapper memory module, which will replace the
  // original mem op.
  auto inst = emitMemoryInstance(mem, wrapper, summary);

  // We fixup the annotations here. We will be copying all annotations on to the
  // module op, so we have to fix up the NLA to have the module as the leaf
  // element.

  auto leafSym = memModule.getModuleNameAttr();
  auto leafAttr = FlatSymbolRefAttr::get(wrapper.getModuleNameAttr());

  // NLAs that we have already processed.
  llvm::SmallDenseMap<StringAttr, StringAttr> processedNLAs;
  auto nonlocalAttr = StringAttr::get(context, "circt.nonlocal");
  bool nlaUpdated = false;
  SmallVector<Annotation> newMemModAnnos;
  OpBuilder nlaBuilder(context);

  AnnotationSet::removeAnnotations(memInst, [&](Annotation anno) -> bool {
    // We're only looking for non-local annotations.
    auto nlaSym = anno.getMember<FlatSymbolRefAttr>(nonlocalAttr);
    if (!nlaSym)
      return false;
    // If we have already seen this NLA, don't re-process it.
    auto newNLAIter = processedNLAs.find(nlaSym.getAttr());
    StringAttr newNLAName;
    if (newNLAIter == processedNLAs.end()) {

      // Update the NLA path to have the additional wrapper module.
      auto nla =
          dyn_cast<hw::HierPathOp>(symbolTable->lookup(nlaSym.getAttr()));
      auto namepath = nla.getNamepath().getValue();
      SmallVector<Attribute> newNamepath(namepath.begin(), namepath.end());
      if (!nla.isComponent())
        newNamepath.back() =
            getInnerRefTo(inst, [&](auto mod) -> hw::InnerSymbolNamespace & {
              return getModuleNamespace(mod);
            });
      newNamepath.push_back(leafAttr);

      nlaBuilder.setInsertionPointAfter(nla);
      auto newNLA = cast<hw::HierPathOp>(nlaBuilder.clone(*nla));
      newNLA.setSymNameAttr(StringAttr::get(
          context, circuitNamespace.newName(nla.getNameAttr().getValue())));
      newNLA.setNamepathAttr(ArrayAttr::get(context, newNamepath));
      newNLAName = newNLA.getNameAttr();
      processedNLAs[nlaSym.getAttr()] = newNLAName;
    } else
      newNLAName = newNLAIter->getSecond();
    anno.setMember("circt.nonlocal", FlatSymbolRefAttr::get(newNLAName));
    nlaUpdated = true;
    newMemModAnnos.push_back(anno);
    return true;
  });
  if (nlaUpdated) {
    memInst.setInnerSymAttr(hw::InnerSymAttr::get(leafSym));
    AnnotationSet newAnnos(memInst);
    newAnnos.addAnnotations(newMemModAnnos);
    newAnnos.applyToOperation(memInst);
  }
  operationsToErase.insert(mem);
  ++numLoweredMems;
}

static SmallVector<SubfieldOp> getAllFieldAccesses(Value structValue,
                                                   StringRef field) {
  SmallVector<SubfieldOp> accesses;
  for (auto *op : structValue.getUsers()) {
    assert(isa<SubfieldOp>(op));
    auto fieldAccess = cast<SubfieldOp>(op);
    auto elemIndex =
        fieldAccess.getInput().getType().base().getElementIndex(field);
    if (elemIndex && *elemIndex == fieldAccess.getFieldIndex())
      accesses.push_back(fieldAccess);
  }
  return accesses;
}

InstanceOp LowerMemoryPass::emitMemoryInstance(MemOp op, FModuleOp module,
                                               const FirMemory &summary) {
  OpBuilder builder(op);
  auto *context = &getContext();
  auto memName = op.getName();
  if (memName.empty())
    memName = "mem";

  // Process each port in turn.
  SmallVector<Type, 8> portTypes;
  SmallVector<Direction> portDirections;
  SmallVector<Attribute> portNames;
  SmallVector<Attribute> domainInfo;
  DenseMap<Operation *, size_t> returnHolder;
  mlir::DominanceInfo domInfo(op->getParentOfType<FModuleOp>());

  // The result values of the memory are not necessarily in the same order as
  // the memory module that we're lowering to.  We need to lower the read
  // ports before the read/write ports, before the write ports.
  for (unsigned memportKindIdx = 0; memportKindIdx != 3; ++memportKindIdx) {
    MemOp::PortKind memportKind = MemOp::PortKind::Read;
    auto *portLabel = "R";
    switch (memportKindIdx) {
    default:
      break;
    case 1:
      memportKind = MemOp::PortKind::ReadWrite;
      portLabel = "RW";
      break;
    case 2:
      memportKind = MemOp::PortKind::Write;
      portLabel = "W";
      break;
    }

    // This is set to the count of the kind of memport we're emitting, for
    // label names.
    unsigned portNumber = 0;

    // Get an unsigned type with the specified width.
    auto getType = [&](size_t width) { return UIntType::get(context, width); };
    auto ui1Type = getType(1);
    auto addressType = getType(std::max(1U, llvm::Log2_64_Ceil(summary.depth)));
    auto dataType = UIntType::get(context, summary.dataWidth);
    auto clockType = ClockType::get(context);

    // Memories return multiple structs, one for each port, which means we
    // have two layers of type to split apart.
    for (size_t i = 0, e = op.getNumResults(); i != e; ++i) {
      // Process all of one kind before the next.
      if (memportKind != op.getPortKind(i))
        continue;

      auto addPort = [&](Direction direction, StringRef field, Type portType) {
        // Map subfields of the memory port to module ports.
        auto accesses = getAllFieldAccesses(op.getResult(i), field);
        for (auto a : accesses)
          returnHolder[a] = portTypes.size();
        // Record the new port information.
        portTypes.push_back(portType);
        portDirections.push_back(direction);
        portNames.push_back(
            builder.getStringAttr(portLabel + Twine(portNumber) + "_" + field));
        domainInfo.push_back(builder.getArrayAttr({}));
      };

      auto getDriver = [&](StringRef field) -> Operation * {
        auto accesses = getAllFieldAccesses(op.getResult(i), field);
        for (auto a : accesses) {
          for (auto *user : a->getUsers()) {
            // If this is a connect driving a value to the field, return it.
            if (auto connect = dyn_cast<FConnectLike>(user);
                connect && connect.getDest() == a)
              return connect;
          }
        }
        return nullptr;
      };

      // Find the value connected to the enable and 'and' it with the mask,
      // and then remove the mask entirely. This is used to remove the mask when
      // it is 1 bit.
      auto removeMask = [&](StringRef enable, StringRef mask) {
        // Get the connect which drives a value to the mask element.
        auto *maskConnect = getDriver(mask);
        if (!maskConnect)
          return;
        // Get the connect which drives a value to the en element
        auto *enConnect = getDriver(enable);
        if (!enConnect)
          return;
        // Find the proper place to create the And operation.  The mask and en
        // signals must both dominate the new operation.
        OpBuilder b(maskConnect);
        if (domInfo.dominates(maskConnect, enConnect))
          b.setInsertionPoint(enConnect);
        // 'and' the enable and mask signals together and use it as the enable.
        auto andOp =
            AndPrimOp::create(b, op->getLoc(), maskConnect->getOperand(1),
                              enConnect->getOperand(1));
        enConnect->setOperand(1, andOp);
        enConnect->moveAfter(andOp);
        // Erase the old mask connect.
        auto *maskField = maskConnect->getOperand(0).getDefiningOp();
        operationsToErase.insert(maskConnect);
        operationsToErase.insert(maskField);
      };

      if (memportKind == MemOp::PortKind::Read) {
        addPort(Direction::In, "addr", addressType);
        addPort(Direction::In, "en", ui1Type);
        addPort(Direction::In, "clk", clockType);
        addPort(Direction::Out, "data", dataType);
      } else if (memportKind == MemOp::PortKind::ReadWrite) {
        addPort(Direction::In, "addr", addressType);
        addPort(Direction::In, "en", ui1Type);
        addPort(Direction::In, "clk", clockType);
        addPort(Direction::In, "wmode", ui1Type);
        addPort(Direction::In, "wdata", dataType);
        addPort(Direction::Out, "rdata", dataType);
        // Ignore mask port, if maskBits =1
        if (summary.isMasked)
          addPort(Direction::In, "wmask", getType(summary.maskBits));
        else
          removeMask("wmode", "wmask");
      } else {
        addPort(Direction::In, "addr", addressType);
        addPort(Direction::In, "en", ui1Type);
        addPort(Direction::In, "clk", clockType);
        addPort(Direction::In, "data", dataType);
        // Ignore mask port, if maskBits == 1
        if (summary.isMasked)
          addPort(Direction::In, "mask", getType(summary.maskBits));
        else
          removeMask("en", "mask");
      }

      ++portNumber;
    }
  }

  // Create the instance to replace the memop. The instance name matches the
  // name of the original memory module before deduplication.
  // TODO: how do we lower port annotations?
  auto inst = InstanceOp::create(
      builder, op.getLoc(), portTypes, module.getNameAttr(),
      summary.getFirMemoryName(), op.getNameKind(), portDirections, portNames,
      domainInfo,
      /*annotations=*/ArrayRef<Attribute>(),
      /*portAnnotations=*/ArrayRef<Attribute>(),
      /*layers=*/ArrayRef<Attribute>(), /*lowerToBind=*/false,
      /*doNotPrint=*/false, op.getInnerSymAttr());

  // Update all users of the result of read ports
  for (auto [subfield, result] : returnHolder) {
    subfield->getResult(0).replaceAllUsesWith(inst.getResult(result));
    operationsToErase.insert(subfield);
  }

  return inst;
}

LogicalResult LowerMemoryPass::runOnModule(FModuleOp moduleOp,
                                           bool shouldDedup) {
  assert(operationsToErase.empty() && "operationsToErase must be empty");

  auto result = moduleOp.walk([&](MemOp op) {
    // Check that the memory has been properly lowered already.
    if (!type_isa<UIntType>(op.getDataType())) {
      op->emitError("memories should be flattened before running LowerMemory");
      return WalkResult::interrupt();
    }

    auto summary = getSummary(op);
    if (summary.isSeqMem())
      lowerMemory(op, summary, shouldDedup);

    return WalkResult::advance();
  });

  if (result.wasInterrupted())
    return failure();

  for (Operation *op : operationsToErase)
    op->erase();

  operationsToErase.clear();

  return success();
}

void LowerMemoryPass::runOnOperation() {
  auto circuit = getOperation();
  auto &instanceInfo = getAnalysis<InstanceInfo>();
  symbolTable = &getAnalysis<SymbolTable>();
  circuitNamespace.add(circuit);

  // We iterate the circuit from top-to-bottom.  This ensures that we get
  // consistent memory names.  (Memory modules will be inserted before the
  // module we are processing to prevent these being unnecessarily visited.)
  // Deduplication of memories is allowed if the module is under the "effective"
  // design-under-test (DUT).
  for (auto moduleOp : circuit.getBodyBlock()->getOps<FModuleOp>()) {
    auto shouldDedup = instanceInfo.anyInstanceInEffectiveDesign(moduleOp);
    if (failed(runOnModule(moduleOp, shouldDedup)))
      return signalPassFailure();
  }

  circuitNamespace.clear();
  symbolTable = nullptr;
  memories.clear();
}
