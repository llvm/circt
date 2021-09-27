//===- HWMemSimImpl.cpp - HW Memory Implementation Pass -------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This transformation pass converts generated FIRRTL memory modules to
// simulation models.
//
//===----------------------------------------------------------------------===//

#include "PassDetail.h"
#include "circt/Dialect/Comb/CombOps.h"
#include "circt/Dialect/HW/HWAttributes.h"
#include "circt/Dialect/HW/HWOps.h"
#include "circt/Dialect/SV/SVPasses.h"
#include "mlir/IR/ImplicitLocOpBuilder.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/JSON.h"

using namespace circt;
using namespace hw;

//===----------------------------------------------------------------------===//
// HWMemSimImplPass Pass
//===----------------------------------------------------------------------===//

static const char testbenchMemAttrName[] = "firrtl.testbenchMemory";
static const char dutMemoryAttrName[] = "firrtl.dutMemory";
static const char seqMemMetadataAttrName[] = "firrtl.seq_mem_verif_data";
static const char memConfAttrName[] = "firrtl.memConfigFile";

namespace {
struct FirMemory {
  size_t numReadPorts;
  size_t numWritePorts;
  size_t numReadWritePorts;
  size_t dataWidth;
  size_t depth;
  size_t maskGran;
  size_t readLatency;
  size_t writeLatency;
  size_t readUnderWrite;
  WUW writeUnderWrite;
  SmallVector<int32_t> writeClockIDs;
};
} // end anonymous namespace

namespace {
struct HWMemSimImplPass : public sv::HWMemSimImplBase<HWMemSimImplPass> {
  void runOnOperation() override;

private:
  void generateMemory(HWModuleOp op, FirMemory mem);
};
} // end anonymous namespace

static FirMemory analyzeMemOp(HWModuleGeneratedOp op) {
  FirMemory mem;
  mem.depth = op->getAttrOfType<IntegerAttr>("depth").getInt();
  mem.numReadPorts = op->getAttrOfType<IntegerAttr>("numReadPorts").getUInt();
  mem.numWritePorts = op->getAttrOfType<IntegerAttr>("numWritePorts").getUInt();
  mem.numReadWritePorts =
      op->getAttrOfType<IntegerAttr>("numReadWritePorts").getUInt();
  mem.readLatency = op->getAttrOfType<IntegerAttr>("readLatency").getUInt();
  mem.writeLatency = op->getAttrOfType<IntegerAttr>("writeLatency").getUInt();
  mem.dataWidth = op->getAttrOfType<IntegerAttr>("width").getUInt();
  mem.maskGran = op->getAttrOfType<IntegerAttr>("maskGran").getUInt();
  mem.readUnderWrite =
      op->getAttrOfType<IntegerAttr>("readUnderWrite").getUInt();
  mem.writeUnderWrite =
      op->getAttrOfType<WUWAttr>("writeUnderWrite").getValue();
  if (auto clockIDsAttr = op->getAttrOfType<ArrayAttr>("writeClockIDs"))
    for (auto clockID : clockIDsAttr)
      mem.writeClockIDs.push_back(
          clockID.cast<IntegerAttr>().getValue().getZExtValue());
  return mem;
}

static Value addPipelineStages(ImplicitLocOpBuilder &b, size_t stages,
                               Value clock, Value data) {
  if (!stages)
    return data;

  while (stages--) {
    auto reg = b.create<sv::RegOp>(data.getType());

    // pipeline stage
    b.create<sv::AlwaysFFOp>(sv::EventControl::AtPosEdge, clock,
                             [&]() { b.create<sv::PAssignOp>(reg, data); });
    data = b.create<sv::ReadInOutOp>(reg);
  }

  return data;
}

void HWMemSimImplPass::generateMemory(HWModuleOp op, FirMemory mem) {
  ImplicitLocOpBuilder b(UnknownLoc::get(&getContext()), op.getBody());

  // Create a register for the memory.
  auto dataType = b.getIntegerType(mem.dataWidth);
  Value reg = b.create<sv::RegOp>(UnpackedArrayType::get(dataType, mem.depth),
                                  b.getStringAttr("Memory"));

  SmallVector<Value, 4> outputs;

  size_t inArg = 0;
  for (size_t i = 0; i < mem.numReadPorts; ++i) {
    Value clock = op.body().getArgument(inArg++);
    Value en = op.body().getArgument(inArg++);
    Value addr = op.body().getArgument(inArg++);
    // Add pipeline stages
    en = addPipelineStages(b, mem.readLatency, clock, en);
    addr = addPipelineStages(b, mem.readLatency, clock, addr);

    // Read Logic
    Value ren =
        b.create<sv::ReadInOutOp>(b.create<sv::ArrayIndexInOutOp>(reg, addr));
    Value x = b.create<sv::ConstantXOp>(dataType);

    Value rdata = b.create<comb::MuxOp>(en, ren, x);
    outputs.push_back(rdata);
  }

  for (size_t i = 0; i < mem.numReadWritePorts; ++i) {
    auto numStages = std::max(mem.readLatency, mem.writeLatency) - 1;
    Value clock = op.body().getArgument(inArg++);
    Value en = op.body().getArgument(inArg++);
    Value addr = op.body().getArgument(inArg++);
    Value wmode = op.body().getArgument(inArg++);
    Value wmask = op.body().getArgument(inArg++);
    Value wdata = op.body().getArgument(inArg++);

    // Add pipeline stages
    en = addPipelineStages(b, numStages, clock, en);
    addr = addPipelineStages(b, numStages, clock, addr);
    wmode = addPipelineStages(b, numStages, clock, wmode);
    wmask = addPipelineStages(b, numStages, clock, wmask);
    wdata = addPipelineStages(b, numStages, clock, wdata);

    // wire to store read result
    auto rWire = b.create<sv::WireOp>(wdata.getType());
    Value rdata = b.create<sv::ReadInOutOp>(rWire);

    // Read logic.
    Value rcond = b.createOrFold<comb::AndOp>(
        en, b.createOrFold<comb::ICmpOp>(
                comb::ICmpPredicate::eq, wmode,
                b.createOrFold<ConstantOp>(wmode.getType(), 0)));
    Value slot = b.create<sv::ArrayIndexInOutOp>(reg, addr);
    Value x = b.create<sv::ConstantXOp>(dataType);
    b.create<sv::AssignOp>(
        rWire,
        b.create<comb::MuxOp>(rcond, b.create<sv::ReadInOutOp>(slot), x));

    // Write logic.
    b.create<sv::AlwaysFFOp>(sv::EventControl::AtPosEdge, clock, [&]() {
      auto wcond = b.createOrFold<comb::AndOp>(
          en, b.createOrFold<comb::AndOp>(wmask, wmode));
      b.create<sv::IfOp>(wcond,
                         [&]() { b.create<sv::PAssignOp>(slot, wdata); });
    });
    outputs.push_back(rdata);
  }

  DenseMap<unsigned, Operation *> writeProcesses;
  for (size_t i = 0; i < mem.numWritePorts; ++i) {
    auto numStages = mem.writeLatency - 1;
    Value clock = op.body().getArgument(inArg++);
    Value en = op.body().getArgument(inArg++);
    Value addr = op.body().getArgument(inArg++);
    Value wmask = op.body().getArgument(inArg++);
    Value wdata = op.body().getArgument(inArg++);
    // Add pipeline stages
    en = addPipelineStages(b, numStages, clock, en);
    addr = addPipelineStages(b, numStages, clock, addr);
    wmask = addPipelineStages(b, numStages, clock, wmask);
    wdata = addPipelineStages(b, numStages, clock, wdata);

    // Build write port logic.
    auto writeLogic = [&] {
      auto wcond = b.createOrFold<comb::AndOp>(en, wmask);
      b.create<sv::IfOp>(wcond, [&]() {
        auto slot = b.create<sv::ArrayIndexInOutOp>(reg, addr);
        b.create<sv::PAssignOp>(slot, wdata);
      });
    };

    // Build a new always block with write port logic.
    auto alwaysBlock = [&] {
      return b.create<sv::AlwaysFFOp>(sv::EventControl::AtPosEdge, clock,
                                      [&]() { writeLogic(); });
    };

    switch (mem.writeUnderWrite) {
    // Undefined write order:  lower each write port into a separate always
    // block.
    case WUW::Undefined:
      alwaysBlock();
      break;
    // Port-ordered write order:  lower each write port into an always block
    // based on its clock ID.
    case WUW::PortOrder:
      if (auto *existingAlwaysBlock =
              writeProcesses.lookup(mem.writeClockIDs[i])) {
        OpBuilder::InsertionGuard guard(b);
        b.setInsertionPointToEnd(
            cast<sv::AlwaysFFOp>(existingAlwaysBlock).getBodyBlock());
        writeLogic();
      } else {
        writeProcesses[i] = alwaysBlock();
      }
    }
  }

  auto outputOp = op.getBodyBlock()->getTerminator();
  outputOp->setOperands(outputs);
}

// Get all the hierarchichal names for a HWModuleOp.
void static getHierarchichalNames(HWModuleOp op,
                                  SmallVector<std::string> &names,
                                  const std::string path,
                                  mlir::SymbolUserMap &symbolUsers) {

  if (!op)
    return;
  bool parentFound = false;
  for (auto u : symbolUsers.getUsers(op)) {
    if (auto inst = dyn_cast<InstanceOp>(u)) {
      auto opPath = inst.instanceName().str() + "." + path;
      getHierarchichalNames(inst->getParentOfType<HWModuleOp>(), names, opPath,
                            symbolUsers);
      parentFound = true;
    }
  }
  if (!parentFound)
    names.push_back(op.getName().str() + "." + path);
}

void HWMemSimImplPass::runOnOperation() {
  auto topModule = getOperation();
  mlir::SymbolTableCollection symbolTable;
  mlir::SymbolUserMap symbolUsers(symbolTable, topModule);

  SmallVector<HWModuleGeneratedOp> toErase;
  bool anythingChanged = false;
  DenseMap<Operation *, SmallVector<Operation *, 8>> genToMemMap;
  SmallDenseMap<Operation *, bool> testBenchOps;
  llvm::SetVector<Operation *> dutOps;
  llvm::SetVector<Operation *> tbOps;
  Attribute tbMetadataFile, dutMetadataFile, memConfAttr;
  for (auto op : topModule.getBody()->getOps<HWModuleGeneratedOp>()) {
    auto hwModule = cast<HWModuleGeneratedOp>(op);
    genToMemMap[op] = {};
    for (auto u : symbolUsers.getUsers(hwModule))
      if (auto inst = dyn_cast<InstanceOp>(u)) {
        genToMemMap[op].push_back(inst);
        if (inst->hasAttr(testbenchMemAttrName)) {
          dutMetadataFile = inst->getAttr(testbenchMemAttrName);
          dutOps.insert(hwModule);
        } else if (inst->hasAttr(dutMemoryAttrName)) {
          tbMetadataFile = inst->getAttr(dutMemoryAttrName);
          tbOps.insert(hwModule);
        }
        if (inst->hasAttr(memConfAttrName))
          memConfAttr = inst->getAttr(memConfAttrName);
      }
  }

  auto genJson = [&](llvm::json::OStream &J, HWModuleGeneratedOp hwModule,
                     std::string &seqMemConfStr, unsigned index,
                     unsigned confIndex) {
    auto mem = analyzeMemOp(hwModule);
    std::string portStr;
    if (mem.numWritePorts)
      portStr += "mwrite";
    if (mem.numReadPorts) {
      if (!portStr.empty())
        portStr += ",";
      portStr += "read";
    }
    if (mem.numReadWritePorts)
      portStr = "mrw";

    seqMemConfStr += "name {{" + std::to_string(confIndex) + "}} depth " +
                     std::to_string(mem.depth) + " width " +
                     std::to_string(mem.depth) + " ports " + portStr +
                     " mask_gran " + std::to_string(mem.maskGran);
    seqMemConfStr += " \n";
    J.attribute("module_name", "{{" + std::to_string(index) + "}}");
    J.attribute("depth", (int64_t)mem.depth);
    J.attribute("width", (int64_t)mem.dataWidth);
    J.attribute("mask", "true");
    J.attribute("read", mem.numReadPorts ? "true" : "false");
    J.attribute("write", mem.numWritePorts ? "true" : "false");
    J.attribute("readwrite", mem.numReadWritePorts ? "true" : "false");
    J.attribute("mask_granularity", (int64_t)mem.maskGran);
    J.attributeArray("extra_ports", [&] {});
    for (auto userOp : genToMemMap[hwModule]) {
      InstanceOp instOp = dyn_cast<InstanceOp>(userOp);
      auto verifData =
          instOp->getAttrOfType<DictionaryAttr>(seqMemMetadataAttrName);
      SmallVector<std::string> hierNames;
      getHierarchichalNames(instOp->getParentOfType<HWModuleOp>(), hierNames,
                            instOp.instanceName().str(), symbolUsers);
      J.attributeArray("hierarchy", [&] {
        for (auto h : hierNames)
          J.value(h);
      });
      J.attributeObject("verification_only_data", [&] {
        for (auto h : hierNames) {
          J.attributeObject(h, [&] {
            for (auto a : verifData) {
              auto id = a.first.strref().str();
              auto v = a.second;
              if (auto intV = v.dyn_cast<IntegerAttr>())
                J.attribute(id, (int64_t)intV.getValue().getZExtValue());
              else if (auto strV = v.dyn_cast<StringAttr>())
                J.attribute(id, strV.getValue().str());
              else if (auto arrV = v.dyn_cast<ArrayAttr>()) {
                std::string indices;
                J.attributeArray(id, [&] {
                  for (auto arrI : llvm::enumerate(arrV)) {
                    auto i = arrI.value();
                    if (auto intV = i.dyn_cast<IntegerAttr>())
                      J.value(std::to_string(intV.getValue().getZExtValue()));
                    else if (auto strV = i.dyn_cast<StringAttr>())
                      J.value(strV.getValue().str());
                  }
                });
              }
            }
          });
        }
      });
    }
  };
  auto builder = OpBuilder::atBlockEnd(topModule.getBody());
  SmallVector<Attribute> confSymbolsVerbatim;
  std::string seqMemConfStr;
  auto seqMemjson = [&](llvm::SetVector<Operation *> &opsSet,
                        Attribute fileAttr, unsigned confIndex) {
    if (opsSet.empty())
      return;
    std::string seqmemJsonBuffer;
    llvm::raw_string_ostream os(seqmemJsonBuffer);
    llvm::json::OStream J(os);
    SmallVector<Attribute> jsonSymbolsVerbatim;
    J.array([&] {
      for (auto opIndex : llvm::enumerate(opsSet)) {
        auto hwModule = cast<HWModuleGeneratedOp>(opIndex.value());
        J.object([&] {
          genJson(J, hwModule, seqMemConfStr, opIndex.index(),
                  confIndex + opIndex.index());
        });
        auto symRef = SymbolRefAttr::get(hwModule);
        confSymbolsVerbatim.push_back(symRef);
        jsonSymbolsVerbatim.push_back(symRef);
      }
    });
    auto v = builder.create<sv::VerbatimOp>(
        builder.getUnknownLoc(), seqmemJsonBuffer, ValueRange(),
        builder.getArrayAttr({jsonSymbolsVerbatim}));
    v->setAttr("output_file", fileAttr);
  };
  seqMemjson(tbOps, tbMetadataFile, 0);
  seqMemjson(dutOps, dutMetadataFile, tbOps.size());
  if (memConfAttr) {
    auto configV = builder.create<sv::VerbatimOp>(
        builder.getUnknownLoc(), seqMemConfStr, ValueRange(),
        builder.getArrayAttr({confSymbolsVerbatim}));
    configV->setAttr("output_file", memConfAttr);
  }
  for (auto op : llvm::make_early_inc_range(
           topModule.getBody()->getOps<HWModuleGeneratedOp>())) {
    auto oldModule = cast<HWModuleGeneratedOp>(op);
    auto gen = oldModule.generatorKind();
    auto genOp = cast<HWGeneratorSchemaOp>(
        SymbolTable::lookupSymbolIn(getOperation(), gen));

    if (genOp.descriptor() == "FIRRTL_Memory") {
      auto mem = analyzeMemOp(oldModule);

      OpBuilder builder(oldModule);
      auto nameAttr = builder.getStringAttr(oldModule.getName());
      auto newModule = builder.create<HWModuleOp>(oldModule.getLoc(), nameAttr,
                                                  oldModule.getPorts());
      generateMemory(newModule, mem);
      oldModule.erase();
      anythingChanged = true;
    }
  }

  if (!anythingChanged)
    markAllAnalysesPreserved();
}

std::unique_ptr<Pass> circt::sv::createHWMemSimImplPass() {
  return std::make_unique<HWMemSimImplPass>();
}
