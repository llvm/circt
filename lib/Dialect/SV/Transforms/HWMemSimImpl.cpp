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
#include "circt/Dialect/HW/HWOps.h"
#include "circt/Dialect/SV/SVPasses.h"
#include "mlir/IR/ImplicitLocOpBuilder.h"
#include "llvm/ADT/TypeSwitch.h"

using namespace circt;
using namespace hw;

//===----------------------------------------------------------------------===//
// HWMemSimImplPass Pass
//===----------------------------------------------------------------------===//

namespace {
struct FirMemory {
  size_t numReadPorts;
  size_t numWritePorts;
  size_t numReadWritePorts;
  size_t dataWidth;
  size_t depth;
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
    Value addr = op.body().getArgument(inArg++);
    Value en = op.body().getArgument(inArg++);
    Value clock = op.body().getArgument(inArg++);
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
    Value addr = op.body().getArgument(inArg++);
    Value en = op.body().getArgument(inArg++);
    Value clock = op.body().getArgument(inArg++);
    Value wmode = op.body().getArgument(inArg++);
    Value wdata = op.body().getArgument(inArg++);
    Value wmask = op.body().getArgument(inArg++);

    // Add pipeline stages
    addr = addPipelineStages(b, numStages, clock, addr);
    en = addPipelineStages(b, numStages, clock, en);
    wmode = addPipelineStages(b, numStages, clock, wmode);
    wdata = addPipelineStages(b, numStages, clock, wdata);
    wmask = addPipelineStages(b, numStages, clock, wmask);

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
    Value addr = op.body().getArgument(inArg++);
    Value en = op.body().getArgument(inArg++);
    Value clock = op.body().getArgument(inArg++);
    Value wdata = op.body().getArgument(inArg++);
    Value wmask = op.body().getArgument(inArg++);
    // Add pipeline stages
    addr = addPipelineStages(b, numStages, clock, addr);
    en = addPipelineStages(b, numStages, clock, en);
    wdata = addPipelineStages(b, numStages, clock, wdata);
    wmask = addPipelineStages(b, numStages, clock, wmask);

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

void HWMemSimImplPass::runOnOperation() {
  auto topModule = getOperation().getBody();

  SmallVector<HWModuleGeneratedOp> toErase;
  bool anythingChanged = false;

  for (auto op :
       llvm::make_early_inc_range(topModule->getOps<HWModuleGeneratedOp>())) {
    auto oldModule = cast<HWModuleGeneratedOp>(op);
    auto gen = oldModule.generatorKind();
    auto genOp = cast<HWGeneratorSchemaOp>(
        SymbolTable::lookupSymbolIn(getOperation(), gen));

    if (genOp.descriptor() == "FIRRTL_Memory") {
      auto mem = analyzeMemOp(oldModule);
        OpBuilder builder(oldModule);
        auto nameAttr = builder.getStringAttr(oldModule.getName());
      if (mem.readLatency == 1 && mem.writeLatency == 1) {
        builder.create<HWModuleExternOp>(oldModule.getLoc(), nameAttr,
            oldModule.getPorts());
      }else {

        auto newModule = builder.create<HWModuleOp>(oldModule.getLoc(), nameAttr,
            oldModule.getPorts());
        generateMemory(newModule, mem);
      }
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
