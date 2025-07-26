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

#include "circt/Dialect/Comb/CombOps.h"
#include "circt/Dialect/Emit/EmitOps.h"
#include "circt/Dialect/HW/HWAttributes.h"
#include "circt/Dialect/HW/HWOps.h"
#include "circt/Dialect/HW/HWSymCache.h"
#include "circt/Dialect/HW/InnerSymbolNamespace.h"
#include "circt/Dialect/SV/SVOps.h"
#include "circt/Dialect/Seq/SeqAttributes.h"
#include "circt/Dialect/Seq/SeqOps.h"
#include "circt/Dialect/Seq/SeqPasses.h"
#include "mlir/IR/ImplicitLocOpBuilder.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/Path.h"

using namespace circt;
using namespace hw;
using namespace seq;

namespace circt {
namespace seq {
#define GEN_PASS_DEF_HWMEMSIMIMPL
#include "circt/Dialect/Seq/SeqPasses.h.inc"
} // namespace seq
} // namespace circt

//===----------------------------------------------------------------------===//
// HWMemSimImplPass Pass
//===----------------------------------------------------------------------===//

namespace {

class HWMemSimImpl {
  ReadEnableMode readEnableMode;
  bool addMuxPragmas;
  bool disableMemRandomization;
  bool disableRegRandomization;
  bool addVivadoRAMAddressConflictSynthesisBugWorkaround;

  SmallVector<sv::RegOp> registers;

  Value addPipelineStages(ImplicitLocOpBuilder &b,
                          hw::InnerSymbolNamespace &moduleNamespace,
                          size_t stages, Value clock, Value data,
                          const Twine &name, Value gate = {});
  sv::AlwaysOp lastPipelineAlwaysOp;

public:
  Namespace &mlirModuleNamespace;

  HWMemSimImpl(ReadEnableMode readEnableMode, bool addMuxPragmas,
               bool disableMemRandomization, bool disableRegRandomization,
               bool addVivadoRAMAddressConflictSynthesisBugWorkaround,
               Namespace &mlirModuleNamespace)
      : readEnableMode(readEnableMode), addMuxPragmas(addMuxPragmas),
        disableMemRandomization(disableMemRandomization),
        disableRegRandomization(disableRegRandomization),
        addVivadoRAMAddressConflictSynthesisBugWorkaround(
            addVivadoRAMAddressConflictSynthesisBugWorkaround),
        mlirModuleNamespace(mlirModuleNamespace) {}

  void generateMemory(HWModuleOp op, FirMemory mem);
};

struct HWMemSimImplPass : public impl::HWMemSimImplBase<HWMemSimImplPass> {
  using HWMemSimImplBase::HWMemSimImplBase;

  void runOnOperation() override;
};

} // end anonymous namespace

/// A helper that returns true if a value definition (or block argument) is
/// visible to another operation, either because it's a block argument or
/// because the defining op is before that other op.
static bool valueDefinedBeforeOp(Value value, Operation *op) {
  Operation *valueOp = value.getDefiningOp();
  Block *valueBlock =
      valueOp ? valueOp->getBlock() : cast<BlockArgument>(value).getOwner();
  while (op->getBlock() && op->getBlock() != valueBlock)
    op = op->getParentOp();
  return valueBlock == op->getBlock() &&
         (!valueOp || valueOp->isBeforeInBlock(op));
}

//
// Construct memory read annotated with mux pragmas in the following
// form:
// ```
//   wire GEN;
//   /* synopsys infer_mux_override */
//   assign GEN = memory[addr] /* cadence map_to_mux */;
// ```
// If `addMuxPragmas` is enabled, just return the read value without
// annotations.
static Value getMemoryRead(ImplicitLocOpBuilder &b, Value memory, Value addr,
                           bool addMuxPragmas) {
  auto slot = sv::ReadInOutOp::create(
      b, sv::ArrayIndexInOutOp::create(b, memory, addr));
  // If we don't want to add mux pragmas, just return the read value.
  if (!addMuxPragmas ||
      cast<hw::UnpackedArrayType>(
          cast<hw::InOutType>(memory.getType()).getElementType())
              .getNumElements() <= 1)
    return slot;
  circt::sv::setSVAttributes(
      slot, sv::SVAttributeAttr::get(b.getContext(), "cadence map_to_mux",
                                     /*emitAsComment=*/true));
  auto valWire = sv::WireOp::create(b, slot.getType());
  auto assignOp = sv::AssignOp::create(b, valWire, slot);
  sv::setSVAttributes(assignOp,
                      sv::SVAttributeAttr::get(b.getContext(),
                                               "synopsys infer_mux_override",
                                               /*emitAsComment=*/true));

  return sv::ReadInOutOp::create(b, valWire);
}

Value HWMemSimImpl::addPipelineStages(ImplicitLocOpBuilder &b,
                                      hw::InnerSymbolNamespace &moduleNamespace,
                                      size_t stages, Value clock, Value data,
                                      const Twine &name, Value gate) {
  if (!stages)
    return data;

  // Try to reuse the previous always block. This is only possible if the clocks
  // agree and the data and gate all dominate the always block.
  auto alwaysOp = lastPipelineAlwaysOp;
  if (alwaysOp) {
    if (alwaysOp.getClocks() != ValueRange{clock} ||
        !valueDefinedBeforeOp(data, alwaysOp) ||
        (gate && !valueDefinedBeforeOp(gate, alwaysOp)))
      alwaysOp = {};
  }
  if (!alwaysOp)
    alwaysOp = sv::AlwaysOp::create(b, sv::EventControl::AtPosEdge, clock);

  // Add the necessary registers.
  auto savedIP = b.saveInsertionPoint();
  SmallVector<sv::RegOp> regs;
  b.setInsertionPoint(alwaysOp);
  for (unsigned i = 0; i < stages; ++i) {
    auto regName =
        b.getStringAttr(moduleNamespace.newName("_" + name + "_d" + Twine(i)));
    auto reg = sv::RegOp::create(b, data.getType(), regName,
                                 hw::InnerSymAttr::get(regName));
    regs.push_back(reg);
    registers.push_back(reg);
  }

  // Populate the assignments in the always block.
  b.setInsertionPointToEnd(alwaysOp.getBodyBlock());
  for (unsigned i = 0; i < stages; ++i) {
    if (i > 0)
      data = sv::ReadInOutOp::create(b, data);
    auto emitAssign = [&] { sv::PAssignOp::create(b, regs[i], data); };
    if (gate)
      sv::IfOp::create(b, gate, [&]() { emitAssign(); });
    else
      emitAssign();
    data = regs[i];
    gate = {};
  }
  b.restoreInsertionPoint(savedIP);
  data = sv::ReadInOutOp::create(b, data);

  lastPipelineAlwaysOp = alwaysOp;
  return data;
}

void HWMemSimImpl::generateMemory(HWModuleOp op, FirMemory mem) {
  ImplicitLocOpBuilder b(op.getLoc(), op.getBody());

  InnerSymbolNamespace moduleNamespace(op);

  // Compute total number of mask bits.
  if (mem.maskGran == 0)
    mem.maskGran = mem.dataWidth;
  auto maskBits = mem.dataWidth / mem.maskGran;
  bool isMasked = maskBits > 1;
  // Each mask bit controls mask-granularity number of data bits.
  auto dataType = b.getIntegerType(mem.dataWidth);

  // Count the total number of ports.
  unsigned numPorts =
      mem.numReadPorts + mem.numWritePorts + mem.numReadWritePorts;

  // Create registers for the memory.
  sv::RegOp reg =
      sv::RegOp::create(b, UnpackedArrayType::get(dataType, mem.depth),
                        b.getStringAttr("Memory"));

  if (addVivadoRAMAddressConflictSynthesisBugWorkaround) {
    if (mem.readLatency == 0) {
      // If the read latency is zero, we regard the memory as write-first.
      // We add a SV attribute to specify a ram style to use LUTs for Vivado
      // to avoid a bug that miscompiles the write-first memory. See "RAM
      // address conflict and Vivado synthesis bug" issue in the vivado forum
      // for the more detail.
      circt::sv::setSVAttributes(
          reg, sv::SVAttributeAttr::get(b.getContext(), "ram_style",
                                        R"("distributed")",
                                        /*emitAsComment=*/false));
    } else if (mem.readLatency == 1 && numPorts > 1) {
      // If the read address is registered and the RAM has multiple ports,
      // force write-first behaviour by setting rw_addr_collision. This avoids
      // unpredictable behaviour. Downstreams flows should watch for `VPL
      // 8-6430`.
      circt::sv::setSVAttributes(
          reg, sv::SVAttributeAttr::get(b.getContext(), "rw_addr_collision",
                                        R"("yes")", /*emitAsComment=*/false));
    }
  }

  SmallVector<Value, 4> outputs;

  size_t inArg = 0;
  for (size_t i = 0; i < mem.numReadPorts; ++i) {
    Value addr = op.getBody().getArgument(inArg++);
    Value en = op.getBody().getArgument(inArg++);
    Value clock = op.getBody().getArgument(inArg++);
    // Add pipeline stages
    if (readEnableMode == ReadEnableMode::Ignore) {
      for (size_t j = 0, e = mem.readLatency; j != e; ++j) {
        auto enLast = en;
        if (j < e - 1)
          en = addPipelineStages(b, moduleNamespace, 1, clock, en,
                                 "R" + Twine(i) + "_en");
        addr = addPipelineStages(b, moduleNamespace, 1, clock, addr,
                                 "R" + Twine(i) + "_addr", enLast);
      }
    } else {
      en = addPipelineStages(b, moduleNamespace, mem.readLatency, clock, en,
                             "R" + Twine(i) + "_en");
      addr = addPipelineStages(b, moduleNamespace, mem.readLatency, clock, addr,
                               "R" + Twine(i) + "_addr");
    }

    // Read Logic
    Value rdata = getMemoryRead(b, reg, addr, addMuxPragmas);
    switch (readEnableMode) {
    case ReadEnableMode::Undefined: {
      Value x = sv::ConstantXOp::create(b, rdata.getType());
      rdata = comb::MuxOp::create(b, en, rdata, x, false);
      break;
    }
    case ReadEnableMode::Zero: {
      Value x = hw::ConstantOp::create(b, rdata.getType(), 0);
      rdata = comb::MuxOp::create(b, en, rdata, x, false);
      break;
    }
    case ReadEnableMode::Ignore:
      break;
    }
    outputs.push_back(rdata);
  }

  for (size_t i = 0; i < mem.numReadWritePorts; ++i) {
    auto numReadStages = mem.readLatency;
    auto numWriteStages = mem.writeLatency - 1;
    auto numCommonStages = std::min(numReadStages, numWriteStages);
    Value addr = op.getBody().getArgument(inArg++);
    Value en = op.getBody().getArgument(inArg++);
    Value clock = op.getBody().getArgument(inArg++);
    Value wmode = op.getBody().getArgument(inArg++);
    Value wdataIn = op.getBody().getArgument(inArg++);
    Value wmaskBits;
    // There are no input mask ports, if maskBits =1. Create a dummy true value
    // for mask.
    if (isMasked)
      wmaskBits = op.getBody().getArgument(inArg++);
    else
      wmaskBits = ConstantOp::create(b, b.getIntegerAttr(en.getType(), 1));

    // Add common pipeline stages.
    addr = addPipelineStages(b, moduleNamespace, numCommonStages, clock, addr,
                             "RW" + Twine(i) + "_addr");
    en = addPipelineStages(b, moduleNamespace, numCommonStages, clock, en,
                           "RW" + Twine(i) + "_en");
    wmode = addPipelineStages(b, moduleNamespace, numCommonStages, clock, wmode,
                              "RW" + Twine(i) + "_mode");

    // Add read-only pipeline stages.
    Value readAddr = addr;
    Value readEn = en;
    if (readEnableMode == ReadEnableMode::Ignore) {
      for (size_t j = 0, e = mem.readLatency; j != e; ++j) {
        auto enLast = en;
        if (j < e - 1)
          readEn = addPipelineStages(b, moduleNamespace, 1, clock, en,
                                     "RW" + Twine(i) + "_ren");
        readAddr = addPipelineStages(b, moduleNamespace, 1, clock, addr,
                                     "RW" + Twine(i) + "_raddr", enLast);
      }
    } else {
      readAddr =
          addPipelineStages(b, moduleNamespace, numReadStages - numCommonStages,
                            clock, addr, "RW" + Twine(i) + "_raddr");
      readEn =
          addPipelineStages(b, moduleNamespace, numReadStages - numCommonStages,
                            clock, en, "RW" + Twine(i) + "_ren");
    }
    auto readWMode =
        addPipelineStages(b, moduleNamespace, numReadStages - numCommonStages,
                          clock, wmode, "RW" + Twine(i) + "_rmode");

    // Add write-only pipeline stages.
    auto writeAddr =
        addPipelineStages(b, moduleNamespace, numWriteStages - numCommonStages,
                          clock, addr, "RW" + Twine(i) + "_waddr");
    auto writeEn =
        addPipelineStages(b, moduleNamespace, numWriteStages - numCommonStages,
                          clock, en, "RW" + Twine(i) + "_wen");
    auto writeWMode =
        addPipelineStages(b, moduleNamespace, numWriteStages - numCommonStages,
                          clock, wmode, "RW" + Twine(i) + "_wmode");
    wdataIn = addPipelineStages(b, moduleNamespace, numWriteStages, clock,
                                wdataIn, "RW" + Twine(i) + "_wdata");
    if (isMasked)
      wmaskBits = addPipelineStages(b, moduleNamespace, numWriteStages, clock,
                                    wmaskBits, "RW" + Twine(i) + "_wmask");

    SmallVector<Value, 4> maskValues(maskBits);
    SmallVector<Value, 4> dataValues(maskBits);
    // For multi-bit mask, extract corresponding write data bits of
    // mask-granularity size each. Each of the extracted data bits will be
    // written to a register, gaurded by the corresponding mask bit.
    for (size_t i = 0; i < maskBits; ++i) {
      maskValues[i] = b.createOrFold<comb::ExtractOp>(wmaskBits, i, 1);
      dataValues[i] = b.createOrFold<comb::ExtractOp>(wdataIn, i * mem.maskGran,
                                                      mem.maskGran);
    }

    // wire to store read result
    auto rWire = sv::WireOp::create(b, wdataIn.getType());
    Value rdata = sv::ReadInOutOp::create(b, rWire);

    // Read logic.
    Value rcond = b.createOrFold<comb::AndOp>(
        readEn,
        b.createOrFold<comb::ICmpOp>(
            comb::ICmpPredicate::eq, readWMode,
            b.createOrFold<ConstantOp>(readWMode.getType(), 0), false),
        false);

    auto val = getMemoryRead(b, reg, readAddr, addMuxPragmas);

    switch (readEnableMode) {
    case ReadEnableMode::Undefined: {
      Value x = sv::ConstantXOp::create(b, val.getType());
      val = comb::MuxOp::create(b, rcond, val, x, false);
      break;
    }
    case ReadEnableMode::Zero: {
      Value x = hw::ConstantOp::create(b, val.getType(), 0);
      val = comb::MuxOp::create(b, rcond, val, x, false);
      break;
    }
    case ReadEnableMode::Ignore:
      break;
    }
    sv::AssignOp::create(b, rWire, val);

    // Write logic gaurded by the corresponding mask bit.
    for (auto wmask : llvm::enumerate(maskValues)) {
      sv::AlwaysOp::create(b, sv::EventControl::AtPosEdge, clock, [&]() {
        auto wcond = b.createOrFold<comb::AndOp>(
            writeEn,
            b.createOrFold<comb::AndOp>(wmask.value(), writeWMode, false),
            false);
        sv::IfOp::create(b, wcond, [&]() {
          Value slotReg = sv::ArrayIndexInOutOp::create(b, reg, writeAddr);
          sv::PAssignOp::create(
              b,
              b.createOrFold<sv::IndexedPartSelectInOutOp>(
                  slotReg,
                  b.createOrFold<ConstantOp>(b.getIntegerType(32),
                                             wmask.index() * mem.maskGran),
                  mem.maskGran),
              dataValues[wmask.index()]);
        });
      });
    }
    outputs.push_back(rdata);
  }

  DenseMap<unsigned, Operation *> writeProcesses;
  for (size_t i = 0; i < mem.numWritePorts; ++i) {
    auto numStages = mem.writeLatency - 1;
    Value addr = op.getBody().getArgument(inArg++);
    Value en = op.getBody().getArgument(inArg++);
    Value clock = op.getBody().getArgument(inArg++);
    Value wdataIn = op.getBody().getArgument(inArg++);
    Value wmaskBits;
    // There are no input mask ports, if maskBits =1. Create a dummy true value
    // for mask.
    if (isMasked)
      wmaskBits = op.getBody().getArgument(inArg++);
    else
      wmaskBits = ConstantOp::create(b, b.getIntegerAttr(en.getType(), 1));
    // Add pipeline stages
    addr = addPipelineStages(b, moduleNamespace, numStages, clock, addr,
                             "W" + Twine(i) + "addr");
    en = addPipelineStages(b, moduleNamespace, numStages, clock, en,
                           "W" + Twine(i) + "en");
    wdataIn = addPipelineStages(b, moduleNamespace, numStages, clock, wdataIn,
                                "W" + Twine(i) + "data");
    if (isMasked)
      wmaskBits = addPipelineStages(b, moduleNamespace, numStages, clock,
                                    wmaskBits, "W" + Twine(i) + "mask");

    SmallVector<Value, 4> maskValues(maskBits);
    SmallVector<Value, 4> dataValues(maskBits);
    // For multi-bit mask, extract corresponding write data bits of
    // mask-granularity size each. Each of the extracted data bits will be
    // written to a register, gaurded by the corresponding mask bit.
    for (size_t i = 0; i < maskBits; ++i) {
      maskValues[i] = b.createOrFold<comb::ExtractOp>(wmaskBits, i, 1);
      dataValues[i] = b.createOrFold<comb::ExtractOp>(wdataIn, i * mem.maskGran,
                                                      mem.maskGran);
    }
    // Build write port logic.
    auto writeLogic = [&] {
      // For each register, create the connections to write the corresponding
      // data into it.
      for (auto wmask : llvm::enumerate(maskValues)) {
        // Guard by corresponding mask bit.
        auto wcond = b.createOrFold<comb::AndOp>(en, wmask.value(), false);
        sv::IfOp::create(b, wcond, [&]() {
          auto slot = sv::ArrayIndexInOutOp::create(b, reg, addr);
          sv::PAssignOp::create(
              b,
              b.createOrFold<sv::IndexedPartSelectInOutOp>(
                  slot,
                  b.createOrFold<ConstantOp>(b.getIntegerType(32),
                                             wmask.index() * mem.maskGran),
                  mem.maskGran),
              dataValues[wmask.index()]);
        });
      }
    };

    // Build a new always block with write port logic.
    auto alwaysBlock = [&] {
      return sv::AlwaysOp::create(b, sv::EventControl::AtPosEdge, clock,
                                  [&]() { writeLogic(); });
    };

    switch (mem.writeUnderWrite) {
    // Undefined write order:  lower each write port into a separate always
    // block.
    case seq::WUW::Undefined:
      alwaysBlock();
      break;
    // Port-ordered write order:  lower each write port into an always block
    // based on its clock ID.
    case seq::WUW::PortOrder:
      if (auto *existingAlwaysBlock =
              writeProcesses.lookup(mem.writeClockIDs[i])) {
        OpBuilder::InsertionGuard guard(b);
        b.setInsertionPointToEnd(
            cast<sv::AlwaysOp>(existingAlwaysBlock).getBodyBlock());
        writeLogic();
      } else {
        writeProcesses[i] = alwaysBlock();
      }
    }
  }

  auto *outputOp = op.getBodyBlock()->getTerminator();
  outputOp->setOperands(outputs);

  // Add logic to initialize the memory based on a file emission request.  This
  // disables randomization.
  if (!mem.initFilename.empty()) {
    // Set an inner symbol on the register if one does not exist.
    if (!reg.getInnerSymAttr())
      reg.setInnerSymAttr(hw::InnerSymAttr::get(
          b.getStringAttr(moduleNamespace.newName(reg.getName()))));

    if (mem.initIsInline) {
      sv::IfDefOp::create(b, "ENABLE_INITIAL_MEM_", [&]() {
        sv::InitialOp::create(b, [&]() {
          sv::ReadMemOp::create(b, reg, mem.initFilename,
                                mem.initIsBinary ? MemBaseTypeAttr::MemBaseBin
                                                 : MemBaseTypeAttr::MemBaseHex);
        });
      });
    } else {
      OpBuilder::InsertionGuard guard(b);

      // Assign a name to the bound module.
      StringAttr boundModuleName =
          b.getStringAttr(mlirModuleNamespace.newName(op.getName() + "_init"));

      // Generate a name for the file containing the bound module and the bind.
      StringAttr filename;
      if (auto fileAttr = op->getAttrOfType<OutputFileAttr>("output_file")) {
        if (!fileAttr.isDirectory()) {
          SmallString<128> path(fileAttr.getFilename().getValue());
          llvm::sys::path::remove_filename(path);
          llvm::sys::path::append(path, boundModuleName.getValue() + ".sv");
          filename = b.getStringAttr(path);
        } else {
          filename = fileAttr.getFilename();
        }
      } else {
        filename = b.getStringAttr(boundModuleName.getValue() + ".sv");
      }

      // Create a new module with the readmem op.
      b.setInsertionPointAfter(op);
      auto boundModule =
          HWModuleOp::create(b, boundModuleName, ArrayRef<PortInfo>());

      // Build the hierpathop
      auto path = hw::HierPathOp::create(
          b, mlirModuleNamespace.newName(op.getName() + "_path"),
          b.getArrayAttr(
              ::InnerRefAttr::get(op.getNameAttr(), reg.getInnerNameAttr())));

      b.setInsertionPointToStart(boundModule.getBodyBlock());
      sv::InitialOp::create(b, [&]() {
        auto xmr =
            sv::XMRRefOp::create(b, reg.getType(), path.getSymNameAttr());
        sv::ReadMemOp::create(b, xmr, mem.initFilename,
                              mem.initIsBinary ? MemBaseTypeAttr::MemBaseBin
                                               : MemBaseTypeAttr::MemBaseHex);
      });

      // Instantiate this new module inside the memory module.
      b.setInsertionPointAfter(reg);
      auto boundInstance = hw::InstanceOp::create(
          b, boundModule, boundModule.getName(), ArrayRef<Value>());
      boundInstance->setAttr(
          "inner_sym",
          hw::InnerSymAttr::get(b.getStringAttr(
              moduleNamespace.newName(boundInstance.getInstanceName()))));
      boundInstance.setDoNotPrintAttr(b.getUnitAttr());

      // Build the file container and reference the module from it.
      b.setInsertionPointAfter(op);
      emit::FileOp::create(b, filename, [&] {
        emit::RefOp::create(b, FlatSymbolRefAttr::get(boundModuleName));
        sv::BindOp::create(b,
                           hw::InnerRefAttr::get(
                               op.getNameAttr(),
                               boundInstance.getInnerSymAttr().getSymName()));
      });
    }
  }

  // Add logic to initialize the memory and any internal registers to random
  // values.
  if (disableMemRandomization && disableRegRandomization)
    return;

  constexpr unsigned randomWidth = 32;
  sv::IfDefOp::create(b, "ENABLE_INITIAL_MEM_", [&]() {
    sv::RegOp randReg;
    SmallVector<sv::RegOp> randRegs;
    if (!disableRegRandomization) {
      sv::IfDefOp::create(b, "RANDOMIZE_REG_INIT", [&]() {
        signed totalWidth = 0;
        for (sv::RegOp &reg : registers)
          totalWidth += reg.getElementType().getIntOrFloatBitWidth();
        while (totalWidth > 0) {
          auto name = b.getStringAttr(moduleNamespace.newName("_RANDOM"));
          auto innerSym = hw::InnerSymAttr::get(name);
          randRegs.push_back(sv::RegOp::create(b, b.getIntegerType(randomWidth),
                                               name, innerSym));
          totalWidth -= randomWidth;
        }
      });
    }
    auto randomMemReg = sv::RegOp::create(
        b,
        b.getIntegerType(llvm::divideCeil(mem.dataWidth, randomWidth) *
                         randomWidth),
        b.getStringAttr("_RANDOM_MEM"));
    sv::InitialOp::create(b, [&]() {
      sv::VerbatimOp::create(b, "`INIT_RANDOM_PROLOG_");

      // Memory randomization logic.  The entire memory is randomized.
      if (!disableMemRandomization) {
        sv::IfDefProceduralOp::create(b, "RANDOMIZE_MEM_INIT", [&]() {
          auto outerLoopIndVarType =
              b.getIntegerType(llvm::Log2_64_Ceil(mem.depth + 1));
          auto innerUpperBoundWidth =
              cast<IntegerType>(randomMemReg.getType().getElementType())
                  .getWidth();
          auto innerLoopIndVarType =
              b.getIntegerType(llvm::Log2_64_Ceil(innerUpperBoundWidth + 1));
          // Construct the following nested for loops:
          // ```
          //   for (int i = 0; i < mem.depth; i++) begin
          //     for (int j = 0; j < randomMeg.size; j += 32)
          //       randomMem[j[mem.width-1: +: 32] = `RANDOM
          //     Memory[i] = randomMem[mem.dataWidth - 1: 0];
          // ```
          sv::ForOp::create(
              b, 0, mem.depth, 1, outerLoopIndVarType, "i",
              [&](BlockArgument outerIndVar) {
                sv::ForOp::create(
                    b, 0, innerUpperBoundWidth, randomWidth,
                    innerLoopIndVarType, "j", [&](BlockArgument innerIndVar) {
                      auto rhs = sv::MacroRefExprSEOp::create(
                          b, b.getIntegerType(randomWidth), "RANDOM");
                      Value truncInnerIndVar;
                      if (mem.dataWidth <= 1)
                        truncInnerIndVar =
                            hw::ConstantOp::create(b, b.getI1Type(), 0);
                      else
                        truncInnerIndVar = b.createOrFold<comb::ExtractOp>(
                            innerIndVar, 0, llvm::Log2_64_Ceil(mem.dataWidth));
                      auto lhs = sv::IndexedPartSelectInOutOp::create(
                          b, randomMemReg, truncInnerIndVar, randomWidth,
                          false);
                      sv::BPAssignOp::create(b, lhs, rhs);
                    });

                Value iterValue = outerIndVar;
                // Truncate the induction variable if necessary.
                if (!outerIndVar.getType().isInteger(
                        llvm::Log2_64_Ceil(mem.depth)))
                  iterValue = b.createOrFold<comb::ExtractOp>(
                      iterValue, 0, llvm::Log2_64_Ceil(mem.depth));
                auto lhs = sv::ArrayIndexInOutOp::create(b, reg, iterValue);
                auto rhs = b.createOrFold<comb::ExtractOp>(
                    sv::ReadInOutOp::create(b, randomMemReg), 0, mem.dataWidth);
                sv::BPAssignOp::create(b, lhs, rhs);
              });
        });
      }

      // Register randomization logic.  Randomize every register to a random
      // making efficient use of available randomization registers.
      //
      // TODO: This shares a lot of common logic with LowerToHW.  Combine
      // these two in a common randomization utility.
      if (!disableRegRandomization) {
        sv::IfDefProceduralOp::create(b, "RANDOMIZE_REG_INIT", [&]() {
          unsigned bits = randomWidth;
          for (sv::RegOp &reg : randRegs)
            sv::VerbatimOp::create(
                b, b.getStringAttr("{{0}} = {`RANDOM};"), ValueRange{},
                b.getArrayAttr(hw::InnerRefAttr::get(op.getNameAttr(),
                                                     reg.getInnerNameAttr())));
          auto randRegIdx = 0;
          for (sv::RegOp &reg : registers) {
            SmallVector<std::pair<Attribute, std::pair<size_t, size_t>>> values;
            auto width = reg.getElementType().getIntOrFloatBitWidth();
            auto widthRemaining = width;
            while (widthRemaining > 0) {
              if (bits == randomWidth) {
                randReg = randRegs[randRegIdx++];
                bits = 0;
              }
              auto innerRef = hw::InnerRefAttr::get(op.getNameAttr(),
                                                    randReg.getInnerNameAttr());
              if (widthRemaining <= randomWidth - bits) {
                values.push_back({innerRef, {bits + widthRemaining - 1, bits}});
                bits += widthRemaining;
                widthRemaining = 0;
                continue;
              }
              values.push_back({innerRef, {randomWidth - 1, bits}});
              widthRemaining -= (randomWidth - bits);
              bits = randomWidth;
            }
            SmallString<32> rhs("{{0}} = ");
            unsigned idx = 1;
            assert(reg.getInnerSymAttr());
            SmallVector<Attribute, 4> symbols({hw::InnerRefAttr::get(
                op.getNameAttr(), reg.getInnerNameAttr())});
            if (values.size() > 1)
              rhs.append("{");
            for (auto &v : values) {
              if (idx > 1)
                rhs.append(", ");
              auto [sym, range] = v;
              symbols.push_back(sym);
              rhs.append(("{{" + Twine(idx++) + "}}").str());
              // Do not emit a part select as the whole value is used.
              if (range.first == randomWidth - 1 && range.second == 0)
                continue;
              // Emit a single bit part select, e.g., "[3]"
              if (range.first == range.second) {
                rhs.append(("[" + Twine(range.first) + "]").str());
                continue;
              }
              // Emit a part select, e.g., "[4:2]"
              rhs.append(
                  ("[" + Twine(range.first) + ":" + Twine(range.second) + "]")
                      .str());
            }
            if (values.size() > 1)
              rhs.append("}");
            rhs.append(";");
            sv::VerbatimOp::create(b, rhs, ValueRange{},
                                   b.getArrayAttr(symbols));
          }
        });
      }
    });
  });
}

void HWMemSimImplPass::runOnOperation() {
  auto topModule = getOperation();

  // Populate a namespace from the symbols visible to the top-level MLIR module.
  // Memories with initializations create modules and these need to be legal
  // symbols.
  SymbolCache symbolCache;
  symbolCache.addDefinitions(topModule);
  Namespace mlirModuleNamespace;
  mlirModuleNamespace.add(symbolCache);

  SmallVector<HWModuleGeneratedOp> toErase;
  bool anythingChanged = false;

  for (auto op :
       llvm::make_early_inc_range(topModule.getOps<HWModuleGeneratedOp>())) {
    auto oldModule = cast<HWModuleGeneratedOp>(op);
    auto gen = oldModule.getGeneratorKind();
    auto genOp = cast<HWGeneratorSchemaOp>(
        SymbolTable::lookupSymbolIn(getOperation(), gen));

    if (genOp.getDescriptor() == "FIRRTL_Memory") {
      FirMemory mem(oldModule);

      OpBuilder builder(oldModule);
      auto nameAttr = builder.getStringAttr(oldModule.getName());

      // The requirements for macro replacement:
      // 1. read latency and write latency of one.
      // 2. undefined read-under-write behavior.
      if (replSeqMem && ((mem.readLatency == 1 && mem.writeLatency == 1) &&
                         mem.dataWidth > 0)) {
        HWModuleExternOp::create(builder, oldModule.getLoc(), nameAttr,
                                 oldModule.getPortList());
      } else {
        auto newModule = HWModuleOp::create(builder, oldModule.getLoc(),
                                            nameAttr, oldModule.getPortList());
        if (auto outdir = oldModule->getAttr("output_file"))
          newModule->setAttr("output_file", outdir);
        newModule.setCommentAttr(
            builder.getStringAttr("VCS coverage exclude_file"));
        newModule.setPrivate();

        HWMemSimImpl(readEnableMode, addMuxPragmas, disableMemRandomization,
                     disableRegRandomization,
                     addVivadoRAMAddressConflictSynthesisBugWorkaround,
                     mlirModuleNamespace)
            .generateMemory(newModule, mem);
        if (auto fragments = oldModule->getAttr(emit::getFragmentsAttrName()))
          newModule->setAttr(emit::getFragmentsAttrName(), fragments);
      }

      oldModule.erase();
      anythingChanged = true;
    }
  }

  if (!anythingChanged)
    markAllAnalysesPreserved();
}

std::unique_ptr<Pass>
circt::seq::createHWMemSimImplPass(const HWMemSimImplOptions &options) {
  return std::make_unique<HWMemSimImplPass>(options);
}
