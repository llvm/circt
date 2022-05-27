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
#include "circt/Dialect/HW/Namespace.h"
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
  size_t maskGran;
  size_t readLatency;
  size_t writeLatency;
  size_t readUnderWrite;
  WUW writeUnderWrite;
  SmallVector<int32_t> writeClockIDs;
};
} // end anonymous namespace

namespace {

class HWMemSimImpl {
  MLIRContext &ctx;
  bool ignoreReadEnableMem;

  SmallVector<sv::RegOp> registers;

  Value addPipelineStages(ImplicitLocOpBuilder &b,
                          ModuleNamespace &moduleNamespace, size_t stages,
                          Value clock, Value data, Value gate = {});

public:
  HWMemSimImpl(MLIRContext &ctx, bool replSeqMem, bool ignoreReadEnableMem)
      : ctx(ctx), ignoreReadEnableMem(ignoreReadEnableMem) {}

  void generateMemory(HWModuleOp op, FirMemory mem);
};

struct HWMemSimImplPass : public sv::HWMemSimImplBase<HWMemSimImplPass> {
  void runOnOperation() override;

public:
  HWMemSimImplPass(bool e, bool ignoreEn) {
    replSeqMem = e;
    ignoreReadEnableMem = ignoreEn;
  }
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
  if (op->hasAttrOfType<IntegerAttr>("maskGran"))
    mem.maskGran = op->getAttrOfType<IntegerAttr>("maskGran").getUInt();
  else
    mem.maskGran = mem.dataWidth;
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

Value HWMemSimImpl::addPipelineStages(ImplicitLocOpBuilder &b,
                                      ModuleNamespace &moduleNamespace,
                                      size_t stages, Value clock, Value data,
                                      Value gate) {
  if (!stages)
    return data;

  while (stages--) {
    auto reg =
        b.create<sv::RegOp>(data.getType(), StringAttr{},
                            b.getStringAttr(moduleNamespace.newName("_GEN")));
    registers.push_back(reg);

    // pipeline stage
    b.create<sv::AlwaysOp>(sv::EventControl::AtPosEdge, clock, [&]() {
      if (gate) {
        b.create<sv::IfOp>(gate, [&]() { b.create<sv::PAssignOp>(reg, data); });
      } else {
        b.create<sv::PAssignOp>(reg, data);
      }
    });
    data = b.create<sv::ReadInOutOp>(reg);
  }

  return data;
}

void HWMemSimImpl::generateMemory(HWModuleOp op, FirMemory mem) {
  ImplicitLocOpBuilder b(UnknownLoc::get(&ctx), op.getBody());

  ModuleNamespace moduleNamespace(op);

  // Compute total number of mask bits.
  if (mem.maskGran == 0)
    mem.maskGran = mem.dataWidth;
  auto maskBits = mem.dataWidth / mem.maskGran;
  bool isMasked = maskBits > 1;
  // Each mask bit controls mask-granularity number of data bits.
  auto dataType = b.getIntegerType(mem.dataWidth);

  // Create registers for the memory.
  Value reg = b.create<sv::RegOp>(UnpackedArrayType::get(dataType, mem.depth),
                                  b.getStringAttr("Memory"));

  SmallVector<Value, 4> outputs;

  size_t inArg = 0;
  for (size_t i = 0; i < mem.numReadPorts; ++i) {
    Value addr = op.body().getArgument(inArg++);
    Value en = op.body().getArgument(inArg++);
    Value clock = op.body().getArgument(inArg++);
    // Add pipeline stages
    if (ignoreReadEnableMem) {
      for (size_t j = 0, e = mem.readLatency; j != e; ++j) {
        auto enLast = en;
        if (j < e - 1)
          en = addPipelineStages(b, moduleNamespace, 1, clock, en);
        addr = addPipelineStages(b, moduleNamespace, 1, clock, addr, enLast);
      }
    } else {
      en = addPipelineStages(b, moduleNamespace, mem.readLatency, clock, en);
      addr =
          addPipelineStages(b, moduleNamespace, mem.readLatency, clock, addr);
    }

    // Read Logic
    Value rdata =
        b.create<sv::ReadInOutOp>(b.create<sv::ArrayIndexInOutOp>(reg, addr));
    if (!ignoreReadEnableMem) {
      Value x = b.create<sv::ConstantXOp>(rdata.getType());
      rdata = b.create<comb::MuxOp>(en, rdata, x);
    }
    outputs.push_back(rdata);
  }

  for (size_t i = 0; i < mem.numReadWritePorts; ++i) {
    auto numStages = std::max(mem.readLatency, mem.writeLatency) - 1;
    Value addr = op.body().getArgument(inArg++);
    Value en = op.body().getArgument(inArg++);
    Value clock = op.body().getArgument(inArg++);
    Value wmode = op.body().getArgument(inArg++);
    Value wdataIn = op.body().getArgument(inArg++);
    Value wmaskBits;
    // There are no input mask ports, if maskBits =1. Create a dummy true value
    // for mask.
    if (isMasked)
      wmaskBits = op.body().getArgument(inArg++);
    else
      wmaskBits = b.create<ConstantOp>(b.getIntegerAttr(en.getType(), 1));

    // Add pipeline stages
    addr = addPipelineStages(b, moduleNamespace, numStages, clock, addr);
    en = addPipelineStages(b, moduleNamespace, numStages, clock, en);
    wmode = addPipelineStages(b, moduleNamespace, numStages, clock, wmode);
    wdataIn = addPipelineStages(b, moduleNamespace, numStages, clock, wdataIn);
    if (isMasked)
      wmaskBits =
          addPipelineStages(b, moduleNamespace, numStages, clock, wmaskBits);
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
    auto rWire = b.create<sv::WireOp>(wdataIn.getType());
    Value rdata = b.create<sv::ReadInOutOp>(rWire);

    // Read logic.
    Value rcond = b.createOrFold<comb::AndOp>(
        en, b.createOrFold<comb::ICmpOp>(
                comb::ICmpPredicate::eq, wmode,
                b.createOrFold<ConstantOp>(wmode.getType(), 0)));
    Value slotReg = b.create<sv::ArrayIndexInOutOp>(reg, addr);
    Value slot = b.create<sv::ReadInOutOp>(slotReg);
    Value x = b.create<sv::ConstantXOp>(slot.getType());
    b.create<sv::AssignOp>(rWire, b.create<comb::MuxOp>(rcond, slot, x));

    // Write logic gaurded by the corresponding mask bit.
    for (auto wmask : llvm::enumerate(maskValues)) {
      b.create<sv::AlwaysOp>(sv::EventControl::AtPosEdge, clock, [&]() {
        auto wcond = b.createOrFold<comb::AndOp>(
            en, b.createOrFold<comb::AndOp>(wmask.value(), wmode));
        b.create<sv::IfOp>(wcond, [&]() {
          b.create<sv::PAssignOp>(
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
    Value addr = op.body().getArgument(inArg++);
    Value en = op.body().getArgument(inArg++);
    Value clock = op.body().getArgument(inArg++);
    Value wdataIn = op.body().getArgument(inArg++);
    Value wmaskBits;
    // There are no input mask ports, if maskBits =1. Create a dummy true value
    // for mask.
    if (isMasked)
      wmaskBits = op.body().getArgument(inArg++);
    else
      wmaskBits = b.create<ConstantOp>(b.getIntegerAttr(en.getType(), 1));
    // Add pipeline stages
    addr = addPipelineStages(b, moduleNamespace, numStages, clock, addr);
    en = addPipelineStages(b, moduleNamespace, numStages, clock, en);
    wdataIn = addPipelineStages(b, moduleNamespace, numStages, clock, wdataIn);
    if (isMasked)
      wmaskBits =
          addPipelineStages(b, moduleNamespace, numStages, clock, wmaskBits);

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
        auto wcond = b.createOrFold<comb::AndOp>(en, wmask.value());
        b.create<sv::IfOp>(wcond, [&]() {
          auto slot = b.create<sv::ArrayIndexInOutOp>(reg, addr);
          b.create<sv::PAssignOp>(
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
      return b.create<sv::AlwaysOp>(sv::EventControl::AtPosEdge, clock,
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
            cast<sv::AlwaysOp>(existingAlwaysBlock).getBodyBlock());
        writeLogic();
      } else {
        writeProcesses[i] = alwaysBlock();
      }
    }
  }

  // Add logic to initialize the memory and any internal registers to random
  // values.
  constexpr unsigned randomWidth = 32;
  sv::RegOp randomMemReg;
  b.create<sv::IfDefOp>("SYNTHESIS", std::function<void()>(), [&]() {
    sv::RegOp randReg;
    StringRef initvar;

    // Declare variables for use by memory randomization logic.
    b.create<sv::IfDefOp>("RANDOMIZE_MEM_INIT", [&]() {
      initvar = moduleNamespace.newName("initvar");
      b.create<sv::VerbatimOp>("integer " + Twine(initvar) + ";\n");
      randomMemReg = b.create<sv::RegOp>(
          b.getIntegerType(llvm::divideCeil(mem.dataWidth, randomWidth) *
                           randomWidth),
          b.getStringAttr("_RANDOM_MEM"),
          b.getStringAttr(moduleNamespace.newName("_RANDOM_MEM")));
    });

    // Declare variables for use by register randomization logic.
    SmallVector<sv::RegOp> randRegs;
    b.create<sv::IfDefOp>("RANDOMIZE_REG_INIT", [&]() {
      signed totalWidth = 0;
      for (sv::RegOp &reg : registers)
        totalWidth += reg.getElementType().getIntOrFloatBitWidth();
      while (totalWidth > 0) {
        auto name = b.getStringAttr(moduleNamespace.newName(Twine("_RANDOM")));
        randRegs.push_back(
            b.create<sv::RegOp>(b.getIntegerType(randomWidth), name, name));
        totalWidth -= randomWidth;
      }
    });

    b.create<sv::InitialOp>([&]() {
      b.create<sv::VerbatimOp>("`INIT_RANDOM_PROLOG_");

      // Memory randomization logic.  The entire memory is randomized.
      b.create<sv::IfDefProceduralOp>("RANDOMIZE_MEM_INIT", [&]() {
        std::string verbatimForLoop;
        llvm::raw_string_ostream s(verbatimForLoop);
        s << "for (" << initvar << " = 0; " << initvar << " < " << mem.depth
          << "; " << initvar << " = " << initvar << " + 1) begin\n"
          << "  {{0}} = ";
        auto repetitionCount = llvm::divideCeil(mem.dataWidth, randomWidth);
        if (repetitionCount > 1)
          s << "{";
        for (size_t i = 0; i != repetitionCount; ++i) {
          if (i > 0)
            s << ", ";
          s << "{`RANDOM}";
        }
        if (repetitionCount > 1)
          s << "}";
        s << ";\n";
        s << "  Memory[" << initvar << "] = "
          << "{{0}}[" << mem.dataWidth - 1 << ":" << 0 << "];\n"
          << "end";
        b.create<sv::VerbatimOp>(
            verbatimForLoop, ValueRange{},
            b.getArrayAttr({hw::InnerRefAttr::get(
                op.getNameAttr(), randomMemReg.inner_symAttr())}));
      });

      // Register randomization logic.  Randomize every register to a random
      // making efficient use of available randomization registers.
      //
      // TODO: This shares a lot of common logic with LowerToHW.  Combine these
      // two in a common randomization utility.
      b.create<sv::IfDefProceduralOp>("RANDOMIZE_REG_INIT", [&]() {
        unsigned bits = randomWidth;
        for (sv::RegOp &reg : randRegs)
          b.create<sv::VerbatimOp>(b.getStringAttr("{{0}} = {`RANDOM};"),
                                   ValueRange{},
                                   b.getArrayAttr(hw::InnerRefAttr::get(
                                       op.getNameAttr(), reg.inner_symAttr())));
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
                                                  randReg.inner_symAttr());
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
          assert(reg.inner_symAttr());
          SmallVector<Attribute, 4> symbols(
              {hw::InnerRefAttr::get(op.getNameAttr(), reg.inner_symAttr())});
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
          b.create<sv::VerbatimOp>(rhs, ValueRange{}, b.getArrayAttr(symbols));
        }
      });
    });
  });

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

      // The requirements for macro replacement:
      // 1. read latency and write latency of one.
      // 2. only one readwrite port or write port.
      // 3. zero or one read port.
      // 4. undefined read-under-write behavior.
      if (replSeqMem && ((mem.readLatency == 1 && mem.writeLatency == 1) &&
                         (mem.numWritePorts + mem.numReadWritePorts == 1) &&
                         (mem.numReadPorts <= 1) && mem.dataWidth > 0)) {
        builder.create<HWModuleExternOp>(oldModule.getLoc(), nameAttr,
                                         oldModule.getPorts());
      } else {
        auto newModule = builder.create<HWModuleOp>(
            oldModule.getLoc(), nameAttr, oldModule.getPorts());
        if (auto outdir = oldModule->getAttr("output_file"))
          newModule->setAttr("output_file", outdir);
        newModule.commentAttr(
            builder.getStringAttr("VCS coverage exclude_file"));

        HWMemSimImpl(getContext(), replSeqMem, ignoreReadEnableMem)
            .generateMemory(newModule, mem);
      }

      oldModule.erase();
      anythingChanged = true;
    }
  }

  if (!anythingChanged)
    markAllAnalysesPreserved();
}

std::unique_ptr<Pass> circt::sv::createHWMemSimImplPass(bool replSeqMem,
                                                        bool ignoreReadEnable) {
  return std::make_unique<HWMemSimImplPass>(replSeqMem, ignoreReadEnable);
}
