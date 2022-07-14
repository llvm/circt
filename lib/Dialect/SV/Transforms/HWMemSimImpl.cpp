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
  bool ignoreReadEnableMem;
  bool readLatencyIsPropagationDelay;

  SmallVector<sv::RegOp> registers;

  Value addPipelineStages(ImplicitLocOpBuilder &b,
                          ModuleNamespace &moduleNamespace, size_t stages,
                          Value clock, Value data, Value gate = {});
  sv::AlwaysOp lastPipelineAlwaysOp;

public:
  HWMemSimImpl(bool ignoreReadEnableMem, bool readLatencyIsPropagationDelay)
      : ignoreReadEnableMem(ignoreReadEnableMem),
        readLatencyIsPropagationDelay(readLatencyIsPropagationDelay) {}

  void generateMemory(HWModuleOp op, FirMemory mem);
};

struct HWMemSimImplPass : public sv::HWMemSimImplBase<HWMemSimImplPass> {
  void runOnOperation() override;

  using sv::HWMemSimImplBase<HWMemSimImplPass>::ignoreReadEnableMem;
  using sv::HWMemSimImplBase<HWMemSimImplPass>::readLatencyIsPropagationDelay;
  using sv::HWMemSimImplBase<HWMemSimImplPass>::replSeqMem;
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

/// A helper that returns true if a value definition (or block argument) is
/// visible to another operation, either because it's a block argument or
/// because the defining op is before that other op.
static bool valueDefinedBeforeOp(Value value, Operation *op) {
  Operation *valueOp = value.getDefiningOp();
  Block *valueBlock =
      valueOp ? valueOp->getBlock() : value.cast<BlockArgument>().getOwner();
  while (op->getBlock() && op->getBlock() != valueBlock)
    op = op->getParentOp();
  return valueBlock == op->getBlock() &&
         (!valueOp || valueOp->isBeforeInBlock(op));
};

Value HWMemSimImpl::addPipelineStages(ImplicitLocOpBuilder &b,
                                      ModuleNamespace &moduleNamespace,
                                      size_t stages, Value clock, Value data,
                                      Value gate) {
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
    alwaysOp = b.create<sv::AlwaysOp>(sv::EventControl::AtPosEdge, clock);

  // Add the necessary registers.
  auto savedIP = b.saveInsertionPoint();
  SmallVector<sv::RegOp> regs;
  b.setInsertionPoint(alwaysOp);
  for (unsigned i = 0; i < stages; ++i) {
    auto regName = b.getStringAttr(moduleNamespace.newName("_GEN"));
    auto reg = b.create<sv::RegOp>(data.getType(), StringAttr{}, regName);
    regs.push_back(reg);
    registers.push_back(reg);
  }

  // Populate the assignments in the always block.
  b.setInsertionPointToEnd(alwaysOp.getBodyBlock());
  for (unsigned i = 0; i < stages; ++i) {
    if (i > 0)
      data = b.create<sv::ReadInOutOp>(data);
    auto emitAssign = [&] { b.create<sv::PAssignOp>(regs[i], data); };
    if (gate)
      b.create<sv::IfOp>(gate, [&]() { emitAssign(); });
    else
      emitAssign();
    data = regs[i];
    gate = {};
  }
  b.restoreInsertionPoint(savedIP);
  data = b.create<sv::ReadInOutOp>(data);

  lastPipelineAlwaysOp = alwaysOp;
  return data;
}

void HWMemSimImpl::generateMemory(HWModuleOp op, FirMemory mem) {
  ImplicitLocOpBuilder b(op.getLoc(), op.getBody());

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

  // Determine the read and write latencies for the memory. We split the read
  // latency into a pre-array and post-array part: the pre-array part controls
  // after how many cycles the actual contents in the storage array are read,
  // and the post-array part controls how long it takes for the content read
  // from the storage array to appear at the output. Physically speaking ,the
  // pre delay models address decoding and word selection delays, while the post
  // delay models the time it takes for the sense amplifiers to probe a value
  // off the storage bits and propagate it to the output. The write latency
  // consists of just the pre part, since there is no sensing and output
  // propagation that happens.
  //
  // Another way to look at this is that the pre and post delays control how the
  // memory behaves for read-under-write accesses. If a read has a pre-array
  // latency of 4 cycles, it will see writes that land in the storage array in
  // the 4 cycles *after* the read is issued, which is when the pre delay
  // expires and the contents of the storage array are probed.
  //
  // CAVEAT(fschuiki): In the memories originally derived from the Scala FIRRTL
  // implementation the read latency is modeled purely as a pre-array delay.
  // However most SRAMs I've seen in the wild have predominantly a post-array
  // delay where the sense amps are active, while the pre-array delay of the
  // address decoding and word selection is pretty much instantaneous. So I
  // suspect what we actually want is to use the FIRRTL `readLatency` parameter
  // for the post-array delay, but this requires lots of checking to see if
  // something breaks. For the time being the latency-is-post-delay modeling is
  // gated behind a pass option such that we maintain the original behaviour and
  // can do some opt-in checks.
  unsigned writeLatency = mem.writeLatency;
  unsigned readPreLatency = mem.readLatency;
  unsigned readPostLatency = 0;
  if (readLatencyIsPropagationDelay)
    std::swap(readPreLatency, readPostLatency);

  // Determine the read and write ports for the memory.
  struct ReadPort {
    Value addr;
    Value en;
    Value clock;
    unsigned preStagesToAdd;
    unsigned postStagesToAdd;
  };

  struct WritePort {
    Value addr;
    Value en;
    Value clock;
    Value wdata;
    Value wmask;
    unsigned stagesToAdd;
  };

  SmallVector<ReadPort> readPorts;
  SmallVector<WritePort> writePorts;
  auto i1Type = b.getI1Type();
  size_t inArg = 0;

  // Handle read ports.
  for (size_t i = 0; i < mem.numReadPorts; ++i) {
    Value addr = op.getBody().getArgument(inArg++);
    Value en = op.getBody().getArgument(inArg++);
    Value clock = op.getBody().getArgument(inArg++);
    readPorts.push_back({addr, en, clock, readPreLatency, readPostLatency});
  }

  // Handle read-write ports.
  for (size_t i = 0; i < mem.numReadWritePorts; ++i) {
    Value addr = op.getBody().getArgument(inArg++);
    Value en = op.getBody().getArgument(inArg++);
    Value clock = op.getBody().getArgument(inArg++);
    Value wmode = op.getBody().getArgument(inArg++);
    Value wdata = op.getBody().getArgument(inArg++);
    Value wmask = isMasked ? op.getBody().getArgument(inArg++) : Value{};

    // As an optimization, create common pipeline registers for the latency
    // shared among the read and write part of the port.
    auto commonLatency = std::min(readPreLatency, writeLatency - 1);
    addr =
        addPipelineStages(b, moduleNamespace, commonLatency, clock, addr, en);
    wmode =
        addPipelineStages(b, moduleNamespace, commonLatency, clock, wmode, en);
    en = addPipelineStages(b, moduleNamespace, commonLatency, clock, en);

    // Fold the wmode into the enable condition on the read and write paths.
    Value wmodeInv =
        b.createOrFold<comb::XorOp>(wmode, b.create<ConstantOp>(i1Type, 1));
    Value readEn = b.createOrFold<comb::AndOp>(en, wmodeInv);
    Value writeEn = b.createOrFold<comb::AndOp>(en, wmode);
    readPorts.push_back(
        {addr, readEn, clock, readPreLatency - commonLatency, readPostLatency});
    writePorts.push_back(
        {addr, writeEn, clock, wdata, wmask, writeLatency - 1 - commonLatency});
  }

  // Handle write ports.
  for (size_t i = 0; i < mem.numWritePorts; ++i) {
    Value addr = op.getBody().getArgument(inArg++);
    Value en = op.getBody().getArgument(inArg++);
    Value clock = op.getBody().getArgument(inArg++);
    Value wdata = op.getBody().getArgument(inArg++);
    Value wmask = isMasked ? op.getBody().getArgument(inArg++) : Value{};
    writePorts.push_back({addr, en, clock, wdata, wmask, writeLatency - 1});
  }

  // Create the logic for read ports.
  SmallVector<Value, 4> outputs;
  for (auto &port : readPorts) {
    // Add the remaining pre-array delay stages which model the latency until
    // the access to the underlying data array happens.
    auto addr = addPipelineStages(b, moduleNamespace, port.preStagesToAdd,
                                  port.clock, port.addr, port.en);
    auto en = addPipelineStages(b, moduleNamespace, port.preStagesToAdd,
                                port.clock, port.en);

    // Actually probe the array.
    Value rdata = b.create<sv::ArrayIndexInOutOp>(reg, addr);
    rdata = b.create<sv::ReadInOutOp>(rdata);

    // Inject `X` when reading is disabled. This makes simulations fail more
    // obviously if a design inadvertently relies on an SRAM's output during its
    // disabled state, which is generally undefined.
    if (!ignoreReadEnableMem) {
      Value x = b.create<sv::ConstantXOp>(rdata.getType());
      rdata = b.create<comb::MuxOp>(en, rdata, x);
    }

    // Add the post-array delay stages which model the latency from the array
    // through bit lines, sense amplifiers, and output buffer/registers. If we
    // are supposed to ignore read enables (and hold onto the current value
    // instead of propagating an `X`), we can use the enable signal to prevent
    // any updates from propagating through the pipeline. Otherwise we don't
    // gate the pipeline stages such that the `X` always properly propagates.
    rdata =
        addPipelineStages(b, moduleNamespace, port.postStagesToAdd, port.clock,
                          rdata, ignoreReadEnableMem ? en : Value{});
    outputs.push_back(rdata);
  }

  // Create the logic for write ports.
  DenseMap<unsigned, Operation *> writeProcesses;
  for (auto &it : llvm::enumerate(writePorts)) {
    auto &port = it.value();
    auto portIdx = it.index();
    // Add the remaining delay stages which model the latency until the access
    // to the underlying data array happens. (Only `addr` and `en` already have
    // pipeline stages inserted, because they are potentially shared with a read
    // port; `wdata`/`wmask` require the full pipeline still.)
    auto addr = addPipelineStages(b, moduleNamespace, port.stagesToAdd,
                                  port.clock, port.addr, port.en);
    auto en = addPipelineStages(b, moduleNamespace, port.stagesToAdd,
                                port.clock, port.en);
    auto wdata = addPipelineStages(b, moduleNamespace, writeLatency - 1,
                                   port.clock, port.wdata, port.en);
    auto wmask = port.wmask
                     ? addPipelineStages(b, moduleNamespace, writeLatency - 1,
                                         port.clock, port.wmask, port.en)
                     : Value{};

    // If the memory doesn't have a mask, create a constant 1 for convenience.
    if (!wmask)
      wmask = b.create<ConstantOp>(i1Type, 1);

    // For multi-bit mask, extract corresponding write data bits of
    // mask-granularity size each. Each of the extracted data bits will be
    // written to a register, gaurded by the corresponding mask bit.
    SmallVector<Value, 4> maskValues(maskBits);
    SmallVector<Value, 4> dataValues(maskBits);
    for (size_t i = 0; i < maskBits; ++i) {
      maskValues[i] = b.createOrFold<comb::ExtractOp>(wmask, i, 1);
      dataValues[i] = b.createOrFold<comb::ExtractOp>(wdata, i * mem.maskGran,
                                                      mem.maskGran);
    }

    // A helper function to build the logic that writes to the memory array.
    auto writeLogic = [&] {
      for (auto wmask : llvm::enumerate(maskValues)) {
        auto wcond = b.createOrFold<comb::AndOp>(en, wmask.value());
        b.create<sv::IfOp>(wcond, [&]() {
          auto wordPtr = b.create<sv::ArrayIndexInOutOp>(reg, addr);
          auto sliceOffset = b.create<ConstantOp>(b.getIntegerType(32),
                                                  wmask.index() * mem.maskGran);
          auto slicePtr = b.createOrFold<sv::IndexedPartSelectInOutOp>(
              wordPtr, sliceOffset, mem.maskGran);
          b.create<sv::PAssignOp>(slicePtr, dataValues[wmask.index()]);
        });
      }
    };

    // A helper function that builds a new `always` block with write logic.
    auto alwaysBlock = [&] {
      return b.create<sv::AlwaysOp>(sv::EventControl::AtPosEdge, port.clock,
                                    [&]() { writeLogic(); });
    };

    // Implement the write logic depending on the desired WUW behaviour.
    switch (mem.writeUnderWrite) {
    case WUW::Undefined:
      // Undefined write order:  lower each write port into a separate always
      // block.
      alwaysBlock();
      break;
    case WUW::PortOrder:
      // Port-ordered write order:  lower each write port into a single always
      // block based on its clock ID, such that conflicting writes resolve in
      // the order the writes are listed in the block, with the last one
      // winning.
      if (auto *existingAlwaysBlock =
              writeProcesses.lookup(mem.writeClockIDs[portIdx])) {
        OpBuilder::InsertionGuard guard(b);
        b.setInsertionPointToEnd(
            cast<sv::AlwaysOp>(existingAlwaysBlock).getBodyBlock());
        writeLogic();
      } else {
        writeProcesses[portIdx] = alwaysBlock();
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
                op.getNameAttr(), randomMemReg.getInnerSymAttr())}));
      });

      // Register randomization logic.  Randomize every register to a random
      // making efficient use of available randomization registers.
      //
      // TODO: This shares a lot of common logic with LowerToHW.  Combine these
      // two in a common randomization utility.
      b.create<sv::IfDefProceduralOp>("RANDOMIZE_REG_INIT", [&]() {
        unsigned bits = randomWidth;
        for (sv::RegOp &reg : randRegs)
          b.create<sv::VerbatimOp>(
              b.getStringAttr("{{0}} = {`RANDOM};"), ValueRange{},
              b.getArrayAttr(hw::InnerRefAttr::get(op.getNameAttr(),
                                                   reg.getInnerSymAttr())));
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
                                                  randReg.getInnerSymAttr());
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
          SmallVector<Attribute, 4> symbols(
              {hw::InnerRefAttr::get(op.getNameAttr(), reg.getInnerSymAttr())});
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
    auto gen = oldModule.getGeneratorKind();
    auto genOp = cast<HWGeneratorSchemaOp>(
        SymbolTable::lookupSymbolIn(getOperation(), gen));

    if (genOp.getDescriptor() == "FIRRTL_Memory") {
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
        newModule.setCommentAttr(
            builder.getStringAttr("VCS coverage exclude_file"));

        HWMemSimImpl(ignoreReadEnableMem, readLatencyIsPropagationDelay)
            .generateMemory(newModule, mem);
      }

      oldModule.erase();
      anythingChanged = true;
    }
  }

  if (!anythingChanged)
    markAllAnalysesPreserved();
}

std::unique_ptr<Pass>
circt::sv::createHWMemSimImplPass(bool replSeqMem, bool ignoreReadEnableMem,
                                  bool readLatencyIsPropagationDelay) {
  auto pass = std::make_unique<HWMemSimImplPass>();
  pass->replSeqMem = replSeqMem;
  pass->ignoreReadEnableMem = ignoreReadEnableMem;
  pass->readLatencyIsPropagationDelay = readLatencyIsPropagationDelay;
  return pass;
}
