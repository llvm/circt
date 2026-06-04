//===- ESIPasses.cpp - Common code for ESI passes ---------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "PassDetails.h"

#include "circt/Dialect/Comb/CombOps.h"
#include "circt/Dialect/ESI/ESIOps.h"
#include "circt/Dialect/HW/HWOps.h"
#include "circt/Dialect/SV/SVOps.h"
#include "circt/Dialect/Seq/SeqOps.h"
#include "circt/Support/BackedgeBuilder.h"
#include "circt/Support/LLVM.h"

#include "mlir/IR/SymbolTable.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"

using namespace circt;
using namespace circt::esi;
using namespace circt::esi::detail;
using namespace circt::hw;
using namespace circt::sv;

StringAttr circt::esi::detail::getTypeID(Type t) {
  std::string typeID;
  llvm::raw_string_ostream(typeID) << t;
  return StringAttr::get(t.getContext(), typeID);
}

uint64_t circt::esi::detail::getWidth(Type t) {
  if (auto ch = dyn_cast<ChannelType>(t))
    t = ch.getInner();
  if (auto win = dyn_cast<WindowType>(t))
    t = win.getLoweredType();
  return hw::getBitWidth(t);
}

//===----------------------------------------------------------------------===//
// ESI custom op builder.
//===----------------------------------------------------------------------===//

// C++ requires this for showing it what object file it should store these
// symbols in. They should be inline but that feature wasn't added until C++17.
constexpr char ESIHWBuilder::dataStr[], ESIHWBuilder::validStr[],
    ESIHWBuilder::readyStr[], ESIHWBuilder::sourceStr[],
    ESIHWBuilder::sinkStr[];

ESIHWBuilder::ESIHWBuilder(Operation *top)
    : ImplicitLocOpBuilder(UnknownLoc::get(top->getContext()), top),
      a(StringAttr::get(getContext(), "a")),
      aValid(StringAttr::get(getContext(), "a_valid")),
      aReady(StringAttr::get(getContext(), "a_ready")),
      x(StringAttr::get(getContext(), "x")),
      xValid(StringAttr::get(getContext(), "x_valid")),
      xReady(StringAttr::get(getContext(), "x_ready")),
      dataOutValid(StringAttr::get(getContext(), "DataOutValid")),
      dataOutReady(StringAttr::get(getContext(), "DataOutReady")),
      dataOut(StringAttr::get(getContext(), "DataOut")),
      dataInValid(StringAttr::get(getContext(), "DataInValid")),
      dataInReady(StringAttr::get(getContext(), "DataInReady")),
      dataIn(StringAttr::get(getContext(), "DataIn")),
      clk(StringAttr::get(getContext(), "clk")),
      rst(StringAttr::get(getContext(), "rst")),
      width(StringAttr::get(getContext(), "WIDTH")) {

  auto regions = top->getRegions();
  if (regions.empty()) {
    top->emitError("ESI HW Builder needs a region to insert HW.");
  }
  auto &region = regions.front();
  if (!region.empty())
    setInsertionPoint(&region.front(), region.front().begin());
}

static StringAttr constructUniqueSymbol(Operation *tableOp,
                                        StringRef proposedNameRef) {
  SmallString<64> proposedName = proposedNameRef;

  // Normalize the type name.
  for (char &ch : proposedName) {
    if (isalpha(ch) || isdigit(ch) || ch == '_')
      continue;
    ch = '_';
  }

  // Make sure that this symbol isn't taken. If it is, append a number and try
  // again.
  size_t baseLength = proposedName.size();
  size_t tries = 0;
  while (SymbolTable::lookupSymbolIn(tableOp, proposedName)) {
    proposedName.resize(baseLength);
    proposedName.append(llvm::utostr(++tries));
  }

  return StringAttr::get(tableOp->getContext(), proposedName);
}

StringAttr ESIHWBuilder::constructInterfaceName(ChannelType port) {
  Operation *tableOp =
      getInsertionPoint()->getParentWithTrait<mlir::OpTrait::SymbolTable>();

  // Get a name based on the type.
  std::string portTypeName;
  llvm::raw_string_ostream nameOS(portTypeName);
  TypeSwitch<Type>(port.getInner())
      .Case([&](hw::ArrayType arr) {
        nameOS << "ArrayOf" << arr.getNumElements() << 'x'
               << arr.getElementType();
      })
      .Case([&](hw::StructType t) { nameOS << "Struct"; })
      .Default([&](Type t) { nameOS << port.getInner(); });

  // Don't allow the name to end with '_'.
  ssize_t i = portTypeName.size() - 1;
  while (i >= 0 && portTypeName[i] == '_') {
    --i;
  }
  portTypeName = portTypeName.substr(0, i + 1);

  // All stage names start with this.
  SmallString<64> proposedName("IValidReady_");
  proposedName.append(portTypeName);
  return constructUniqueSymbol(tableOp, proposedName);
}

/// Return a parameter list for the stage module with the specified value.
ArrayAttr ESIHWBuilder::getStageParameterList(Attribute value) {
  auto type = IntegerType::get(width.getContext(), 32, IntegerType::Unsigned);
  auto widthParam = ParamDeclAttr::get(width.getContext(), width, type, value);
  return ArrayAttr::get(width.getContext(), widthParam);
}

/// Write an 'ExternModuleOp' to use a hand-coded SystemVerilog module. Said
/// module implements pipeline stage, adding 1 cycle latency. This particular
/// implementation is double-buffered and fully pipelines the reverse-flow ready
/// signal.
HWModuleExternOp ESIHWBuilder::declareStage(Operation *symTable,
                                            PipelineStageOp stage) {
  Type dataType = stage.innerType();
  HWModuleExternOp &stageMod = declaredStage[dataType];
  if (stageMod)
    return stageMod;

  // Since this module has parameterized widths on the a input and x output,
  // give the extern declation a None type since nothing else makes sense.
  // Will be refining this when we decide how to better handle parameterized
  // types and ops.
  size_t argn = 0;
  size_t resn = 0;
  llvm::SmallVector<PortInfo> ports = {
      {{clk, getClockType(), ModulePort::Direction::Input}, argn++},
      {{rst, getI1Type(), ModulePort::Direction::Input}, argn++}};

  ports.push_back({{a, dataType, ModulePort::Direction::Input}, argn++});
  ports.push_back(
      {{aValid, getI1Type(), ModulePort::Direction::Input}, argn++});
  ports.push_back(
      {{aReady, getI1Type(), ModulePort::Direction::Output}, resn++});
  ports.push_back({{x, dataType, ModulePort::Direction::Output}, resn++});

  ports.push_back(
      {{xValid, getI1Type(), ModulePort::Direction::Output}, resn++});
  ports.push_back(
      {{xReady, getI1Type(), ModulePort::Direction::Input}, argn++});

  stageMod = HWModuleExternOp::create(
      *this, constructUniqueSymbol(symTable, "ESI_PipelineStage"), ports,
      "ESI_PipelineStage", getStageParameterList({}));
  return stageMod;
}

/// Construct a concrete (monomorphized) channel buffer module implementing a
/// feed-forward register chain (valid + data) followed by a run-out FIFO which
/// absorbs the backpressure latency. One module is generated per unique
/// (width, stages, slack) tuple and cached. The data is carried as a plain
/// `iN` so the module is reusable across all data types of the same bitwidth.
/// For zero-width data a saturating credit counter replaces the FIFO.
HWModuleOp ESIHWBuilder::declareChannelBuffer(Operation *symTable,
                                              unsigned width, uint64_t stages,
                                              uint64_t slack) {
  auto key = std::make_tuple(width, stages, slack);
  auto cached = declaredChannelBuffer.find(key);
  if (cached != declaredChannelBuffer.end())
    return cached->second;

  Type clkType = getClockType();
  Type i1 = getI1Type();
  Type dataType = IntegerType::get(getContext(), width);
  bool hasData = width > 0;

  // Build the port list. Inputs and outputs are tracked with separate indices,
  // mirroring `declareStage`.
  size_t argn = 0, resn = 0;
  llvm::SmallVector<PortInfo> ports = {
      {{clk, clkType, ModulePort::Direction::Input}, argn++},
      {{rst, i1, ModulePort::Direction::Input}, argn++}};
  if (hasData)
    ports.push_back({{a, dataType, ModulePort::Direction::Input}, argn++});
  ports.push_back({{aValid, i1, ModulePort::Direction::Input}, argn++});
  ports.push_back({{aReady, i1, ModulePort::Direction::Output}, resn++});
  if (hasData)
    ports.push_back({{x, dataType, ModulePort::Direction::Output}, resn++});
  ports.push_back({{xValid, i1, ModulePort::Direction::Output}, resn++});
  ports.push_back({{xReady, i1, ModulePort::Direction::Input}, argn++});

  SmallString<64> name("ESI_FIFOBuffer_w");
  name += llvm::utostr(width);
  name += "_s";
  name += llvm::utostr(stages);
  name += "_k";
  name += llvm::utostr(slack);

  HWModuleOp mod =
      HWModuleOp::create(*this, constructUniqueSymbol(symTable, name), ports);
  declaredChannelBuffer[key] = mod;

  // Build the module body before the auto-created `hw.output` terminator.
  Location loc = mod.getLoc();
  Block *body = mod.getBodyBlock();
  Operation *outputOp = body->getTerminator();
  ImplicitLocOpBuilder b(loc, outputOp);
  BackedgeBuilder bb(b, loc);

  // Map the block arguments (in input-port listing order).
  auto args = body->getArguments();
  size_t argIdx = 0;
  Value clkArg = args[argIdx++];
  Value rstArg = args[argIdx++];
  Value aArg = hasData ? args[argIdx++] : Value();
  Value aValidArg = args[argIdx++];
  Value xReadyArg = args[argIdx++];

  Value c0 = hw::ConstantOp::create(b, i1, b.getBoolAttr(false));
  Value c1 = hw::ConstantOp::create(b, i1, b.getBoolAttr(true));

  // Attach an `sv.namehint` to the op defining `v` so the generated Verilog
  // carries a meaningful net name, then return `v` for inline use.
  auto named = [&](Value v, StringRef hint) -> Value {
    if (Operation *op = v.getDefiningOp())
      op->setAttr("sv.namehint", b.getStringAttr(hint));
    return v;
  };

  // The run-out storage reports `almostFull` to throttle the producer. It is
  // produced below; use a backedge to break the combinational loop, then
  // pipeline it back to the producer (reset to "full" so nothing is accepted
  // until the storage has reported its state).
  Backedge almostFullBE = bb.get(i1);
  Value af = almostFullBE;
  for (uint64_t i = 0; i < stages; ++i)
    af = seq::CompRegOp::create(b, af, clkArg, rstArg, c1, "backpressure");
  Value aReadyVal = named(comb::createOrFoldNot(b, loc, af), "a_ready");
  Value aXfer = named(comb::AndOp::create(b, aValidArg, aReadyVal), "a_xfer");

  // Feed-forward register chain for the valid (and, if present, data) signals.
  Value chainValid = aXfer;
  Value chainData = aArg;
  for (uint64_t i = 0; i < stages; ++i) {
    chainValid =
        seq::CompRegOp::create(b, chainValid, clkArg, rstArg, c0, "ff_valid");
    if (hasData)
      chainData = seq::CompRegOp::create(b, chainData, clkArg, "ff_data");
  }

  // Storage depth: chain length (stages) + tokens sent during the backpressure
  // latency (stages) + user-requested slack.
  uint64_t depth = 2 * stages + slack;

  Value xValidOut, xDataOut;
  if (hasData) {
    // Run-out FIFO. `almostFull` asserts at `slack` entries, leaving room for
    // the (up to) 2*stages tokens still in flight when the producer is stopped.
    //
    // The FIFO uses a registered (synchronous) read (`rdLatency = 1`): a read
    // issued in one cycle delivers its data on `fifo.output` the next cycle,
    // and that data is valid for exactly one cycle (the read register then
    // advances to follow the read pointer). Using a registered read keeps the
    // SRAM read off the module's output path, improving FMax, but means the
    // read result must be captured the cycle it appears.
    //
    // A depth-2 output skid buffer decouples this one-cycle read latency from
    // the downstream valid/ready handshake and presents a registered,
    // full-throughput, hold-under-backpressure output:
    //   * `out0`/`out0Valid` is the consumer-facing registered output.
    //   * `out1`/`out1Valid` is a skid slot that captures a read result which
    //     arrives while the consumer is stalling (a read may already be in
    //     flight when the decision to stop reading is made).
    //   * `rdInFlight` marks that a read issued last cycle is delivering its
    //     (single-cycle-valid) data on `fifo.output` this cycle.
    Backedge fifoRdEn = bb.get(i1);
    auto fifo = seq::FIFOOp::create(
        b, dataType, /*full=*/i1, /*empty=*/i1, /*almostFull=*/i1,
        /*almostEmpty=*/Type(), chainData, fifoRdEn, chainValid, clkArg, rstArg,
        b.getI64IntegerAttr(depth), /*rdLatency=*/b.getI64IntegerAttr(1),
        /*almostFullThreshold=*/b.getI64IntegerAttr(slack),
        /*almostEmptyThreshold=*/IntegerAttr());
    almostFullBE.setValue(fifo.getAlmostFull());
    Value notEmpty =
        named(comb::createOrFoldNot(b, loc, fifo.getEmpty()), "not_empty");

    // Output skid buffer state.
    Backedge out0ValidBE = bb.get(i1);
    Backedge out1ValidBE = bb.get(i1);
    Backedge out0DataBE = bb.get(dataType);
    Backedge out1DataBE = bb.get(dataType);
    Backedge rdInFlightBE = bb.get(i1);
    Value out0Valid = seq::CompRegOp::create(b, out0ValidBE, clkArg, rstArg, c0,
                                             "out0_valid");
    Value out1Valid = seq::CompRegOp::create(b, out1ValidBE, clkArg, rstArg, c0,
                                             "out1_valid");
    Value out0Data = seq::CompRegOp::create(b, out0DataBE, clkArg, "out0_data");
    Value out1Data = seq::CompRegOp::create(b, out1DataBE, clkArg, "out1_data");
    Value rdInFlight = seq::CompRegOp::create(b, rdInFlightBE, clkArg, rstArg,
                                              c0, "rd_in_flight");

    // The head is dequeued when it is valid and the consumer is ready. A read
    // result arrives this cycle (and must be captured) when a read was in
    // flight.
    Value deq = named(comb::AndOp::create(b, out0Valid, xReadyArg), "deq");
    Value arrive = rdInFlight;
    Value aData = fifo.getOutput();

    // Shift the buffer down when the head is dequeued.
    Value notDeq = named(comb::createOrFoldNot(b, loc, deq), "not_deq");
    Value s0Valid = named(comb::MuxOp::create(b, deq, out1Valid, out0Valid),
                          "shift0_valid");
    Value s0Data =
        named(comb::MuxOp::create(b, deq, out1Data, out0Data), "shift0_data");
    Value s1Valid =
        named(comb::AndOp::create(b, notDeq, out1Valid), "shift1_valid");

    // Insert an arriving read result into the lowest free slot. The read-issue
    // condition below guarantees there is always room, so no element is lost.
    Value fillLow =
        named(comb::AndOp::create(b, arrive,
                                  named(comb::createOrFoldNot(b, loc, s0Valid),
                                        "shift0_not_valid")),
              "fill_low");
    Value fillHigh = named(
        comb::AndOp::create(
            b, arrive,
            comb::AndOp::create(b, s0Valid,
                                named(comb::createOrFoldNot(b, loc, s1Valid),
                                      "shift1_not_valid"))),
        "fill_high");
    out0ValidBE.setValue(
        named(comb::OrOp::create(b, s0Valid, fillLow), "out0_valid_next"));
    out0DataBE.setValue(named(comb::MuxOp::create(b, fillLow, aData, s0Data),
                              "out0_data_next"));
    out1ValidBE.setValue(
        named(comb::OrOp::create(b, s1Valid, fillHigh), "out1_valid_next"));
    out1DataBE.setValue(named(comb::MuxOp::create(b, fillHigh, aData, out1Data),
                              "out1_data_next"));

    // Issue a FIFO read whenever it has data and the output buffer, counting
    // the read already in flight, has room for one more element. The buffer
    // holds at most two elements (held plus in flight) by construction, so a
    // read is allowed when at most one of {out0Valid, out1Valid, rdInFlight} is
    // set, or a dequeue this cycle frees a slot.
    Value pairAB =
        comb::AndOp::create(b, out0Valid, out1Valid); // both output slots full
    Value pairAC =
        comb::AndOp::create(b, out0Valid, rdInFlight); // slot0 full + in flight
    Value pairBC =
        comb::AndOp::create(b, out1Valid, rdInFlight); // slot1 full + in flight
    Value anyTwo =
        comb::OrOp::create(b, comb::OrOp::create(b, pairAB, pairAC), pairBC);
    Value countLe1 =
        named(comb::createOrFoldNot(b, loc, anyTwo), "out_count_le1");
    Value roomToRead =
        named(comb::OrOp::create(b, countLe1, deq), "room_to_read");
    Value readFire =
        named(comb::AndOp::create(b, notEmpty, roomToRead), "read_fire");
    fifoRdEn.setValue(readFire);
    rdInFlightBE.setValue(readFire);

    xValidOut = out0Valid;
    xDataOut = out0Data;
  } else {
    // Zero-width: a saturating credit counter tracks the number of buffered
    // tokens. `depth >= 3`, so the counter is always at least 2 bits wide.
    unsigned cntWidth = llvm::Log2_64_Ceil(depth + 1);
    Type cntType = b.getIntegerType(cntWidth);
    Value cntZero = hw::ConstantOp::create(b, cntType, 0);
    Backedge cntNextBE = bb.get(cntType);
    Value cnt =
        seq::CompRegOp::create(b, cntNextBE, clkArg, rstArg, cntZero, "count");
    Value notEmpty =
        named(comb::ICmpOp::create(b, comb::ICmpPredicate::ne, cnt, cntZero),
              "not_empty");
    xValidOut = notEmpty;
    Value deq = named(comb::AndOp::create(b, notEmpty, xReadyArg), "deq");
    Value enq = chainValid;
    // cnt_next = cnt + enq - deq (all zero-extended to the counter width).
    Value zeros = hw::ConstantOp::create(b, b.getIntegerType(cntWidth - 1), 0);
    Value enqExt = named(comb::ConcatOp::create(b, zeros, enq), "enq_ext");
    Value deqExt = named(comb::ConcatOp::create(b, zeros, deq), "deq_ext");
    Value cntNext = named(
        comb::SubOp::create(
            b, named(comb::AddOp::create(b, cnt, enqExt), "count_plus_enq"),
            deqExt),
        "count_next");
    cntNextBE.setValue(cntNext);
    Value slackConst = hw::ConstantOp::create(b, cntType, slack);
    almostFullBE.setValue(named(
        comb::ICmpOp::create(b, comb::ICmpPredicate::uge, cnt, slackConst),
        "almost_full"));
  }

  // Wire up the outputs (in output-port listing order).
  SmallVector<Value> outputs;
  outputs.push_back(aReadyVal);
  if (hasData)
    outputs.push_back(xDataOut);
  outputs.push_back(xValidOut);
  outputOp->setOperands(outputs);

  return mod;
}

/// Write an 'ExternModuleOp' to use a hand-coded SystemVerilog module. Said
/// module contains a bi-directional Cosimulation DPI interface with valid/ready
/// semantics.
HWModuleExternOp
ESIHWBuilder::declareCosimEndpointToHostModule(Operation *symTable) {
  if (declaredCosimEndpointToHostModule)
    return *declaredCosimEndpointToHostModule;

  SmallVector<Attribute, 8> params;
  params.push_back(
      ParamDeclAttr::get("ENDPOINT_ID", NoneType::get(getContext())));
  params.push_back(
      ParamDeclAttr::get("TO_HOST_TYPE_ID", NoneType::get(getContext())));
  params.push_back(ParamDeclAttr::get("TO_HOST_SIZE_BITS", getI32Type()));

  auto dataInType = hw::IntType::get(hw::ParamDeclRefAttr::get(
      StringAttr::get(getContext(), "TO_HOST_SIZE_BITS"),
      getIntegerType(32, false)));
  PortInfo ports[] = {
      {{clk, getClockType(), ModulePort::Direction::Input}, 0},
      {{rst, getI1Type(), ModulePort::Direction::Input}, 1},
      {{dataInValid, getI1Type(), ModulePort::Direction::Input}, 2},
      {{dataInReady, getI1Type(), ModulePort::Direction::Output}, 3},
      {{dataIn, dataInType, ModulePort::Direction::Input}, 4}};

  declaredCosimEndpointToHostModule = HWModuleExternOp::create(
      *this, constructUniqueSymbol(symTable, "Cosim_Endpoint_ToHost"), ports,
      "Cosim_Endpoint_ToHost", ArrayAttr::get(getContext(), params));
  return *declaredCosimEndpointToHostModule;
}

HWModuleExternOp
ESIHWBuilder::declareCosimEndpointFromHostModule(Operation *symTable) {
  if (declaredCosimEndpointFromHostModule)
    return *declaredCosimEndpointFromHostModule;

  SmallVector<Attribute, 8> params;
  params.push_back(
      ParamDeclAttr::get("ENDPOINT_ID", NoneType::get(getContext())));
  params.push_back(
      ParamDeclAttr::get("FROM_HOST_TYPE_ID", NoneType::get(getContext())));
  params.push_back(ParamDeclAttr::get("FROM_HOST_SIZE_BITS", getI32Type()));

  auto dataInType = hw::IntType::get(hw::ParamDeclRefAttr::get(
      StringAttr::get(getContext(), "FROM_HOST_SIZE_BITS"),
      getIntegerType(32, false)));
  PortInfo ports[] = {
      {{clk, getClockType(), ModulePort::Direction::Input}, 0},
      {{rst, getI1Type(), ModulePort::Direction::Input}, 1},
      {{dataOutValid, getI1Type(), ModulePort::Direction::Output}, 2},
      {{dataOutReady, getI1Type(), ModulePort::Direction::Input}, 3},
      {{dataOut, dataInType, ModulePort::Direction::Output}, 4}};

  declaredCosimEndpointFromHostModule = HWModuleExternOp::create(
      *this, constructUniqueSymbol(symTable, "Cosim_Endpoint_FromHost"), ports,
      "Cosim_Endpoint_FromHost", ArrayAttr::get(getContext(), params));
  return *declaredCosimEndpointFromHostModule;
}

/// Return the InterfaceType which corresponds to an ESI port type. If it
/// doesn't exist in the cache, build the InterfaceOp and the corresponding
/// type.
InterfaceOp ESIHWBuilder::getOrConstructInterface(ChannelType t) {
  auto ifaceIter = portTypeLookup.find(t);
  if (ifaceIter != portTypeLookup.end())
    return ifaceIter->second;
  auto iface = constructInterface(t);
  portTypeLookup[t] = iface;
  return iface;
}

InterfaceOp ESIHWBuilder::constructInterface(ChannelType chan) {
  return InterfaceOp::create(
      *this, constructInterfaceName(chan).getValue(), [&]() {
        InterfaceSignalOp::create(*this, validStr, getI1Type());
        InterfaceSignalOp::create(*this, readyStr, getI1Type());
        InterfaceSignalOp::create(*this, dataStr, chan.getInner());
        llvm::SmallVector<StringRef> validDataStrs;
        validDataStrs.push_back(validStr);
        validDataStrs.push_back(dataStr);
        InterfaceModportOp::create(*this, sinkStr,
                                   /*inputs=*/ArrayRef<StringRef>{readyStr},
                                   /*outputs=*/validDataStrs);
        InterfaceModportOp::create(*this, sourceStr,
                                   /*inputs=*/validDataStrs,
                                   /*outputs=*/ArrayRef<StringRef>{readyStr});
      });
}

Type ESIHWBuilder::getClockType() { return seq::ClockType::get(getContext()); }

void circt::esi::registerESIPasses() { registerPasses(); }
