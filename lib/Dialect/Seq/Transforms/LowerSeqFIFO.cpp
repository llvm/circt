//===- LowerSeqFIFO.cpp - seq.fifo lowering -------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "circt/Dialect/Comb/CombOps.h"
#include "circt/Dialect/HW/HWOps.h"
#include "circt/Dialect/SV/SVOps.h"
#include "circt/Dialect/Seq/SeqOps.h"
#include "circt/Dialect/Seq/SeqPasses.h"
#include "circt/Dialect/Verif/VerifOps.h"
#include "circt/Support/BackedgeBuilder.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include "llvm/ADT/StringSet.h"
#include "llvm/ADT/TypeSwitch.h"

namespace circt {
namespace seq {
#define GEN_PASS_DEF_LOWERSEQFIFO
#include "circt/Dialect/Seq/SeqPasses.h.inc"
} // namespace seq
} // namespace circt

using namespace circt;
using namespace seq;

namespace {

/// The set of parameters which uniquely determine the lowered hardware of a
/// `seq.fifo`. Two FIFOs with identical configurations can share a single
/// outlined module.
struct FIFOConfig {
  Type eltType;
  uint64_t depth;
  uint64_t rdLatency;
  std::optional<uint64_t> almostFull;
  std::optional<uint64_t> almostEmpty;
};

/// Build the FIFO lowering logic using `builder`, reading from the provided
/// input values and returning the result values in the same order as the
/// `seq.fifo` results: output, full, empty, [almostFull], [almostEmpty]. The
/// `bb` backedge builder must be constructed by the caller from the same
/// builder (or, when running under a dialect conversion, from the
/// `PatternRewriter`) so that the temporary backedge placeholders are erased
/// through the right API.
static llvm::SmallVector<Value>
buildFIFOLogic(OpBuilder &builder, BackedgeBuilder &bb, Location loc, Value clk,
               Value rst, Value input, Value rdEn, Value wrEn,
               const FIFOConfig &cfg) {
  Type eltType = cfg.eltType;
  size_t depth = cfg.depth;
  Type countType = builder.getIntegerType(llvm::Log2_64_Ceil(depth + 1));
  Type ptrType = builder.getIntegerType(llvm::Log2_64_Ceil(depth));
  Backedge rdAddrNext = bb.get(ptrType);
  Backedge wrAddrNext = bb.get(ptrType);
  Backedge nextCount = bb.get(countType);

  // ====== Some constants ======
  Value countTcFull = hw::ConstantOp::create(builder, loc, countType, depth);
  Value countTc1 = hw::ConstantOp::create(builder, loc, countType, 1);
  Value countTc0 = hw::ConstantOp::create(builder, loc, countType, 0);
  Value ptrTc0 = hw::ConstantOp::create(builder, loc, ptrType, 0);
  Value ptrTc1 = hw::ConstantOp::create(builder, loc, ptrType, 1);
  Value ptrTcFull = hw::ConstantOp::create(builder, loc, ptrType, depth);

  // ====== Hardware units ======
  Value count = seq::CompRegOp::create(
      builder, loc, nextCount, clk, rst,
      hw::ConstantOp::create(builder, loc, countType, 0), "fifo_count");
  seq::HLMemOp hlmem = seq::HLMemOp::create(
      builder, loc, clk, rst, "fifo_mem",
      llvm::SmallVector<int64_t>{static_cast<int64_t>(depth)}, eltType);
  Value rdAddr = seq::CompRegOp::create(
      builder, loc, rdAddrNext, clk, rst,
      hw::ConstantOp::create(builder, loc, ptrType, 0), "fifo_rd_addr");
  Value wrAddr = seq::CompRegOp::create(
      builder, loc, wrAddrNext, clk, rst,
      hw::ConstantOp::create(builder, loc, ptrType, 0), "fifo_wr_addr");

  Value readData = seq::ReadPortOp::create(builder, loc, hlmem,
                                           llvm::SmallVector<Value>{rdAddr},
                                           rdEn, cfg.rdLatency);
  seq::WritePortOp::create(builder, loc, hlmem,
                           llvm::SmallVector<Value>{wrAddr}, input, wrEn,
                           /*latency*/ 1);

  // ====== some more constants =====
  comb::ICmpOp fifoFull = comb::ICmpOp::create(
      builder, loc, comb::ICmpPredicate::eq, count, countTcFull);
  fifoFull->setAttr("sv.namehint", builder.getStringAttr("fifo_full"));
  comb::ICmpOp fifoEmpty = comb::ICmpOp::create(
      builder, loc, comb::ICmpPredicate::eq, count, countTc0);
  fifoEmpty->setAttr("sv.namehint", builder.getStringAttr("fifo_empty"));

  // ====== Next-state count ======
  auto notRdEn = comb::createOrFoldNot(builder, loc, rdEn);
  auto notWrEn = comb::createOrFoldNot(builder, loc, wrEn);
  Value rdEnNandWrEn = comb::AndOp::create(builder, loc, notRdEn, notWrEn);
  Value rdEnAndNotWrEn = comb::AndOp::create(builder, loc, rdEn, notWrEn);
  Value wrEnAndNotRdEn = comb::AndOp::create(builder, loc, wrEn, notRdEn);

  auto countEqTcFull = comb::ICmpOp::create(
      builder, loc, comb::ICmpPredicate::eq, count, countTcFull);
  auto addCountTc1 = comb::AddOp::create(builder, loc, count, countTc1);
  Value wrEnNext = comb::MuxOp::create(builder, loc, countEqTcFull,
                                       // keep value
                                       count,
                                       // increment
                                       addCountTc1);
  auto countEqTc0 = comb::ICmpOp::create(builder, loc, comb::ICmpPredicate::eq,
                                         count, countTc0);
  auto subCountTc1 = comb::SubOp::create(builder, loc, count, countTc1);

  Value rdEnNext = comb::MuxOp::create(builder, loc, countEqTc0,
                                       // keep value
                                       count,
                                       // decrement
                                       subCountTc1);

  auto nextInnerMux =
      comb::MuxOp::create(builder, loc, rdEnAndNotWrEn, rdEnNext, count);
  auto nextMux =
      comb::MuxOp::create(builder, loc, wrEnAndNotRdEn, wrEnNext, nextInnerMux);
  nextCount.setValue(comb::MuxOp::create(builder, loc, rdEnNandWrEn,
                                         /*keep value*/ count, nextMux));
  static_cast<Value>(nextCount).getDefiningOp()->setAttr(
      "sv.namehint", builder.getStringAttr("fifo_count_next"));

  // ====== Read/write pointers ======
  Value wrAndNotFull = comb::AndOp::create(
      builder, loc, wrEn, comb::createOrFoldNot(builder, loc, fifoFull));
  auto addWrAddrPtrTc1 = comb::AddOp::create(builder, loc, wrAddr, ptrTc1);

  auto wrAddrNextNoRollover =
      comb::MuxOp::create(builder, loc, wrAndNotFull, addWrAddrPtrTc1, wrAddr);
  auto isMaxAddrWr = comb::ICmpOp::create(builder, loc, comb::ICmpPredicate::eq,
                                          wrAddrNextNoRollover, ptrTcFull);
  wrAddrNext.setValue(comb::MuxOp::create(builder, loc, isMaxAddrWr, ptrTc0,
                                          wrAddrNextNoRollover));
  static_cast<Value>(wrAddrNext)
      .getDefiningOp()
      ->setAttr("sv.namehint", builder.getStringAttr("fifo_wr_addr_next"));

  auto notFifoEmpty = comb::createOrFoldNot(builder, loc, fifoEmpty);
  Value rdAndNotEmpty = comb::AndOp::create(builder, loc, rdEn, notFifoEmpty);
  auto addRdAddrPtrTc1 = comb::AddOp::create(builder, loc, rdAddr, ptrTc1);
  auto rdAddrNextNoRollover =
      comb::MuxOp::create(builder, loc, rdAndNotEmpty, addRdAddrPtrTc1, rdAddr);
  auto isMaxAddrRd = comb::ICmpOp::create(builder, loc, comb::ICmpPredicate::eq,
                                          rdAddrNextNoRollover, ptrTcFull);
  rdAddrNext.setValue(comb::MuxOp::create(builder, loc, isMaxAddrRd, ptrTc0,
                                          rdAddrNextNoRollover));
  static_cast<Value>(rdAddrNext)
      .getDefiningOp()
      ->setAttr("sv.namehint", builder.getStringAttr("fifo_rd_addr_next"));

  // ====== Result values ======
  llvm::SmallVector<Value> results;

  // Data
  results.push_back(readData);
  // Full
  results.push_back(fifoFull);
  // Empty
  results.push_back(fifoEmpty);

  if (auto almostFull = cfg.almostFull) {
    results.push_back(comb::ICmpOp::create(
        builder, loc, comb::ICmpPredicate::uge, count,
        hw::ConstantOp::create(builder, loc, countType, almostFull.value())));
    static_cast<Value>(results.back())
        .getDefiningOp()
        ->setAttr("sv.namehint", builder.getStringAttr("fifo_almost_full"));
  }

  if (auto almostEmpty = cfg.almostEmpty) {
    results.push_back(comb::ICmpOp::create(
        builder, loc, comb::ICmpPredicate::ule, count,
        hw::ConstantOp::create(builder, loc, countType, almostEmpty.value())));
    static_cast<Value>(results.back())
        .getDefiningOp()
        ->setAttr("sv.namehint", builder.getStringAttr("fifo_almost_empty"));
  }

  // ====== Protocol checks =====
  Value clkI1 = seq::FromClockOp::create(builder, loc, clk);
  // Disable the protocol checks during reset: the count register (and thus
  // the empty/full signals) is undefined until the first reset clock edge, so
  // an ungated assertion would observe X and fail in 4-state simulators.
  Value notRst = comb::createOrFoldNot(builder, loc, rst);
  Value notEmptyAndRden = comb::createOrFoldNot(
      builder, loc, comb::AndOp::create(builder, loc, rdEn, fifoEmpty));
  verif::ClockedAssertOp::create(
      builder, loc, notEmptyAndRden, verif::ClockEdge::Pos, clkI1,
      /*enable=*/notRst, builder.getStringAttr("FIFO empty when read enabled"));
  Value notFullAndWren = comb::createOrFoldNot(
      builder, loc, comb::AndOp::create(builder, loc, wrEn, fifoFull));
  verif::ClockedAssertOp::create(
      builder, loc, notFullAndWren, verif::ClockEdge::Pos, clkI1,
      /*enable=*/notRst, builder.getStringAttr("FIFO full when write enabled"));

  return results;
}

/// Build the `FIFOConfig` describing a `seq.fifo` op.
static FIFOConfig getFIFOConfig(seq::FIFOOp mem) {
  return FIFOConfig{mem.getInput().getType(), mem.getDepth(),
                    mem.getRdLatency(), mem.getAlmostFullThreshold(),
                    mem.getAlmostEmptyThreshold()};
}

struct FIFOLowering : public OpConversionPattern<seq::FIFOOp> {
public:
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(seq::FIFOOp mem, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const final {
    FIFOConfig cfg{adaptor.getInput().getType(), mem.getDepth(),
                   mem.getRdLatency(), mem.getAlmostFullThreshold(),
                   mem.getAlmostEmptyThreshold()};
    // Construct the backedge builder from the `PatternRewriter` so that the
    // temporary backedge placeholders are erased through the conversion driver
    // (a raw `OpBuilder` would erase them out from under it and crash).
    BackedgeBuilder bb(rewriter, mem.getLoc());
    llvm::SmallVector<Value> results = buildFIFOLogic(
        rewriter, bb, mem.getLoc(), adaptor.getClk(), adaptor.getRst(),
        adaptor.getInput(), adaptor.getRdEn(), adaptor.getWrEn(), cfg);
    rewriter.replaceOp(mem, results);
    return success();
  }
};

struct LowerSeqFIFOPass
    : public circt::seq::impl::LowerSeqFIFOBase<LowerSeqFIFOPass> {
  using circt::seq::impl::LowerSeqFIFOBase<LowerSeqFIFOPass>::LowerSeqFIFOBase;
  void runOnOperation() override;

private:
  /// Lower all `seq.fifo` ops inline within their parent module.
  LogicalResult runInline();
  /// Lower each unique `seq.fifo` configuration into its own `hw.module` and
  /// replace each `seq.fifo` with an instance of the matching module.
  LogicalResult runOutline();
  /// Create (and cache) the `hw.module` implementing `cfg`.
  hw::HWModuleOp getOrCreateFIFOModule(const FIFOConfig &cfg, Location loc);

  mlir::ModuleOp topModule;
  DenseMap<Attribute, hw::HWModuleOp> moduleCache;
  llvm::StringSet<> usedNames;
};

} // namespace

/// Build an attribute uniquely identifying a FIFO configuration, suitable for
/// use as a cache key.
static Attribute makeConfigKey(OpBuilder &builder, const FIFOConfig &cfg) {
  Attribute none = builder.getUnitAttr();
  return builder.getArrayAttr(
      {TypeAttr::get(cfg.eltType), builder.getI64IntegerAttr(cfg.depth),
       builder.getI64IntegerAttr(cfg.rdLatency),
       cfg.almostFull ? Attribute(builder.getI64IntegerAttr(*cfg.almostFull))
                      : none,
       cfg.almostEmpty ? Attribute(builder.getI64IntegerAttr(*cfg.almostEmpty))
                       : none});
}

/// Generate a deterministic, human-readable module name from a configuration.
static std::string makeModuleName(const FIFOConfig &cfg) {
  std::string name = "seq_fifo";
  // Encode the element width so configs with different element types get
  // distinct, self-describing names. Aggregate/opaque types may report a
  // negative width; fall back to omitting the width in that case (the uniquing
  // suffix still guarantees a unique symbol).
  int64_t width = hw::getBitWidth(cfg.eltType);
  if (width >= 0)
    name += "_w" + llvm::utostr(width);
  name += "_d" + llvm::utostr(cfg.depth);
  if (cfg.rdLatency)
    name += "_rl" + llvm::utostr(cfg.rdLatency);
  if (cfg.almostFull)
    name += "_af" + llvm::utostr(*cfg.almostFull);
  if (cfg.almostEmpty)
    name += "_ae" + llvm::utostr(*cfg.almostEmpty);
  return name;
}

hw::HWModuleOp LowerSeqFIFOPass::getOrCreateFIFOModule(const FIFOConfig &cfg,
                                                       Location loc) {
  OpBuilder builder(topModule.getContext());
  Attribute key = makeConfigKey(builder, cfg);
  if (hw::HWModuleOp cached = moduleCache.lookup(key))
    return cached;

  Type i1 = builder.getI1Type();
  Type clkType = seq::ClockType::get(builder.getContext());

  size_t argn = 0, resn = 0;
  auto port = [&](StringRef name, Type type, hw::ModulePort::Direction dir) {
    size_t &idx = dir == hw::ModulePort::Direction::Output ? resn : argn;
    return hw::PortInfo{{builder.getStringAttr(name), type, dir}, idx++};
  };
  llvm::SmallVector<hw::PortInfo> ports = {
      port("clk", clkType, hw::ModulePort::Direction::Input),
      port("rst", i1, hw::ModulePort::Direction::Input),
      port("in", cfg.eltType, hw::ModulePort::Direction::Input),
      port("rdEn", i1, hw::ModulePort::Direction::Input),
      port("wrEn", i1, hw::ModulePort::Direction::Input),
      port("out", cfg.eltType, hw::ModulePort::Direction::Output),
      port("full", i1, hw::ModulePort::Direction::Output),
      port("empty", i1, hw::ModulePort::Direction::Output)};
  if (cfg.almostFull)
    ports.push_back(port("almost_full", i1, hw::ModulePort::Direction::Output));
  if (cfg.almostEmpty)
    ports.push_back(
        port("almost_empty", i1, hw::ModulePort::Direction::Output));

  // Generate a unique symbol name.
  std::string baseName = makeModuleName(cfg);
  std::string name = baseName;
  for (unsigned i = 0; !usedNames.insert(name).second; ++i)
    name = baseName + "_" + llvm::utostr(i);

  builder.setInsertionPointToEnd(topModule.getBody());
  auto mod =
      hw::HWModuleOp::create(builder, loc, builder.getStringAttr(name), ports);

  // Build the module body before the auto-created `hw.output` terminator.
  Block *body = mod.getBodyBlock();
  auto args = body->getArguments();
  OpBuilder bodyBuilder(body->getTerminator());
  BackedgeBuilder bb(bodyBuilder, loc);
  llvm::SmallVector<Value> results = buildFIFOLogic(
      bodyBuilder, bb, loc, /*clk=*/args[0], /*rst=*/args[1], /*input=*/args[2],
      /*rdEn=*/args[3], /*wrEn=*/args[4], cfg);
  body->getTerminator()->setOperands(results);

  moduleCache[key] = mod;
  return mod;
}

LogicalResult LowerSeqFIFOPass::runOutline() {
  llvm::SmallVector<seq::FIFOOp> fifos;
  topModule.walk([&](seq::FIFOOp op) { fifos.push_back(op); });

  // Seed the used-name set with the existing top-level symbols to avoid
  // colliding with user modules.
  for (auto &op : *topModule.getBody())
    if (auto sym =
            op.getAttrOfType<StringAttr>(SymbolTable::getSymbolAttrName()))
      usedNames.insert(sym.getValue());

  unsigned instCounter = 0;
  for (seq::FIFOOp fifo : fifos) {
    FIFOConfig cfg = getFIFOConfig(fifo);
    hw::HWModuleOp mod = getOrCreateFIFOModule(cfg, fifo.getLoc());

    OpBuilder builder(fifo);
    llvm::SmallVector<Value> operands = {fifo.getClk(), fifo.getRst(),
                                         fifo.getInput(), fifo.getRdEn(),
                                         fifo.getWrEn()};
    auto inst = hw::InstanceOp::create(
        builder, fifo.getLoc(), mod,
        builder.getStringAttr("fifo_inst_" + llvm::utostr(instCounter++)),
        operands);
    fifo.replaceAllUsesWith(inst.getResults());
    fifo.erase();
  }
  return success();
}

LogicalResult LowerSeqFIFOPass::runInline() {
  MLIRContext &ctxt = getContext();
  ConversionTarget target(ctxt);

  // Lowering patterns must lower away all HLMem-related operations.
  target.addIllegalOp<seq::FIFOOp>();
  target.addLegalDialect<seq::SeqDialect, hw::HWDialect, comb::CombDialect,
                         verif::VerifDialect>();
  RewritePatternSet patterns(&ctxt);
  patterns.add<FIFOLowering>(&ctxt);

  return applyPartialConversion(getOperation(), target, std::move(patterns));
}

void LowerSeqFIFOPass::runOnOperation() {
  topModule = getOperation();
  if (failed(outlineModules ? runOutline() : runInline()))
    signalPassFailure();
}
