//===- UnrollMemoryLatencies.cpp - seq.shiftreg lowering ------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "circt/Dialect/Comb/CombOps.h"
#include "circt/Dialect/SV/SVOps.h"
#include "circt/Dialect/Seq/SeqOps.h"
#include "circt/Dialect/Seq/SeqPasses.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include "llvm/ADT/TypeSwitch.h"

namespace circt {
namespace seq {
#define GEN_PASS_DEF_UNROLLMEMORYLATENCIES
#include "circt/Dialect/Seq/SeqPasses.h.inc"
} // namespace seq
} // namespace circt

using namespace circt;
using namespace seq;

namespace {
struct UnrollMemoryLatenciesPass
    : public circt::seq::impl::UnrollMemoryLatenciesBase<
          UnrollMemoryLatenciesPass> {
  void runOnOperation() override;

private:
  /// Delays all of the addresses in the given value range using the given delay
  /// and stores the delayed addresses in the given vector.
  void delayAddresses(OpBuilder &builder, Location loc, Value clk, Value ce,
                      MutableOperandRange addrs, IntegerAttr delayAttr) const {

    builder.setInsertionPoint(addrs.getOwner());

    // My understanding is that when multiple addresses are given, the same
    // data is written to each location. Therefore we must delay all of the
    // addresses individually.
    for (auto &addr : addrs)
      // delay the addr
      addr.set(seq::ShiftRegOp::create(builder, loc, delayAttr, addr.get(), clk,
                                       ce, {}, {}, {}, {}, {}));
  }

  /// Unrolling for the HLMem write ports. This reduces the latency attribute
  /// of the port to 1 and implements that latency as a delay of n-1 on its
  /// data and address inputs.
  void hlMemOpWritePortUnrolling(seq::WritePortOp &op,
                                 OpBuilder &builder) const {

    // Compute the new latency attributes
    IntegerAttr latencyAttr = op.getLatencyAttr();
    IntegerAttr delayAttr =
        IntegerAttr::get(latencyAttr.getType(), latencyAttr.getInt() - 1);
    IntegerAttr oneAttr = IntegerAttr::get(latencyAttr.getType(), 1);

    // Retrieve all useful arguments
    OpOperand &data = op.getInDataMutable();
    OpOperand &wrEn = op.getWrEnMutable();
    MutableOperandRange addrs = op.getAddressesMutable();
    Value clk = op.getMemory().getDefiningOp()->getOperand(0);

    builder.setInsertionPointAfter(op);

    // A global clock enable for all ports that is always true
    Value ce = builder.createOrFold<hw::ConstantOp>(op.getLoc(),
                                                    builder.getI1Type(), 1);

    // Delay all of the addresses
    delayAddresses(builder, op.getLoc(), clk, ce, addrs, delayAttr);

    // delay the data
    data.set(seq::ShiftRegOp::create(builder, op.getLoc(), delayAttr,
                                     data.get(), clk, ce, {}, {}, {}, {}, {}));

    // delay the write enable
    wrEn.set(seq::ShiftRegOp::create(builder, op.getLoc(), delayAttr,
                                     wrEn.get(), clk, ce, {}, {}, {}, {}, {}));

    // Update the latency
    op.setLatencyAttr(oneAttr);
  }

  /// Unrolling for the HLMem Read ports. This reduces the latency attribute
  /// of the port to 0 and implements that latency as a delay of n, modeled
  /// as a shift-register, on its address input.
  void hlMemOpReadPortUnrolling(seq::ReadPortOp &op, OpBuilder &builder) const {
    // Compute the new latency attributes
    IntegerAttr latencyAttr = op.getLatencyAttr();
    IntegerAttr zeroAttr = IntegerAttr::get(latencyAttr.getType(), 0);

    // Retrieve all useful arguments
    MutableOperandRange addrs = op.getAddressesMutable();
    MutableOperandRange rdEn = op.getRdEnMutable();
    Value clk = op.getMemory().getDefiningOp()->getOperand(0);

    builder.setInsertionPointAfter(op);

    // A global clock enable for all ports that is always true
    Value ce = builder.createOrFold<hw::ConstantOp>(op.getLoc(),
                                                    builder.getI1Type(), 1);

    // Same as write port, delay all of the addresses
    delayAddresses(builder, op.getLoc(), clk, ce, addrs, latencyAttr);

    // delay the read enable
    for (auto &rden : rdEn)
      rden.set(seq::ShiftRegOp::create(builder, op.getLoc(), latencyAttr,
                                       rden.get(), clk, ce, {}, {}, {}, {},
                                       {}));

    op.setLatencyAttr(zeroAttr);
  }

  /// Unrolling for FirMem latencies. Unlike HLMem, FirMem stores
  /// latencies it the memory declaration op rather than the ports themselves.
  /// This makes the unrolling a little more annoying, as we need to do fetch
  /// all of the ports from the memory itself, i.e. walk its users to find the
  /// all of the relevant ports to delay before updating the memory declaration
  /// to have its new delays. The actual port updates are identical to those
  /// done in the HLMem ports.
  void unrollFirMemWritePort(seq::FirMemWriteOp &op, IntegerAttr delayAttr,
                             Value ce, OpBuilder &builder) const {

    // Retrieve all of the relevant arguments
    OpOperand &addr = op.getAddressMutable();
    OpOperand &data = op.getDataMutable();
    MutableOperandRange ens = op.getEnableMutable();
    Value clk = op.getClk();

    // Delay the address, data, and enable inputs
    data.set(seq::ShiftRegOp::create(builder, op.getLoc(), delayAttr,
                                     data.get(), clk, ce, {}, {}, {}, {}, {}));

    addr.set(seq::ShiftRegOp::create(builder, op.getLoc(), delayAttr,
                                     addr.get(), clk, ce, {}, {}, {}, {}, {}));

    for (auto &en : ens)
      en.set(seq::ShiftRegOp::create(builder, op.getLoc(), delayAttr, en.get(),
                                     clk, ce, {}, {}, {}, {}, {}));
  }

  /// Performs the latency unrolling of an individual read port
  void unrollFirMemReadPort(seq::FirMemReadOp &op, IntegerAttr delayRdAttr,
                            Value ce, OpBuilder &builder) const {
    // Retrieve all of the relevant arguments
    OpOperand &addr = op.getAddressMutable();
    MutableOperandRange ens = op.getEnableMutable();
    Value clk = op.getClk();

    addr.set(seq::ShiftRegOp::create(builder, op.getLoc(), delayRdAttr,
                                     addr.get(), clk, ce, {}, {}, {}, {}, {}));

    for (auto &en : ens)
      en.set(seq::ShiftRegOp::create(builder, op.getLoc(), delayRdAttr,
                                     en.get(), clk, ce, {}, {}, {}, {}, {}));
  }

  /// Performs the latency unrolling of a read-write port.
  /// These are annoying as they require implementing both the read and write
  /// unrolling and then muxing the two delays using the mode as a selector.
  void unrollFirMemReadWritePort(seq::FirMemReadWriteOp &op,
                                 IntegerAttr delayWrAttr,
                                 IntegerAttr delayRdAttr, Value ce,
                                 OpBuilder &builder) const {
    // Retrieve all of the relevant arguments
    OpOperand &addr = op.getAddressMutable();
    OpOperand &data = op.getWriteDataMutable();
    OpOperand &mode = op.getModeMutable();
    MutableOperandRange ens = op.getEnableMutable();
    Value clk = op.getClk();

    // Create two versions of the delayed address, one using the write
    // latency, one using the read latency
    auto delayedReadAddr =
        seq::ShiftRegOp::create(builder, op.getLoc(), delayRdAttr, addr.get(),
                                clk, ce, {}, {}, {}, {}, {});

    auto delayedWriteAddr =
        seq::ShiftRegOp::create(builder, op.getLoc(), delayWrAttr, addr.get(),
                                clk, ce, {}, {}, {}, {}, {});

    // Delay the write data, this will only be used in case of write so no
    // need for a second mux.
    data.set(seq::ShiftRegOp::create(builder, op.getLoc(), delayWrAttr,
                                     data.get(), clk, ce, {}, {}, {}, {}, {}));

    // Delay mode to match read and write inputs
    // Select the delay sucht that mode = mux(
    //  mode.dealy(min(rd_lat, wr_lat)),
    //  mode.delay(wr_lat), mode.delay(rd_lat)
    // )
    // This makes sure that we always select the correct address using the
    // relevant mode

    auto delayedModeWr =
        seq::ShiftRegOp::create(builder, op.getLoc(), delayWrAttr, mode.get(),
                                clk, ce, {}, {}, {}, {}, {});

    auto delayedModeRd =
        seq::ShiftRegOp::create(builder, op.getLoc(), delayRdAttr, mode.get(),
                                clk, ce, {}, {}, {}, {}, {});

    // We reuse the smallest delayed mode to avoid creating a duplicate
    // shiftreg
    auto selectMode = comb::MuxOp::create(
        builder, op.getLoc(),
        delayWrAttr.getInt() < delayRdAttr.getInt() ? delayedModeWr
                                                    : delayedModeRd,
        delayedModeWr, delayedModeRd);
    mode.set(selectMode);

    // Select which delayed version we want based on the mode
    addr.set(comb::MuxOp::create(builder, op.getLoc(), selectMode,
                                 delayedWriteAddr, delayedReadAddr));

    // Finally also delay the enable signal
    for (auto &en : ens)
      en.set(seq::ShiftRegOp::create(builder, op.getLoc(), delayWrAttr,
                                     en.get(), clk, ce, {}, {}, {}, {}, {}));
  }

  void firMemOpUnrolling(seq::FirMemOp &mem, OpBuilder &builder) const {

    // Retrieve read and write latencies
    IntegerAttr wrLatency = mem.getWriteLatencyAttr();
    IntegerAttr rdLatency = mem.getReadLatencyAttr();
    IntegerAttr delayRdAttr = builder.getI64IntegerAttr(rdLatency.getInt());
    IntegerAttr delayWrAttr = builder.getI64IntegerAttr(wrLatency.getInt() - 1);
    IntegerAttr zeroAttr = builder.getI32IntegerAttr(0);
    IntegerAttr oneAttr = builder.getI32IntegerAttr(1);

    builder.setInsertionPointAfter(mem);

    // A global clock enable for all ports that is always true
    Value ce = builder.createOrFold<hw::ConstantOp>(mem.getLoc(),
                                                    builder.getI1Type(), 1);

    // Find all of the relevant ports and update them
    for (auto *user : mem->getUsers()) {
      // Figure out what type of port it is an unroll it
      TypeSwitch<Operation *, void>(user)
          .Case<seq::FirMemWriteOp>([&](auto port) {
            unrollFirMemWritePort(port, delayWrAttr, ce, builder);
          })
          .Case<seq::FirMemReadOp>([&](auto port) {
            unrollFirMemReadPort(port, delayRdAttr, ce, builder);
          })
          .Case<seq::FirMemReadWriteOp>([&](auto port) {
            unrollFirMemReadWritePort(port, delayWrAttr, delayRdAttr, ce,
                                      builder);
          })
          .Default([](auto) { /*ignore non-port users*/ });
    }

    // Update the memory's latency attributes
    mem.setReadLatencyAttr(zeroAttr);
    mem.setWriteLatencyAttr(oneAttr);
  }
};

} // namespace

void UnrollMemoryLatenciesPass::runOnOperation() {
  auto top = getOperation();
  OpBuilder builder(top);

  // Match over all hlmemport ops and firmem ops
  top->walk([&](Operation *op) {
    TypeSwitch<Operation *>(op)
        .Case<seq::WritePortOp>(
            [&](auto &wrop) { hlMemOpWritePortUnrolling(wrop, builder); })
        .Case<seq::ReadPortOp>(
            [&](auto &rdop) { hlMemOpReadPortUnrolling(rdop, builder); })
        .Case<seq::FirMemOp>(
            [&](auto &firmem) { firMemOpUnrolling(firmem, builder); })
        .Default([](auto) {});
  });
}
