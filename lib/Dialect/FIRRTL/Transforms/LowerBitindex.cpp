//===- LowerBitindex.cpp - Lower Bitindex
//-----------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines the LowerBitindex pass.
//
//===----------------------------------------------------------------------===//

#include "PassDetails.h"
#include "circt/Dialect/FIRRTL/FIRRTLOps.h"
#include "circt/Dialect/FIRRTL/FIRRTLUtils.h"
#include "circt/Dialect/FIRRTL/Passes.h"
#include "mlir/IR/ImplicitLocOpBuilder.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/Parallel.h"

#define DEBUG_TYPE "firrtl-lower-bitindex"

using namespace circt;
using namespace firrtl;

namespace {
struct LowerBitIndexPass : public LowerBitindexBase<LowerBitIndexPass> {
  void runOnOperation() override;
};

void LowerBitIndexPass::runOnOperation() {
  LLVM_DEBUG(
      llvm::dbgs() << "===- Running LowerBitIndex Pass "
                      "------------------------------------------------===\n");

  DenseSet<Value> variables;

  // collect all values that are used as input to a bitindex
  for (auto bitindex :
       llvm::make_early_inc_range(getOperation().getOps<BitindexOp>())) {
    if (bitindex->getUses().empty()) {
      bitindex.erase();
      continue;
    }
    variables.insert(bitindex.input());
  }

  // make a new wire for each bitindexed value
  for (auto var : variables) {
    auto *defn = var.getDefiningOp();
    llvm::StringRef name;
    ImplicitLocOpBuilder builder(var.getLoc(), var.getContext());
    if (!defn) {
      auto mod = getOperation();
      name = "port.bitindex.wrapper";
      builder.setInsertionPointToStart(mod.getBody());
      if (auto blockArg = var.dyn_cast<BlockArgument>()) {
        name = mod.getPortName(blockArg.getArgNumber());
      }
    } else {
      name = "local.bitindex.wrapper";
      builder.setInsertionPointAfter(defn);
      // improve the name if it was defined as a wire or reg
      if (auto wireOp = dyn_cast<WireOp>(defn)) {
        name = wireOp.name();
      } else if (auto regOp = dyn_cast<RegOp>(defn)) {
        name = regOp.name();
      }
    }
    // must be an int type to be the input to a bitindex
    if (auto i = var.getType().dyn_cast<IntType>()) {
      // must have defined width
      if (!i.hasWidth()) {
        signalPassFailure();
        return;
      }
      auto w = i.getWidth().getValue();
      // wrapper wire that is a UInt<1>[var.width]
      auto wire = builder.create<WireOp>(
          FVectorType::get(UIntType::get(var.getContext(), 1), w), name);

      // for every use of var replace any connection to it with a connection
      // that assigns each individual bit of the src to the dest
      for (auto *op : var.getUsers()) {
        // transforms `var <= src` to `wire[i] <= bits(src, i, i)` 0 <= i < w
        if (auto connect = dyn_cast<StrictConnectOp>(op)) {
          if (var == connect.dest()) {
            ImplicitLocOpBuilder builder(connect.getLoc(), connect);
            for (int i = 0; i < w; i++) {
              Value bitsOp = builder.create<BitsPrimOp>(connect.src(), i, i);
              Value subidxOp = builder.create<SubindexOp>(wire, i);
              builder.create<StrictConnectOp>(subidxOp, bitsOp);
            }
            connect.erase();
          }
        }
      }

      // constructs the full concatenation of wire and assigns it to var:
      // var <= cat(..., cat(wire[2], cat(wire[1], wire[0])))
      Value prev = builder.create<SubindexOp>(wire, 0);
      for (int i = 1; i < w; i++) {
        Value subidx = builder.create<SubindexOp>(wire, i);
        Value cat = builder.create<CatPrimOp>(subidx, prev);
        prev = cat;
      }
      if (i.isa<SIntType>()) {
        prev = builder.create<AsSIntPrimOp>(prev);
      }
      // connect here after replacing all other connects
      builder.create<StrictConnectOp>(var, prev);

      for (auto *op : var.getUsers()) {
        if (auto bitindex = dyn_cast<BitindexOp>(op)) {
          // replaces any occurrence of `var[n] <= ...` with `wire[n] <= ...`
          ImplicitLocOpBuilder builder(bitindex.getLoc(), bitindex);
          Value subidx = builder.create<SubindexOp>(wire, bitindex.index());
          bitindex.replaceAllUsesWith(subidx);
          bitindex.erase();
        } else if (auto bits = dyn_cast<BitsPrimOp>(op)) {
          // replaces any occurrence of `bits(var, hi, lo)` with
          // `cat(wire[hi], cat(..., wire[lo]))`
          ImplicitLocOpBuilder builder(bits.getLoc(), bits);
          Value prev = builder.create<SubindexOp>(wire, bits.lo());
          for (unsigned i = bits.lo() + 1; i < bits.hi(); i++) {
            Value subidx = builder.create<SubindexOp>(wire, i);
            Value cat = builder.create<CatPrimOp>(subidx, prev);
            prev = cat;
          }
          bits.replaceAllUsesWith(prev);
          bits.erase();
        }
      }
      // if the bit-indexed value is a register then we need to wire it up to
      // preserve its value by doing `wire[i] <= bits(var, i, i)`
      if (isa_and_nonnull<RegOp>(defn) || isa_and_nonnull<RegResetOp>(defn)) {
        ImplicitLocOpBuilder builder(defn->getLoc(), defn);
        builder.setInsertionPointAfter(wire);
        for (int i = 0; i < w; i++) {
          Value bitsOp = builder.create<BitsPrimOp>(var, i, i);
          Value subidxOp = builder.create<SubindexOp>(wire, i);
          builder.create<StrictConnectOp>(subidxOp, bitsOp);
        }
      }
    }
  }
}

} // end anonymous namespace

std::unique_ptr<mlir::Pass> circt::firrtl::createLowerBitindexPass() {
  return std::make_unique<LowerBitIndexPass>();
}
