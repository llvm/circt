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
// The lowering algorithm looks for all values that are bit-indexed. For each
// such value `val`, it creates a new wire `v_wire` that is a vector
// `UInt<1>[val.width]`.
// Then it makes the following transformations:
//
// * `val <= e` becomes `v_wire[i] <= bits(e, i, i)` for all `i` up to `val.width`
// * `val[n] <= e` becomes `v_wire[n] <= e`.
// * `bits(val, hi, lo)` becomes `cat(v_wire[hi], v_wire[hi-1], ..., v_wire[lo])`.
//
// Then it connects the concatenation of the vector v_wire to the variable (`val <=
// cat(v_wire[n], v_wire[n-1], ..., v_wire[0])` where `n` is `val.width-1`).
//
// If the bit-indexed value is a register, the pass additionally creates
// state-preserving connections `v_wire[i] <= bits(val, i, i)` just after the
// definition of `v_wire`.
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

  DenseSet<Value> values;

  // collect all values that are used as input to a bitindex
  for (auto bitindex :
       llvm::make_early_inc_range(getOperation().getOps<BitindexOp>())) {
    if (bitindex->getUses().empty()) {
      bitindex.erase();
      continue;
    }
    values.insert(bitindex.input());
  }

  // make a new wire for each bitindexed value
  for (auto val : values) {
    auto *defn = val.getDefiningOp();
    llvm::StringRef name;
    ImplicitLocOpBuilder builder(val.getLoc(), val.getContext());
    if (auto blockArg = val.dyn_cast<BlockArgument>()) {
      auto mod = getOperation();
      builder.setInsertionPointToStart(mod.getBody());
      name = mod.getPortName(blockArg.getArgNumber());
    } else {
      builder.setInsertionPointAfter(defn);
      // improve the name if it was defined as a wire or reg
      if (auto wireOp = dyn_cast<WireOp>(defn)) {
        name = wireOp.name();
      } else if (auto regOp = dyn_cast<RegOp>(defn)) {
        name = regOp.name();
      } else if (auto regResetOp = dyn_cast<RegResetOp>(defn)) {
        name = regResetOp.name();
      }
    }

    // must be an int type to be the input to a bitindex
    if (auto i = val.getType().dyn_cast<IntType>()) {
      // must have defined width
      if (!i.hasWidth()) {
        signalPassFailure();
        return;
      }
      auto w = i.getWidth().getValue();
      // wrapper wire that is a UInt<1>[val.width]
      auto wire = builder.create<WireOp>(
          FVectorType::get(UIntType::get(val.getContext(), 1), w), name);

      // for every use of val replace any connection to it with a connection
      // that assigns each individual bit of the src to the dest
      for (auto *op : val.getUsers()) {
        // transforms `val <= src` to `wire[i] <= bits(src, i, i)` 0 <= i < w
        if (auto connect = dyn_cast<StrictConnectOp>(op)) {
          if (val == connect.dest()) {
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

      // constructs the full concatenation of wire and connects it to val:
      // val <= cat(..., cat(wire[2], cat(wire[1], wire[0])))
      Value prev = builder.create<SubindexOp>(wire, 0);
      for (int i = 1; i < w; i++) {
        Value subidx = builder.create<SubindexOp>(wire, i);
        Value cat = builder.create<CatPrimOp>(subidx, prev);
        prev = cat;
      }
      // cast back to SInt if necessary to connect to val
      if (i.isa<SIntType>()) {
        prev = builder.create<AsSIntPrimOp>(prev);
      }
      // connect here after replacing all other connects
      builder.create<StrictConnectOp>(val, prev);

      for (auto *op : val.getUsers()) {
        if (auto bitindex = dyn_cast<BitindexOp>(op)) {
          // replaces any occurrence of `val[n] <= ...` with `wire[n] <= ...`
          ImplicitLocOpBuilder builder(bitindex.getLoc(), bitindex);
          Value subidx = builder.create<SubindexOp>(wire, bitindex.index());
          bitindex.replaceAllUsesWith(subidx);
          bitindex.erase();
        } else if (auto bits = dyn_cast<BitsPrimOp>(op)) {
          // replaces any occurrence of `bits(val, hi, lo)` with
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
      // preserve its value by doing `wire[i] <= bits(val, i, i)`
      if (isa_and_nonnull<RegOp>(defn) || isa_and_nonnull<RegResetOp>(defn)) {
        ImplicitLocOpBuilder builder(defn->getLoc(), defn);
        // this must be done immediately after the wrapper wire declaration to
        // properly follow last-connect semantics
        builder.setInsertionPointAfter(wire);
        for (int i = 0; i < w; i++) {
          Value bitsOp = builder.create<BitsPrimOp>(val, i, i);
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
