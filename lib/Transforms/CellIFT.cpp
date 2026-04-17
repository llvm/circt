//===- CellIFT.cpp - Cell-level Information Flow Tracking -------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This pass instruments HW/Comb/Seq IR with parallel taint-tracking logic
// using the CellIFT methodology (cell-level dynamic information flow tracking).
//
// For every data value v : iN, the pass creates a parallel taint value
// v_t : iN. Module signatures are extended with taint ports, instances are
// updated accordingly, and combinational/sequential operations are instrumented
// with taint propagation rules that operate at the macrocell level.
//
// References:
//   F. Music, F. K. Gürkaynak, et al., "CellIFT: Leveraging Cells for
//   Scalable and Precise Dynamic Information Flow Tracking in RTL,"
//   USENIX Security 2022.
//
//===----------------------------------------------------------------------===//

#include "circt/Dialect/Comb/CombOps.h"
#include "circt/Dialect/HW/HWInstanceGraph.h"
#include "circt/Dialect/HW/HWOps.h"
#include "circt/Dialect/HW/HWTypes.h"
#include "circt/Dialect/Seq/SeqOps.h"
#include "circt/Transforms/Passes.h"
#include "mlir/Analysis/TopologicalSortUtils.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/ImplicitLocOpBuilder.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "cellift-instrument"

namespace circt {
#define GEN_PASS_DEF_CELLIFTINSTRUMENT
#include "circt/Transforms/Passes.h.inc"
} // namespace circt

using namespace mlir;
using namespace circt;
using namespace circt::hw;
using namespace circt::comb;
using namespace circt::seq;

//===----------------------------------------------------------------------===//
// CellIFT Pass
//===----------------------------------------------------------------------===//

namespace {

class CellIFTInstrumentPass
    : public circt::impl::CellIFTInstrumentBase<CellIFTInstrumentPass> {
public:
  using CellIFTInstrumentBase::CellIFTInstrumentBase;
  void runOnOperation() override;

private:
  // ---- Taint map --------------------------------------------------------
  DenseMap<Value, Value> taintOf;

  // ---- Helpers ----------------------------------------------------------
  Value getZero(ImplicitLocOpBuilder &b, Type ty) {
    return hw::ConstantOp::create(b,
                                  APInt(cast<IntegerType>(ty).getWidth(), 0));
  }
  Value getAllOnes(ImplicitLocOpBuilder &b, Type ty) {
    return hw::ConstantOp::create(
        b, APInt::getAllOnes(cast<IntegerType>(ty).getWidth()));
  }
  Value orReduce(ImplicitLocOpBuilder &b, Value v) {
    auto w = cast<IntegerType>(v.getType()).getWidth();
    if (w == 1)
      return v;
    return comb::ICmpOp::create(b, comb::ICmpPredicate::ne, v,
                                getZero(b, v.getType()));
  }
  Value broadcast(ImplicitLocOpBuilder &b, Value bit, Type ty) {
    return b.createOrFold<comb::ReplicateOp>(ty, bit);
  }
  /// Conservative: OR-reduce all input taints, broadcast to result width.
  Value conservativeTaint(ImplicitLocOpBuilder &b, ValueRange ins, Type resTy) {
    SmallVector<Value> bits;
    for (auto v : ins) {
      if (auto it = taintOf.find(v); it != taintOf.end())
        bits.push_back(orReduce(b, it->second));
    }
    Value any;
    if (bits.empty())
      return getZero(b, resTy);
    if (bits.size() == 1)
      any = bits[0];
    else
      any = comb::OrOp::create(b, bits, /*twoState=*/false);
    return broadcast(b, any, resTy);
  }

  Value getTaint(Value v) {
    auto it = taintOf.find(v);
    assert(it != taintOf.end() && "missing taint");
    return it->second;
  }

  // ---- Taint rules per op -----------------------------------------------
  void instrConst(hw::ConstantOp, ImplicitLocOpBuilder &);
  void instrAnd(comb::AndOp, ImplicitLocOpBuilder &);
  void instrOr(comb::OrOp, ImplicitLocOpBuilder &);
  void instrXor(comb::XorOp, ImplicitLocOpBuilder &);
  void instrAdd(comb::AddOp, ImplicitLocOpBuilder &);
  void instrSub(comb::SubOp, ImplicitLocOpBuilder &);
  void instrMul(comb::MulOp, ImplicitLocOpBuilder &);
  void instrMux(comb::MuxOp, ImplicitLocOpBuilder &);
  void instrConcat(comb::ConcatOp, ImplicitLocOpBuilder &);
  void instrExtract(comb::ExtractOp, ImplicitLocOpBuilder &);
  void instrReplicate(comb::ReplicateOp, ImplicitLocOpBuilder &);
  void instrICmp(comb::ICmpOp, ImplicitLocOpBuilder &);
  void instrShl(comb::ShlOp, ImplicitLocOpBuilder &);
  void instrShrU(comb::ShrUOp, ImplicitLocOpBuilder &);
  void instrShrS(comb::ShrSOp, ImplicitLocOpBuilder &);
  void instrParity(comb::ParityOp, ImplicitLocOpBuilder &);
  void instrDiv(Operation *, ImplicitLocOpBuilder &);
  void instrMod(Operation *, ImplicitLocOpBuilder &);
  void instrInstance(hw::InstanceOp, ImplicitLocOpBuilder &);

  Value shiftTaintPrecise(ImplicitLocOpBuilder &b, Value data, Value dataT,
                          Value amt, Value amtT, Type resTy, bool left,
                          bool arith);

  // ---- Module-level passes ----------------------------------------------
  void rewriteModuleSignature(HWModuleOp mod);
  void instrumentModuleBody(HWModuleOp mod);
  void rewriteOutputOp(HWModuleOp mod);
};

} // namespace

//===----------------------------------------------------------------------===//
// Taint Rules
//===----------------------------------------------------------------------===//

void CellIFTInstrumentPass::instrConst(hw::ConstantOp op,
                                       ImplicitLocOpBuilder &b) {
  taintOf[op.getResult()] =
      taintConstants ? getAllOnes(b, op.getType()) : getZero(b, op.getType());
}

// AND: y_t = (a & b_t) | (b & a_t) | (a_t & b_t)
void CellIFTInstrumentPass::instrAnd(comb::AndOp op, ImplicitLocOpBuilder &b) {
  auto ins = op.getInputs();
  // Fold pairwise from left.
  Value val = ins[0], vt = getTaint(ins[0]);
  for (size_t i = 1; i < ins.size(); ++i) {
    Value c = ins[i], ct = getTaint(ins[i]);
    Value t1 = comb::AndOp::create(b, val, ct);
    Value t2 = comb::AndOp::create(b, c, vt);
    Value t3 = comb::AndOp::create(b, vt, ct);
    vt = comb::OrOp::create(b, ValueRange{t1, t2, t3}, /*twoState=*/false);
    // Update running value (the non-taint AND so far).
    val = comb::AndOp::create(b, val, c);
  }
  taintOf[op.getResult()] = vt;
}

// OR: y_t = (~a & b_t) | (~b & a_t) | (a_t & b_t)
void CellIFTInstrumentPass::instrOr(comb::OrOp op, ImplicitLocOpBuilder &b) {
  auto ins = op.getInputs();
  Value val = ins[0], vt = getTaint(ins[0]);
  auto ones = getAllOnes(b, val.getType());
  for (size_t i = 1; i < ins.size(); ++i) {
    Value c = ins[i], ct = getTaint(ins[i]);
    Value notVal = comb::XorOp::create(b, val, ones);
    Value notC = comb::XorOp::create(b, c, ones);
    Value t1 = comb::AndOp::create(b, notVal, ct);
    Value t2 = comb::AndOp::create(b, notC, vt);
    Value t3 = comb::AndOp::create(b, vt, ct);
    vt = comb::OrOp::create(b, ValueRange{t1, t2, t3}, /*twoState=*/false);
    val = comb::OrOp::create(b, ValueRange{val, c}, /*twoState=*/false);
  }
  taintOf[op.getResult()] = vt;
}

// XOR: y_t = OR of all input taints.
void CellIFTInstrumentPass::instrXor(comb::XorOp op, ImplicitLocOpBuilder &b) {
  SmallVector<Value> ts;
  for (auto v : op.getInputs())
    ts.push_back(getTaint(v));
  taintOf[op.getResult()] =
      ts.size() == 1 ? ts[0] : comb::OrOp::create(b, ts, /*twoState=*/false);
}

// ADD (precise): y_t = ((a&~a_t)+(b&~b_t)) XOR ((a|a_t)+(b|b_t)) | a_t | b_t
void CellIFTInstrumentPass::instrAdd(comb::AddOp op, ImplicitLocOpBuilder &b) {
  auto ins = op.getInputs();
  Value val = ins[0], vt = getTaint(ins[0]);
  auto ty = val.getType();
  auto ones = getAllOnes(b, ty);

  for (size_t i = 1; i < ins.size(); ++i) {
    Value c = ins[i], ct = getTaint(ins[i]);
    Value notVt = comb::XorOp::create(b, vt, ones);
    Value notCt = comb::XorOp::create(b, ct, ones);
    Value aZero = comb::AndOp::create(b, val, notVt);
    Value bZero = comb::AndOp::create(b, c, notCt);
    Value aOne = comb::OrOp::create(b, val, vt);
    Value bOne = comb::OrOp::create(b, c, ct);
    Value sumMin = comb::AddOp::create(b, aZero, bZero);
    Value sumMax = comb::AddOp::create(b, aOne, bOne);
    Value xorResult = comb::XorOp::create(b, sumMin, sumMax);
    vt = comb::OrOp::create(b, ValueRange{xorResult, vt, ct},
                            /*twoState=*/false);
    val = comb::AddOp::create(b, val, c);
  }
  taintOf[op.getResult()] = vt;
}

// SUB (precise): y_t = ((a|a_t)-(b&~b_t)) XOR ((a&~a_t)-(b|b_t)) | a_t | b_t
void CellIFTInstrumentPass::instrSub(comb::SubOp op, ImplicitLocOpBuilder &b) {
  Value a = op.getLhs(), aT = getTaint(a);
  Value bv = op.getRhs(), bT = getTaint(bv);
  auto ty = a.getType();
  auto ones = getAllOnes(b, ty);

  Value notAT = comb::XorOp::create(b, aT, ones);
  Value notBT = comb::XorOp::create(b, bT, ones);
  Value aOne = comb::OrOp::create(b, a, aT);
  Value bZero = comb::AndOp::create(b, bv, notBT);
  Value aZero = comb::AndOp::create(b, a, notAT);
  Value bOne = comb::OrOp::create(b, bv, bT);

  Value sub1 = comb::SubOp::create(b, aOne, bZero);
  Value sub2 = comb::SubOp::create(b, aZero, bOne);
  Value xorResult = comb::XorOp::create(b, sub1, sub2);
  taintOf[op.getResult()] =
      comb::OrOp::create(b, ValueRange{xorResult, aT, bT}, /*twoState=*/false);
}

void CellIFTInstrumentPass::instrMul(comb::MulOp op, ImplicitLocOpBuilder &b) {
  taintOf[op.getResult()] = conservativeTaint(b, op.getInputs(), op.getType());
}

// DIV (conservative): any tainted input taints the full result.
void CellIFTInstrumentPass::instrDiv(Operation *op, ImplicitLocOpBuilder &b) {
  taintOf[op->getResult(0)] =
      conservativeTaint(b, op->getOperands(), op->getResult(0).getType());
}

// MOD (precise per Yosys): y_t = mod(a_t, b) | broadcast(reduce_or(b_t))
void CellIFTInstrumentPass::instrMod(Operation *op, ImplicitLocOpBuilder &b) {
  Value a = op->getOperand(0), bv = op->getOperand(1);
  Value aT = getTaint(a), bT = getTaint(bv);
  auto resTy = op->getResult(0).getType();

  // mod(a_t, b): taint of A propagated through modulo.
  Value modTaint;
  if (isa<comb::ModSOp>(op))
    modTaint = comb::ModSOp::create(b, aT, bv);
  else
    modTaint = comb::ModUOp::create(b, aT, bv);

  // If B is tainted at all, entire result is tainted.
  Value bTaintBit = orReduce(b, bT);
  Value bTaintBroad = broadcast(b, bTaintBit, resTy);
  taintOf[op->getResult(0)] = comb::OrOp::create(b, modTaint, bTaintBroad);
}

// MUX: y_t = mux(sel, t_t, f_t) | replicate(sel_t) & (t^f | t_t | f_t)
void CellIFTInstrumentPass::instrMux(comb::MuxOp op, ImplicitLocOpBuilder &b) {
  Value sel = op.getCond(), t = op.getTrueValue(), f = op.getFalseValue();
  Value sT = getTaint(sel), tT = getTaint(t), fT = getTaint(f);
  auto resTy = op.getResult().getType();

  Value dataTaint = comb::MuxOp::create(b, sel, tT, fT);
  Value selBroad = broadcast(b, sT, resTy);
  Value diff = comb::XorOp::create(b, t, f);
  Value inner =
      comb::OrOp::create(b, ValueRange{diff, tT, fT}, /*twoState=*/false);
  Value ctrlTaint = comb::AndOp::create(b, selBroad, inner);
  taintOf[op.getResult()] = comb::OrOp::create(b, dataTaint, ctrlTaint);
}

// CONCAT: y_t = concat(each input_t)
void CellIFTInstrumentPass::instrConcat(comb::ConcatOp op,
                                        ImplicitLocOpBuilder &b) {
  SmallVector<Value> ts;
  for (auto v : op.getInputs())
    ts.push_back(getTaint(v));
  taintOf[op.getResult()] = comb::ConcatOp::create(b, ts);
}

// EXTRACT: y_t = extract(input_t, lowBit)
void CellIFTInstrumentPass::instrExtract(comb::ExtractOp op,
                                         ImplicitLocOpBuilder &b) {
  taintOf[op.getResult()] = comb::ExtractOp::create(
      b, op.getResult().getType(), getTaint(op.getInput()), op.getLowBit());
}

// REPLICATE: y_t = replicate(input_t)
void CellIFTInstrumentPass::instrReplicate(comb::ReplicateOp op,
                                           ImplicitLocOpBuilder &b) {
  taintOf[op.getResult()] = comb::ReplicateOp::create(
      b, op.getResult().getType(), getTaint(op.getInput()));
}

// ICMP (precise): different rules per predicate.
void CellIFTInstrumentPass::instrICmp(comb::ICmpOp op,
                                      ImplicitLocOpBuilder &b) {
  Value a = op.getLhs(), aT = getTaint(a);
  Value bv = op.getRhs(), bT = getTaint(bv);
  auto pred = op.getPredicate();
  auto ty = a.getType();
  unsigned w = cast<IntegerType>(ty).getWidth();
  auto ones = getAllOnes(b, ty);

  if (pred == ICmpPredicate::eq || pred == ICmpPredicate::ne) {
    // Precise eq/ne: taint iff non-tainted bits match and any bit is tainted.
    Value combined = comb::OrOp::create(b, aT, bT);
    Value hasTaint = orReduce(b, combined);
    Value mask = comb::XorOp::create(b, combined, ones); // ~(a_t | b_t)
    Value maskedA = comb::AndOp::create(b, a, mask);
    Value maskedB = comb::AndOp::create(b, bv, mask);
    Value eqUntainted =
        comb::ICmpOp::create(b, ICmpPredicate::eq, maskedA, maskedB);
    taintOf[op.getResult()] = comb::AndOp::create(b, hasTaint, eqUntainted);
  } else {
    // ge/gt/le/lt: compute min/max values, compare extremes, XOR.
    bool isSigned = (pred == ICmpPredicate::slt || pred == ICmpPredicate::sle ||
                     pred == ICmpPredicate::sgt || pred == ICmpPredicate::sge);

    Value notAT = comb::XorOp::create(b, aT, ones);
    Value notBT = comb::XorOp::create(b, bT, ones);

    Value minA, maxA, minB, maxB;
    if (w == 1 || !isSigned) {
      // Unsigned: min = clear tainted bits, max = set tainted bits.
      minA = comb::AndOp::create(b, a, notAT);
      maxA = comb::OrOp::create(b, a, aT);
      minB = comb::AndOp::create(b, bv, notBT);
      maxB = comb::OrOp::create(b, bv, bT);
    } else {
      // Signed: MSB minimizes to 1 (negative), maximizes to 0 (positive).
      auto lsbTy = IntegerType::get(b.getContext(), w - 1);
      auto bitTy = IntegerType::get(b.getContext(), 1);

      Value aLsbs = comb::ExtractOp::create(b, lsbTy, a, 0);
      Value aMsb = comb::ExtractOp::create(b, bitTy, a, w - 1);
      Value aTLsbs = comb::ExtractOp::create(b, lsbTy, aT, 0);
      Value aTMsb = comb::ExtractOp::create(b, bitTy, aT, w - 1);

      Value bLsbs = comb::ExtractOp::create(b, lsbTy, bv, 0);
      Value bMsb = comb::ExtractOp::create(b, bitTy, bv, w - 1);
      Value bTLsbs = comb::ExtractOp::create(b, lsbTy, bT, 0);
      Value bTMsb = comb::ExtractOp::create(b, bitTy, bT, w - 1);

      auto lsbOnes = getAllOnes(b, lsbTy);
      Value notATLsbs = comb::XorOp::create(b, aTLsbs, lsbOnes);
      Value notBTLsbs = comb::XorOp::create(b, bTLsbs, lsbOnes);

      auto bitOnes = getAllOnes(b, bitTy);
      Value notATMsb = comb::XorOp::create(b, aTMsb, bitOnes);
      Value notBTMsb = comb::XorOp::create(b, bTMsb, bitOnes);

      // LSBs: unsigned convention.
      Value minALsbs = comb::AndOp::create(b, aLsbs, notATLsbs);
      Value maxALsbs = comb::OrOp::create(b, aLsbs, aTLsbs);
      Value minBLsbs = comb::AndOp::create(b, bLsbs, notBTLsbs);
      Value maxBLsbs = comb::OrOp::create(b, bLsbs, bTLsbs);

      // MSB: signed convention (min->1, max->0).
      Value minAMsb = comb::OrOp::create(b, aMsb, aTMsb);
      Value maxAMsb = comb::AndOp::create(b, aMsb, notATMsb);
      Value minBMsb = comb::OrOp::create(b, bMsb, bTMsb);
      Value maxBMsb = comb::AndOp::create(b, bMsb, notBTMsb);

      minA = comb::ConcatOp::create(b, minAMsb, minALsbs);
      maxA = comb::ConcatOp::create(b, maxAMsb, maxALsbs);
      minB = comb::ConcatOp::create(b, minBMsb, minBLsbs);
      maxB = comb::ConcatOp::create(b, maxBMsb, maxBLsbs);
    }

    // Compare extremes with the same predicate.
    Value cmp1 = comb::ICmpOp::create(b, pred, minA, maxB);
    Value cmp2 = comb::ICmpOp::create(b, pred, maxA, minB);
    taintOf[op.getResult()] = comb::XorOp::create(b, cmp1, cmp2);
  }
}

// Precise shift taint: two-phase approach matching Yosys CellIFT.
// Phase 1: shift by untainted portion of the shift amount.
// Phase 2: for each possible delta from tainted shift bits, check if the
//          output bit values would differ.
Value CellIFTInstrumentPass::shiftTaintPrecise(ImplicitLocOpBuilder &b,
                                               Value data, Value dataT,
                                               Value amt, Value amtT,
                                               Type resTy, bool left,
                                               bool arith) {
  unsigned W = cast<IntegerType>(resTy).getWidth();
  if (W <= 1)
    return conservativeTaint(b, ValueRange{data, amt}, resTy);

  // Phase 1: Shift by untainted amount.
  auto onesAmt = getAllOnes(b, amt.getType());
  Value notAmtT = comb::XorOp::create(b, amtT, onesAmt);
  Value untaintedAmt = comb::AndOp::create(b, amt, notAmtT);

  Value intermA, intermAT;
  if (left) {
    intermA = comb::ShlOp::create(b, data, untaintedAmt);
    intermAT = comb::ShlOp::create(b, dataT, untaintedAmt);
  } else if (arith) {
    intermA = comb::ShrSOp::create(b, data, untaintedAmt);
    intermAT = comb::ShrSOp::create(b, dataT, untaintedAmt);
  } else {
    intermA = comb::ShrUOp::create(b, data, untaintedAmt);
    intermAT = comb::ShrUOp::create(b, dataT, untaintedAmt);
  }

  Value intermAOrT = comb::OrOp::create(b, intermA, intermAT);

  // Phase 2: For each possible delta k from 1 to W-1, check if tainted
  // shift amount can reach delta k and whether it changes the output.
  SmallVector<Value> contributions;
  contributions.push_back(intermAT);

  unsigned amtW = cast<IntegerType>(amt.getType()).getWidth();
  for (unsigned k = 1; k < W; k++) {
    if (k >= (1u << amtW))
      break;

    // Can tainted shift bits produce delta k? Check (amtT & k) == k.
    Value constKAmt = hw::ConstantOp::create(b, APInt(amtW, k));
    Value kAndAmtT = comb::AndOp::create(b, amtT, constKAmt);
    Value canReach =
        comb::ICmpOp::create(b, ICmpPredicate::eq, kAndAmtT, constKAmt);

    // Shift intermediate result by constant k in the same direction.
    Value constKW = hw::ConstantOp::create(b, APInt(W, k));
    Value shifted, shiftedT;
    if (left) {
      shifted = comb::ShlOp::create(b, intermA, constKW);
      shiftedT = comb::ShlOp::create(b, intermAT, constKW);
    } else if (arith) {
      shifted = comb::ShrSOp::create(b, intermA, constKW);
      shiftedT = comb::ShrSOp::create(b, intermAT, constKW);
    } else {
      shifted = comb::ShrUOp::create(b, intermA, constKW);
      shiftedT = comb::ShrUOp::create(b, intermAT, constKW);
    }

    // Bits where values differ or either is tainted.
    Value differ = comb::XorOp::create(b, intermA, shifted);
    Value taintEither = comb::OrOp::create(b, intermAT, shiftedT);
    Value tod = comb::OrOp::create(b, differ, taintEither);

    Value canReachBroad = broadcast(b, canReach, resTy);
    contributions.push_back(comb::AndOp::create(b, canReachBroad, tod));
  }

  // Large shift: amtT >= W means shift could exceed data width.
  if (W < (1u << amtW)) {
    Value constW = hw::ConstantOp::create(b, APInt(amtW, W));
    Value isLargeShift =
        comb::ICmpOp::create(b, ICmpPredicate::uge, amtT, constW);
    Value largeBroad = broadcast(b, isLargeShift, resTy);

    if (left || !arith) {
      // SHL or logical SHR: zeros fill, taint if intermAOrT is nonzero.
      contributions.push_back(comb::AndOp::create(b, largeBroad, intermAOrT));
    } else {
      // Arithmetic SHR: sign bit fills. Compare each position with sign bit.
      auto bitTy = IntegerType::get(b.getContext(), 1);
      Value signBit = comb::ExtractOp::create(b, bitTy, intermA, W - 1);
      Value signTaint = comb::ExtractOp::create(b, bitTy, intermAT, W - 1);
      Value signBroad = broadcast(b, signBit, resTy);
      Value signTBroad = broadcast(b, signTaint, resTy);
      Value diffSign = comb::XorOp::create(b, intermA, signBroad);
      Value taintSign = comb::OrOp::create(b, intermAT, signTBroad);
      Value todSign = comb::OrOp::create(b, diffSign, taintSign);
      contributions.push_back(comb::AndOp::create(b, largeBroad, todSign));
    }
  }

  if (contributions.size() == 1)
    return contributions[0];
  return comb::OrOp::create(b, contributions, /*twoState=*/false);
}

void CellIFTInstrumentPass::instrShl(comb::ShlOp op, ImplicitLocOpBuilder &b) {
  taintOf[op.getResult()] =
      shiftTaintPrecise(b, op.getLhs(), getTaint(op.getLhs()), op.getRhs(),
                        getTaint(op.getRhs()), op.getType(), true, false);
}
void CellIFTInstrumentPass::instrShrU(comb::ShrUOp op,
                                      ImplicitLocOpBuilder &b) {
  taintOf[op.getResult()] =
      shiftTaintPrecise(b, op.getLhs(), getTaint(op.getLhs()), op.getRhs(),
                        getTaint(op.getRhs()), op.getType(), false, false);
}
void CellIFTInstrumentPass::instrShrS(comb::ShrSOp op,
                                      ImplicitLocOpBuilder &b) {
  taintOf[op.getResult()] =
      shiftTaintPrecise(b, op.getLhs(), getTaint(op.getLhs()), op.getRhs(),
                        getTaint(op.getRhs()), op.getType(), false, true);
}

// PARITY: y_t = OR-reduce(input_t)
void CellIFTInstrumentPass::instrParity(comb::ParityOp op,
                                        ImplicitLocOpBuilder &b) {
  taintOf[op.getResult()] = orReduce(b, getTaint(op.getInput()));
}

//===----------------------------------------------------------------------===//
// Instance Instrumentation
//===----------------------------------------------------------------------===//

void CellIFTInstrumentPass::instrInstance(hw::InstanceOp op,
                                          ImplicitLocOpBuilder &b) {
  SmallVector<Value> newInputs;
  SmallVector<Attribute> newArgNames, newResNames;
  SmallVector<Type> newResTys;

  for (auto [input, nameAttr] :
       llvm::zip_equal(op.getInputs(), op.getInputNames())) {
    auto name = cast<StringAttr>(nameAttr);
    newInputs.push_back(input);
    newArgNames.push_back(name);

    if (!isa<IntegerType>(input.getType()))
      continue;

    auto it = taintOf.find(input);
    newInputs.push_back(it != taintOf.end() ? it->second
                                            : getZero(b, input.getType()));
    newArgNames.push_back(b.getStringAttr(name.getValue().str() + taintSuffix));
  }

  for (auto [result, nameAttr] :
       llvm::zip_equal(op.getResults(), op.getOutputNames())) {
    auto name = cast<StringAttr>(nameAttr);
    newResTys.push_back(result.getType());
    newResNames.push_back(name);

    if (!isa<IntegerType>(result.getType()))
      continue;

    newResTys.push_back(result.getType());
    newResNames.push_back(b.getStringAttr(name.getValue().str() + taintSuffix));
  }

  auto newInst = hw::InstanceOp::create(
      b, op.getLoc(), newResTys, op.getInstanceNameAttr(),
      op.getModuleNameAttr(), newInputs, b.getArrayAttr(newArgNames),
      b.getArrayAttr(newResNames), op.getParametersAttr(), op.getInnerSymAttr(),
      op.getDoNotPrintAttr());

  unsigned newResIdx = 0;
  for (auto result : op.getResults()) {
    Value newResult = newInst.getResult(newResIdx++);
    result.replaceAllUsesWith(newResult);
    if (isa<IntegerType>(result.getType()))
      taintOf[newResult] = newInst.getResult(newResIdx++);
  }

  op.erase();
}

//===----------------------------------------------------------------------===//
// Module Signature Rewriting
//===----------------------------------------------------------------------===//

void CellIFTInstrumentPass::rewriteModuleSignature(HWModuleOp mod) {
  auto *ctx = mod.getContext();
  auto portList = mod.getPortList();
  Block *body = mod.getBodyBlock();

  SmallVector<std::pair<unsigned, PortInfo>> insIns, insOuts;

  // Insert a taint port after every integer-typed input/output.
  // Also insert block arguments for the new input ports.
  unsigned inIdx = 0, outIdx = 0;
  unsigned blockArgInsertionOffset = 0;
  for (auto port : portList) {
    if (port.isInput()) {
      if (isa<IntegerType>(port.type)) {
        PortInfo tp;
        tp.name = StringAttr::get(ctx, port.getName().str() + taintSuffix);
        tp.type = port.type;
        tp.dir = ModulePort::Direction::Input;
        tp.loc = port.loc ? port.loc : UnknownLoc::get(ctx);
        insIns.push_back({inIdx + 1, tp});
        // Insert the block argument right after the current input's arg.
        body->insertArgument(inIdx + 1 + blockArgInsertionOffset, tp.type,
                             cast<Location>(tp.loc));
        blockArgInsertionOffset++;
      }
      inIdx++;
    } else if (port.isOutput()) {
      if (isa<IntegerType>(port.type)) {
        PortInfo tp;
        tp.name = StringAttr::get(ctx, port.getName().str() + taintSuffix);
        tp.type = port.type;
        tp.dir = ModulePort::Direction::Output;
        tp.loc = port.loc ? port.loc : UnknownLoc::get(ctx);
        insOuts.push_back({outIdx + 1, tp});
      }
      outIdx++;
    }
  }

  mod.modifyPorts(insIns, insOuts, {}, {});
}

//===----------------------------------------------------------------------===//
// Body Instrumentation
//===----------------------------------------------------------------------===//

void CellIFTInstrumentPass::instrumentModuleBody(HWModuleOp mod) {
  taintOf.clear();
  Block *body = mod.getBodyBlock();

  // Map input block args -> their taint block args.
  unsigned n = mod.getNumInputPorts();
  for (unsigned i = 0; i < n; ++i) {
    StringRef nm = mod.getInputName(i);
    if (nm.ends_with(taintSuffix) && nm.size() > taintSuffix.size() && i > 0) {
      taintOf[body->getArgument(i - 1)] = body->getArgument(i);
    }
  }

  // Sort the block topologically to handle graph regions where ops may not
  // be in def-use order. Register outputs are considered "ready" since we
  // pre-create their taint registers in phase 1.
  mlir::sortTopologically(body, [](Value v, Operation *) {
    if (isa<BlockArgument>(v))
      return true;
    if (auto defOp = v.getDefiningOp())
      if (isa<seq::CompRegOp, seq::FirRegOp>(defOp))
        return true;
    return false;
  });

  // Collect ops to instrument (avoid iterator invalidation).
  SmallVector<Operation *> ops;
  for (auto &op : *body)
    if (!isa<hw::OutputOp>(op))
      ops.push_back(&op);

  ImplicitLocOpBuilder b(mod.getLoc(), mod.getContext());

  // Phase 1: Pre-create taint registers for all seq registers.
  // This breaks cycles: register output taints are available before we
  // process combinational ops that form feedback loops.
  struct PendingReg {
    Operation *taintReg;
    Value origInput; // the data input whose taint we need
  };
  SmallVector<PendingReg> pendingRegs;

  for (auto *op : ops) {
    b.setLoc(op->getLoc());
    b.setInsertionPointAfter(op);

    if (auto compreg = dyn_cast<seq::CompRegOp>(op)) {
      StringAttr name;
      if (auto n = compreg.getNameAttr())
        name = b.getStringAttr(n.getValue().str() + taintSuffix);

      // Create with zero placeholder input; we'll fix it in phase 3.
      Value zero = getZero(b, compreg.getType());
      Value tReg;
      if (compreg.getReset()) {
        Value rstVal = getZero(b, compreg.getType());
        tReg =
            seq::CompRegOp::create(b, compreg.getLoc(), zero, compreg.getClk(),
                                   compreg.getReset(), rstVal, name);
      } else {
        tReg = seq::CompRegOp::create(b, compreg.getLoc(), zero,
                                      compreg.getClk(), name);
      }
      taintOf[compreg.getData()] = tReg;
      pendingRegs.push_back({tReg.getDefiningOp(), compreg.getInput()});
    } else if (auto firreg = dyn_cast<seq::FirRegOp>(op)) {
      StringAttr name = b.getStringAttr(firreg.getName().str() + taintSuffix);

      Value zero = getZero(b, firreg.getType());
      Value tReg;
      if (firreg.hasReset()) {
        Value rstVal = getZero(b, firreg.getType());
        tReg = seq::FirRegOp::create(
            b, zero, firreg.getClk(), name, firreg.getReset(), rstVal,
            firreg.getInnerSymAttr(), firreg.getIsAsync());
      } else {
        tReg = seq::FirRegOp::create(b, zero, firreg.getClk(), name);
      }
      taintOf[firreg.getData()] = tReg;
      pendingRegs.push_back({tReg.getDefiningOp(), firreg.getNext()});
    }
  }

  // Phase 2: Instrument all other ops (register taints are now available).
  for (auto *op : ops) {
    if (isa<seq::CompRegOp, seq::FirRegOp>(op))
      continue; // already handled in phase 1
    b.setLoc(op->getLoc());
    b.setInsertionPointAfter(op);

    llvm::TypeSwitch<Operation *>(op)
        .Case<hw::ConstantOp>([&](auto o) { instrConst(o, b); })
        .Case<comb::AndOp>([&](auto o) { instrAnd(o, b); })
        .Case<comb::OrOp>([&](auto o) { instrOr(o, b); })
        .Case<comb::XorOp>([&](auto o) { instrXor(o, b); })
        .Case<comb::AddOp>([&](auto o) { instrAdd(o, b); })
        .Case<comb::SubOp>([&](auto o) { instrSub(o, b); })
        .Case<comb::MulOp>([&](auto o) { instrMul(o, b); })
        .Case<comb::DivUOp, comb::DivSOp>([&](auto o) { instrDiv(o, b); })
        .Case<comb::ModUOp, comb::ModSOp>([&](auto o) { instrMod(o, b); })
        .Case<comb::MuxOp>([&](auto o) { instrMux(o, b); })
        .Case<comb::ConcatOp>([&](auto o) { instrConcat(o, b); })
        .Case<comb::ExtractOp>([&](auto o) { instrExtract(o, b); })
        .Case<comb::ReplicateOp>([&](auto o) { instrReplicate(o, b); })
        .Case<comb::ICmpOp>([&](auto o) { instrICmp(o, b); })
        .Case<comb::ShlOp>([&](auto o) { instrShl(o, b); })
        .Case<comb::ShrUOp>([&](auto o) { instrShrU(o, b); })
        .Case<comb::ShrSOp>([&](auto o) { instrShrS(o, b); })
        .Case<comb::ParityOp>([&](auto o) { instrParity(o, b); })
        .Case<hw::InstanceOp>([&](auto o) { instrInstance(o, b); })
        .Default([&](Operation *o) {
          // Conservative fallback for unknown ops with integer results.
          for (auto res : o->getResults())
            if (isa<IntegerType>(res.getType()))
              taintOf[res] =
                  conservativeTaint(b, o->getOperands(), res.getType());
        });
  }

  // Phase 3: Fix up taint register inputs now that all taints are computed.
  for (auto &pr : pendingRegs) {
    Value inputT = getTaint(pr.origInput);
    if (auto compreg = dyn_cast<seq::CompRegOp>(pr.taintReg))
      compreg.getInputMutable().assign(inputT);
    else if (auto firreg = dyn_cast<seq::FirRegOp>(pr.taintReg))
      firreg.getNextMutable().assign(inputT);
  }
}

void CellIFTInstrumentPass::rewriteOutputOp(HWModuleOp mod) {
  Block *body = mod.getBodyBlock();
  auto outputOp = cast<hw::OutputOp>(body->getTerminator());

  SmallVector<PortInfo> outPorts;
  for (auto p : mod.getPortList())
    if (p.isOutput())
      outPorts.push_back(p);

  SmallVector<Value> newOuts;
  unsigned origIdx = 0;
  ImplicitLocOpBuilder b(outputOp.getLoc(), outputOp);

  for (auto &port : outPorts) {
    StringRef pn = port.getName();
    if (pn.ends_with(taintSuffix) && pn.size() > taintSuffix.size()) {
      // Taint output: provide taint of the previous output value.
      Value prev = newOuts.back();
      auto it = taintOf.find(prev);
      if (it != taintOf.end())
        newOuts.push_back(it->second);
      else
        newOuts.push_back(getZero(b, port.type));
    } else {
      newOuts.push_back(outputOp.getOperand(origIdx++));
    }
  }

  hw::OutputOp::create(b, outputOp.getLoc(), newOuts);
  outputOp.erase();
}

//===----------------------------------------------------------------------===//
// Main
//===----------------------------------------------------------------------===//

void CellIFTInstrumentPass::runOnOperation() {
  SmallVector<HWModuleOp> modules;
  auto &instanceGraph = getAnalysis<hw::InstanceGraph>();
  instanceGraph.walkPostOrder([&](igraph::InstanceGraphNode &node) {
    if (auto mod =
            dyn_cast_or_null<HWModuleOp>(node.getModule().getOperation()))
      modules.push_back(mod);
  });

  for (auto mod : modules) {
    // Step 1: Rewrite module signature (add taint ports).
    rewriteModuleSignature(mod);
    // Step 2: Instrument all ops in the body (including instances).
    instrumentModuleBody(mod);
    // Step 3: Rewrite the output op.
    rewriteOutputOp(mod);
  }
}

//===----------------------------------------------------------------------===//
// Registration
//===----------------------------------------------------------------------===//

namespace circt {
std::unique_ptr<mlir::Pass> createCellIFTInstrumentPass() {
  return std::make_unique<CellIFTInstrumentPass>();
}
} // namespace circt
