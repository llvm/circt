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
#include "mlir/IR/Threading.h"
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

using TaintMap = DenseMap<Value, Value>;

Value getZero(ImplicitLocOpBuilder &b, Type ty);
Value getAllOnes(ImplicitLocOpBuilder &b, Type ty);
Value orReduce(ImplicitLocOpBuilder &b, Value v);
Value broadcast(ImplicitLocOpBuilder &b, Value bit, Type ty);
Value getTaint(const TaintMap &taintOf, Value v);
Value conservativeTaint(ImplicitLocOpBuilder &b, ValueRange taintInputs,
                        Type resTy);
Value conservativeTaint(ImplicitLocOpBuilder &b, const TaintMap &taintOf,
                        ValueRange inputs, Type resTy);
Value shiftTaintPrecise(ImplicitLocOpBuilder &b, Value data, Value dataT,
                        Value amt, Value amtT, Type resTy, bool left,
                        bool arith);

Value instrumentOperation(ImplicitLocOpBuilder &b, hw::ConstantOp op,
                          bool taintConstants);
Value instrumentOperation(ImplicitLocOpBuilder &b, comb::AndOp op,
                          comb::AndOp::Adaptor taintAdaptor);
Value instrumentOperation(ImplicitLocOpBuilder &b, comb::OrOp op,
                          comb::OrOp::Adaptor taintAdaptor);
Value instrumentOperation(ImplicitLocOpBuilder &b, comb::XorOp op,
                          comb::XorOp::Adaptor taintAdaptor);
Value instrumentOperation(ImplicitLocOpBuilder &b, comb::AddOp op,
                          comb::AddOp::Adaptor taintAdaptor);
Value instrumentOperation(ImplicitLocOpBuilder &b, comb::SubOp op,
                          comb::SubOp::Adaptor taintAdaptor);
Value instrumentOperation(ImplicitLocOpBuilder &b, comb::MulOp op,
                          comb::MulOp::Adaptor taintAdaptor);
Value instrumentOperation(ImplicitLocOpBuilder &b, comb::DivUOp op,
                          comb::DivUOp::Adaptor taintAdaptor);
Value instrumentOperation(ImplicitLocOpBuilder &b, comb::DivSOp op,
                          comb::DivSOp::Adaptor taintAdaptor);
Value instrumentOperation(ImplicitLocOpBuilder &b, comb::ModUOp op,
                          comb::ModUOp::Adaptor taintAdaptor);
Value instrumentOperation(ImplicitLocOpBuilder &b, comb::ModSOp op,
                          comb::ModSOp::Adaptor taintAdaptor);
Value instrumentOperation(ImplicitLocOpBuilder &b, comb::MuxOp op,
                          comb::MuxOp::Adaptor taintAdaptor);
Value instrumentOperation(ImplicitLocOpBuilder &b, comb::ConcatOp op,
                          comb::ConcatOp::Adaptor taintAdaptor);
Value instrumentOperation(ImplicitLocOpBuilder &b, comb::ExtractOp op,
                          comb::ExtractOp::Adaptor taintAdaptor);
Value instrumentOperation(ImplicitLocOpBuilder &b, comb::ReplicateOp op,
                          comb::ReplicateOp::Adaptor taintAdaptor);
Value instrumentOperation(ImplicitLocOpBuilder &b, comb::ICmpOp op,
                          comb::ICmpOp::Adaptor taintAdaptor);
Value instrumentOperation(ImplicitLocOpBuilder &b, comb::ShlOp op,
                          comb::ShlOp::Adaptor taintAdaptor);
Value instrumentOperation(ImplicitLocOpBuilder &b, comb::ShrUOp op,
                          comb::ShrUOp::Adaptor taintAdaptor);
Value instrumentOperation(ImplicitLocOpBuilder &b, comb::ShrSOp op,
                          comb::ShrSOp::Adaptor taintAdaptor);
Value instrumentOperation(ImplicitLocOpBuilder &b, comb::ParityOp op,
                          comb::ParityOp::Adaptor taintAdaptor);
Value instrumentOperation(ImplicitLocOpBuilder &b, comb::ReverseOp op,
                          comb::ReverseOp::Adaptor taintAdaptor);

template <typename OpT>
void instrumentSingleResultOperation(TaintMap &taintOf, ImplicitLocOpBuilder &b,
                                     OpT op) {
  SmallVector<Value> taintOperands;
  taintOperands.reserve(op->getNumOperands());
  for (auto operand : op->getOperands())
    taintOperands.push_back(getTaint(taintOf, operand));
  typename OpT::Adaptor taintAdaptor(taintOperands, op->getAttrDictionary());
  taintOf[op.getResult()] = instrumentOperation(b, op, taintAdaptor);
}

void instrumentInstance(TaintMap &taintOf, StringRef taintSuffix,
                        hw::InstanceOp op, ImplicitLocOpBuilder &b);

class CellIFTInstrumentPass
    : public circt::impl::CellIFTInstrumentBase<CellIFTInstrumentPass> {
public:
  using CellIFTInstrumentBase::CellIFTInstrumentBase;
  void runOnOperation() override;

private:
  // ---- Module-level passes ----------------------------------------------
  void rewriteModuleSignature(HWModuleOp mod, StringRef taintSuffix);
  void instrumentModuleBody(HWModuleOp mod, TaintMap &taintOf,
                            bool taintConstants, StringRef taintSuffix);
  void rewriteOutputOp(HWModuleOp mod, TaintMap &taintOf,
                       StringRef taintSuffix);
};

} // namespace

//===----------------------------------------------------------------------===//
// Taint Rules
//===----------------------------------------------------------------------===//

namespace {

Value getZero(ImplicitLocOpBuilder &b, Type ty) {
  return hw::ConstantOp::create(b, APInt(cast<IntegerType>(ty).getWidth(), 0));
}

Value getAllOnes(ImplicitLocOpBuilder &b, Type ty) {
  return hw::ConstantOp::create(
      b, APInt::getAllOnes(cast<IntegerType>(ty).getWidth()));
}

Value orReduce(ImplicitLocOpBuilder &b, Value v) {
  auto width = cast<IntegerType>(v.getType()).getWidth();
  if (width == 1)
    return v;
  return comb::ICmpOp::create(b, comb::ICmpPredicate::ne, v,
                              getZero(b, v.getType()));
}

Value broadcast(ImplicitLocOpBuilder &b, Value bit, Type ty) {
  return b.createOrFold<comb::ReplicateOp>(ty, bit);
}

Value getTaint(const TaintMap &taintOf, Value v) {
  auto it = taintOf.find(v);
  assert(it != taintOf.end() && "missing taint");
  return it->second;
}

Value conservativeTaint(ImplicitLocOpBuilder &b, ValueRange taintInputs,
                        Type resTy) {
  SmallVector<Value> bits;
  bits.reserve(taintInputs.size());
  for (auto taint : taintInputs)
    bits.push_back(orReduce(b, taint));

  if (bits.empty())
    return getZero(b, resTy);

  Value any = bits.size() == 1
                  ? bits.front()
                  : comb::OrOp::create(b, bits, /*twoState=*/false);
  return broadcast(b, any, resTy);
}

Value conservativeTaint(ImplicitLocOpBuilder &b, const TaintMap &taintOf,
                        ValueRange inputs, Type resTy) {
  SmallVector<Value> taintInputs;
  taintInputs.reserve(inputs.size());
  for (auto input : inputs)
    if (auto it = taintOf.find(input); it != taintOf.end())
      taintInputs.push_back(it->second);
  return conservativeTaint(b, taintInputs, resTy);
}

Value instrumentOperation(ImplicitLocOpBuilder &b, hw::ConstantOp op,
                          bool taintConstants) {
  return taintConstants ? getAllOnes(b, op.getType())
                        : getZero(b, op.getType());
}

// AND: y_t = (a & b_t) | (b & a_t) | (a_t & b_t)
Value instrumentOperation(ImplicitLocOpBuilder &b, comb::AndOp op,
                          comb::AndOp::Adaptor taintAdaptor) {
  auto inputs = op.getInputs();
  auto taintInputs = taintAdaptor.getInputs();
  Value value = inputs.front();
  Value taint = taintInputs.front();
  for (auto [input, inputTaint] :
       llvm::zip_equal(inputs.drop_front(), taintInputs.drop_front())) {
    Value t1 = comb::AndOp::create(b, value, inputTaint);
    Value t2 = comb::AndOp::create(b, input, taint);
    Value t3 = comb::AndOp::create(b, taint, inputTaint);
    taint = comb::OrOp::create(b, ValueRange{t1, t2, t3}, /*twoState=*/false);
    value = comb::AndOp::create(b, value, input);
  }
  return taint;
}

// OR: y_t = (~a & b_t) | (~b & a_t) | (a_t & b_t)
Value instrumentOperation(ImplicitLocOpBuilder &b, comb::OrOp op,
                          comb::OrOp::Adaptor taintAdaptor) {
  auto inputs = op.getInputs();
  auto taintInputs = taintAdaptor.getInputs();
  Value value = inputs.front();
  Value taint = taintInputs.front();
  auto ones = getAllOnes(b, value.getType());
  for (auto [input, inputTaint] :
       llvm::zip_equal(inputs.drop_front(), taintInputs.drop_front())) {
    Value notValue = comb::XorOp::create(b, value, ones);
    Value notInput = comb::XorOp::create(b, input, ones);
    Value t1 = comb::AndOp::create(b, notValue, inputTaint);
    Value t2 = comb::AndOp::create(b, notInput, taint);
    Value t3 = comb::AndOp::create(b, taint, inputTaint);
    taint = comb::OrOp::create(b, ValueRange{t1, t2, t3}, /*twoState=*/false);
    value = comb::OrOp::create(b, ValueRange{value, input},
                               /*twoState=*/false);
  }
  return taint;
}

// XOR: y_t = OR of all input taints.
Value instrumentOperation(ImplicitLocOpBuilder &b, comb::XorOp,
                          comb::XorOp::Adaptor taintAdaptor) {
  auto taintInputs = taintAdaptor.getInputs();
  return taintInputs.size() == 1
             ? taintInputs.front()
             : comb::OrOp::create(b, taintInputs, /*twoState=*/false);
}

// ADD (precise): y_t = ((a&~a_t)+(b&~b_t)) XOR ((a|a_t)+(b|b_t)) | a_t | b_t
Value instrumentOperation(ImplicitLocOpBuilder &b, comb::AddOp op,
                          comb::AddOp::Adaptor taintAdaptor) {
  auto inputs = op.getInputs();
  auto taintInputs = taintAdaptor.getInputs();
  Value value = inputs.front();
  Value taint = taintInputs.front();
  auto ones = getAllOnes(b, value.getType());

  for (auto [input, inputTaint] :
       llvm::zip_equal(inputs.drop_front(), taintInputs.drop_front())) {
    Value notTaint = comb::XorOp::create(b, taint, ones);
    Value notInputTaint = comb::XorOp::create(b, inputTaint, ones);
    Value valueZero = comb::AndOp::create(b, value, notTaint);
    Value inputZero = comb::AndOp::create(b, input, notInputTaint);
    Value valueOne = comb::OrOp::create(b, value, taint);
    Value inputOne = comb::OrOp::create(b, input, inputTaint);
    Value sumMin = comb::AddOp::create(b, valueZero, inputZero);
    Value sumMax = comb::AddOp::create(b, valueOne, inputOne);
    Value xorResult = comb::XorOp::create(b, sumMin, sumMax);
    taint = comb::OrOp::create(b, ValueRange{xorResult, taint, inputTaint},
                               /*twoState=*/false);
    value = comb::AddOp::create(b, value, input);
  }
  return taint;
}

// SUB (precise): y_t = ((a|a_t)-(b&~b_t)) XOR ((a&~a_t)-(b|b_t)) | a_t | b_t
Value instrumentOperation(ImplicitLocOpBuilder &b, comb::SubOp op,
                          comb::SubOp::Adaptor taintAdaptor) {
  Value a = op.getLhs();
  Value aT = taintAdaptor.getLhs();
  Value bv = op.getRhs();
  Value bT = taintAdaptor.getRhs();
  auto ones = getAllOnes(b, a.getType());

  Value notAT = comb::XorOp::create(b, aT, ones);
  Value notBT = comb::XorOp::create(b, bT, ones);
  Value aOne = comb::OrOp::create(b, a, aT);
  Value bZero = comb::AndOp::create(b, bv, notBT);
  Value aZero = comb::AndOp::create(b, a, notAT);
  Value bOne = comb::OrOp::create(b, bv, bT);

  Value sub1 = comb::SubOp::create(b, aOne, bZero);
  Value sub2 = comb::SubOp::create(b, aZero, bOne);
  Value xorResult = comb::XorOp::create(b, sub1, sub2);
  return comb::OrOp::create(b, ValueRange{xorResult, aT, bT},
                            /*twoState=*/false);
}

Value instrumentOperation(ImplicitLocOpBuilder &b, comb::MulOp op,
                          comb::MulOp::Adaptor taintAdaptor) {
  return conservativeTaint(b, taintAdaptor.getInputs(), op.getType());
}

// DIV (conservative): any tainted input taints the full result.
Value instrumentOperation(ImplicitLocOpBuilder &b, comb::DivUOp op,
                          comb::DivUOp::Adaptor taintAdaptor) {
  SmallVector<Value> taintInputs{taintAdaptor.getLhs(), taintAdaptor.getRhs()};
  return conservativeTaint(b, taintInputs, op.getType());
}

Value instrumentOperation(ImplicitLocOpBuilder &b, comb::DivSOp op,
                          comb::DivSOp::Adaptor taintAdaptor) {
  SmallVector<Value> taintInputs{taintAdaptor.getLhs(), taintAdaptor.getRhs()};
  return conservativeTaint(b, taintInputs, op.getType());
}

// MOD (precise per Yosys): y_t = mod(a_t, b) | broadcast(reduce_or(b_t))
Value instrumentOperation(ImplicitLocOpBuilder &b, comb::ModUOp op,
                          comb::ModUOp::Adaptor taintAdaptor) {
  Value modTaint = comb::ModUOp::create(b, taintAdaptor.getLhs(), op.getRhs());
  Value bTaintBit = orReduce(b, taintAdaptor.getRhs());
  Value bTaintBroad = broadcast(b, bTaintBit, op.getType());
  return comb::OrOp::create(b, modTaint, bTaintBroad);
}

Value instrumentOperation(ImplicitLocOpBuilder &b, comb::ModSOp op,
                          comb::ModSOp::Adaptor taintAdaptor) {
  Value modTaint = comb::ModSOp::create(b, taintAdaptor.getLhs(), op.getRhs());
  Value bTaintBit = orReduce(b, taintAdaptor.getRhs());
  Value bTaintBroad = broadcast(b, bTaintBit, op.getType());
  return comb::OrOp::create(b, modTaint, bTaintBroad);
}

// MUX: y_t = mux(sel, t_t, f_t) | replicate(sel_t) & (t^f | t_t | f_t)
Value instrumentOperation(ImplicitLocOpBuilder &b, comb::MuxOp op,
                          comb::MuxOp::Adaptor taintAdaptor) {
  Value dataTaint =
      comb::MuxOp::create(b, op.getCond(), taintAdaptor.getTrueValue(),
                          taintAdaptor.getFalseValue());
  Value selBroad = broadcast(b, taintAdaptor.getCond(), op.getType());
  Value diff = comb::XorOp::create(b, op.getTrueValue(), op.getFalseValue());
  Value inner = comb::OrOp::create(b,
                                   ValueRange{diff, taintAdaptor.getTrueValue(),
                                              taintAdaptor.getFalseValue()},
                                   /*twoState=*/false);
  Value ctrlTaint = comb::AndOp::create(b, selBroad, inner);
  return comb::OrOp::create(b, dataTaint, ctrlTaint);
}

// CONCAT: y_t = concat(each input_t)
Value instrumentOperation(ImplicitLocOpBuilder &b, comb::ConcatOp,
                          comb::ConcatOp::Adaptor taintAdaptor) {
  return comb::ConcatOp::create(b, taintAdaptor.getInputs());
}

// EXTRACT: y_t = extract(input_t, lowBit)
Value instrumentOperation(ImplicitLocOpBuilder &b, comb::ExtractOp op,
                          comb::ExtractOp::Adaptor taintAdaptor) {
  return comb::ExtractOp::create(b, op.getResult().getType(),
                                 taintAdaptor.getInput(), op.getLowBit());
}

// REPLICATE: y_t = replicate(input_t)
Value instrumentOperation(ImplicitLocOpBuilder &b, comb::ReplicateOp op,
                          comb::ReplicateOp::Adaptor taintAdaptor) {
  return comb::ReplicateOp::create(b, op.getResult().getType(),
                                   taintAdaptor.getInput());
}

// ICMP (precise): different rules per predicate.
Value instrumentOperation(ImplicitLocOpBuilder &b, comb::ICmpOp op,
                          comb::ICmpOp::Adaptor taintAdaptor) {
  Value a = op.getLhs();
  Value aT = taintAdaptor.getLhs();
  Value bv = op.getRhs();
  Value bT = taintAdaptor.getRhs();
  auto pred = op.getPredicate();
  auto ty = a.getType();
  unsigned width = cast<IntegerType>(ty).getWidth();
  auto ones = getAllOnes(b, ty);

  if (pred == ICmpPredicate::eq || pred == ICmpPredicate::ne) {
    Value combined = comb::OrOp::create(b, aT, bT);
    Value hasTaint = orReduce(b, combined);
    Value mask = comb::XorOp::create(b, combined, ones); // ~(a_t | b_t)
    Value maskedA = comb::AndOp::create(b, a, mask);
    Value maskedB = comb::AndOp::create(b, bv, mask);
    Value eqUntainted =
        comb::ICmpOp::create(b, ICmpPredicate::eq, maskedA, maskedB);
    return comb::AndOp::create(b, hasTaint, eqUntainted);
  }

  bool isSigned = pred == ICmpPredicate::slt || pred == ICmpPredicate::sle ||
                  pred == ICmpPredicate::sgt || pred == ICmpPredicate::sge;

  Value notAT = comb::XorOp::create(b, aT, ones);
  Value notBT = comb::XorOp::create(b, bT, ones);

  Value minA, maxA, minB, maxB;
  if (width == 1 || !isSigned) {
    minA = comb::AndOp::create(b, a, notAT);
    maxA = comb::OrOp::create(b, a, aT);
    minB = comb::AndOp::create(b, bv, notBT);
    maxB = comb::OrOp::create(b, bv, bT);
  } else {
    auto lsbTy = IntegerType::get(b.getContext(), width - 1);
    auto bitTy = IntegerType::get(b.getContext(), 1);

    Value aLsbs = comb::ExtractOp::create(b, lsbTy, a, 0);
    Value aMsb = comb::ExtractOp::create(b, bitTy, a, width - 1);
    Value aTLsbs = comb::ExtractOp::create(b, lsbTy, aT, 0);
    Value aTMsb = comb::ExtractOp::create(b, bitTy, aT, width - 1);

    Value bLsbs = comb::ExtractOp::create(b, lsbTy, bv, 0);
    Value bMsb = comb::ExtractOp::create(b, bitTy, bv, width - 1);
    Value bTLsbs = comb::ExtractOp::create(b, lsbTy, bT, 0);
    Value bTMsb = comb::ExtractOp::create(b, bitTy, bT, width - 1);

    auto lsbOnes = getAllOnes(b, lsbTy);
    Value notATLsbs = comb::XorOp::create(b, aTLsbs, lsbOnes);
    Value notBTLsbs = comb::XorOp::create(b, bTLsbs, lsbOnes);

    auto bitOnes = getAllOnes(b, bitTy);
    Value notATMsb = comb::XorOp::create(b, aTMsb, bitOnes);
    Value notBTMsb = comb::XorOp::create(b, bTMsb, bitOnes);

    Value minALsbs = comb::AndOp::create(b, aLsbs, notATLsbs);
    Value maxALsbs = comb::OrOp::create(b, aLsbs, aTLsbs);
    Value minBLsbs = comb::AndOp::create(b, bLsbs, notBTLsbs);
    Value maxBLsbs = comb::OrOp::create(b, bLsbs, bTLsbs);

    Value minAMsb = comb::OrOp::create(b, aMsb, aTMsb);
    Value maxAMsb = comb::AndOp::create(b, aMsb, notATMsb);
    Value minBMsb = comb::OrOp::create(b, bMsb, bTMsb);
    Value maxBMsb = comb::AndOp::create(b, bMsb, notBTMsb);

    minA = comb::ConcatOp::create(b, minAMsb, minALsbs);
    maxA = comb::ConcatOp::create(b, maxAMsb, maxALsbs);
    minB = comb::ConcatOp::create(b, minBMsb, minBLsbs);
    maxB = comb::ConcatOp::create(b, maxBMsb, maxBLsbs);
  }

  Value cmp1 = comb::ICmpOp::create(b, pred, minA, maxB);
  Value cmp2 = comb::ICmpOp::create(b, pred, maxA, minB);
  return comb::XorOp::create(b, cmp1, cmp2);
}

// Precise shift taint: two-phase approach matching Yosys CellIFT.
// Phase 1: shift by untainted portion of the shift amount.
// Phase 2: for each possible delta from tainted shift bits, check if the
//          output bit values would differ.
Value shiftTaintPrecise(ImplicitLocOpBuilder &b, Value data, Value dataT,
                        Value amt, Value amtT, Type resTy, bool left,
                        bool arith) {
  unsigned W = cast<IntegerType>(resTy).getWidth();
  if (W <= 1) {
    SmallVector<Value> taintInputs{dataT, amtT};
    return conservativeTaint(b, taintInputs, resTy);
  }

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

Value instrumentOperation(ImplicitLocOpBuilder &b, comb::ShlOp op,
                          comb::ShlOp::Adaptor taintAdaptor) {
  return shiftTaintPrecise(b, op.getLhs(), taintAdaptor.getLhs(), op.getRhs(),
                           taintAdaptor.getRhs(), op.getType(), true, false);
}

Value instrumentOperation(ImplicitLocOpBuilder &b, comb::ShrUOp op,
                          comb::ShrUOp::Adaptor taintAdaptor) {
  return shiftTaintPrecise(b, op.getLhs(), taintAdaptor.getLhs(), op.getRhs(),
                           taintAdaptor.getRhs(), op.getType(), false, false);
}

Value instrumentOperation(ImplicitLocOpBuilder &b, comb::ShrSOp op,
                          comb::ShrSOp::Adaptor taintAdaptor) {
  return shiftTaintPrecise(b, op.getLhs(), taintAdaptor.getLhs(), op.getRhs(),
                           taintAdaptor.getRhs(), op.getType(), false, true);
}

// PARITY: y_t = OR-reduce(input_t)
Value instrumentOperation(ImplicitLocOpBuilder &b, comb::ParityOp,
                          comb::ParityOp::Adaptor taintAdaptor) {
  return orReduce(b, taintAdaptor.getInput());
}

// REVERSE: y_t = reverse(input_t)
Value instrumentOperation(ImplicitLocOpBuilder &b, comb::ReverseOp op,
                          comb::ReverseOp::Adaptor taintAdaptor) {
  return comb::ReverseOp::create(b, b.getLoc(), op.getType(),
                                 taintAdaptor.getInput());
}

//===----------------------------------------------------------------------===//
// Instance Instrumentation
//===----------------------------------------------------------------------===//

void instrumentInstance(TaintMap &taintOf, StringRef taintSuffix,
                        hw::InstanceOp op, ImplicitLocOpBuilder &b) {
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

} // namespace

//===----------------------------------------------------------------------===//
// Module Signature Rewriting
//===----------------------------------------------------------------------===//

void CellIFTInstrumentPass::rewriteModuleSignature(HWModuleOp mod,
                                                   StringRef taintSuffix) {
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

void CellIFTInstrumentPass::instrumentModuleBody(HWModuleOp mod,
                                                 TaintMap &taintOf,
                                                 bool taintConstants,
                                                 StringRef taintSuffix) {
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
        .Case<hw::ConstantOp>([&](auto o) {
          taintOf[o.getResult()] = instrumentOperation(b, o, taintConstants);
        })
        .Case<comb::AndOp, comb::OrOp, comb::XorOp, comb::AddOp, comb::SubOp,
              comb::MulOp, comb::DivUOp, comb::DivSOp, comb::ModUOp,
              comb::ModSOp, comb::MuxOp, comb::ConcatOp, comb::ExtractOp,
              comb::ReplicateOp, comb::ICmpOp, comb::ShlOp, comb::ShrUOp,
              comb::ShrSOp, comb::ParityOp, comb::ReverseOp>(
            [&](auto o) { instrumentSingleResultOperation(taintOf, b, o); })
        .Case<hw::InstanceOp>(
            [&](auto o) { instrumentInstance(taintOf, taintSuffix, o, b); })
        .Default([&](Operation *o) {
          // Conservative fallback for unknown ops with integer results.
          for (auto res : o->getResults())
            if (isa<IntegerType>(res.getType()))
              taintOf[res] = conservativeTaint(b, taintOf, o->getOperands(),
                                               res.getType());
        });
  }

  // Phase 3: Fix up taint register inputs now that all taints are computed.
  for (auto &pr : pendingRegs) {
    Value inputT = getTaint(taintOf, pr.origInput);
    if (auto compreg = dyn_cast<seq::CompRegOp>(pr.taintReg))
      compreg.getInputMutable().assign(inputT);
    else if (auto firreg = dyn_cast<seq::FirRegOp>(pr.taintReg))
      firreg.getNextMutable().assign(inputT);
  }
}

void CellIFTInstrumentPass::rewriteOutputOp(HWModuleOp mod, TaintMap &taintOf,
                                            StringRef taintSuffix) {
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

  std::string taintSuffix = this->taintSuffix;
  bool taintConstants = this->taintConstants;

  for (auto mod : modules)
    rewriteModuleSignature(mod, taintSuffix);

  parallelForEach(&getContext(), modules, [&](HWModuleOp mod) {
    TaintMap taintOf;
    instrumentModuleBody(mod, taintOf, taintConstants, taintSuffix);
    rewriteOutputOp(mod, taintOf, taintSuffix);
  });
}

//===----------------------------------------------------------------------===//
// Registration
//===----------------------------------------------------------------------===//

namespace circt {
std::unique_ptr<mlir::Pass> createCellIFTInstrumentPass() {
  return std::make_unique<CellIFTInstrumentPass>();
}
} // namespace circt
