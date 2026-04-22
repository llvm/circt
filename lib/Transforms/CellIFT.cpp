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
//   F. Solt, B. Gras., and K. Razavi., "CellIFT: Leveraging Cells for
//   Scalable and Precise Dynamic Information Flow Tracking in RTL,"
//   USENIX Security 2022.
//
//===----------------------------------------------------------------------===//

#include "circt/Dialect/Comb/CombOps.h"
#include "circt/Dialect/HW/HWOps.h"
#include "circt/Dialect/HW/HWTypes.h"
#include "circt/Dialect/Seq/SeqOps.h"
#include "circt/Support/BackedgeBuilder.h"
#include "circt/Transforms/Passes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/ImplicitLocOpBuilder.h"
#include "mlir/IR/Threading.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringSet.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/Debug.h"

#include <type_traits>

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
using PendingTaintMap = DenseMap<Value, Backedge>;

struct ModulePortTaintInfo {
  SmallVector<unsigned> taintInputSources;
  SmallVector<unsigned> taintOutputSources;
};

struct ModuleInstrumentationInfo {
  HWModuleOp mod;
  ModulePortTaintInfo portInfo;
};

Value getZero(ImplicitLocOpBuilder &b, Type ty);
Value getAllOnes(ImplicitLocOpBuilder &b, Type ty);
Value orReduce(ImplicitLocOpBuilder &b, Value v);
Value broadcast(ImplicitLocOpBuilder &b, Value bit, Type ty);
Value getTaint(TaintMap &taintOf, PendingTaintMap &pendingTaints,
               BackedgeBuilder &backedgeBuilder, Value v);
void setTaint(TaintMap &taintOf, PendingTaintMap &pendingTaints, Value v,
              Value taint);
Value conservativeTaint(ImplicitLocOpBuilder &b, ValueRange taintInputs,
                        Type resTy);
Value conservativeTaint(ImplicitLocOpBuilder &b, TaintMap &taintOf,
                        PendingTaintMap &pendingTaints,
                        BackedgeBuilder &backedgeBuilder, ValueRange inputs,
                        Type resTy);

template <typename ShiftOp>
using EnableIfCombShiftOp = std::enable_if_t<
    std::is_same_v<ShiftOp, comb::ShlOp> ||
    std::is_same_v<ShiftOp, comb::ShrUOp> ||
    std::is_same_v<ShiftOp, comb::ShrSOp>>;

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
template <typename ShiftOp, typename = EnableIfCombShiftOp<ShiftOp>>
Value instrumentOperation(ImplicitLocOpBuilder &b, ShiftOp op,
                          typename ShiftOp::Adaptor taintAdaptor);
Value instrumentOperation(ImplicitLocOpBuilder &b, comb::ParityOp op,
                          comb::ParityOp::Adaptor taintAdaptor);
Value instrumentOperation(ImplicitLocOpBuilder &b, comb::ReverseOp op,
                          comb::ReverseOp::Adaptor taintAdaptor);

template <typename OpT>
void instrumentSingleResultOperation(TaintMap &taintOf,
                                     PendingTaintMap &pendingTaints,
                                     BackedgeBuilder &backedgeBuilder,
                                     ImplicitLocOpBuilder &b, OpT op) {
  SmallVector<Value> taintOperands;
  taintOperands.reserve(op->getNumOperands());
  for (auto operand : op->getOperands())
    taintOperands.push_back(
        getTaint(taintOf, pendingTaints, backedgeBuilder, operand));
  typename OpT::Adaptor taintAdaptor(taintOperands, op->getAttrDictionary());
  setTaint(taintOf, pendingTaints, op.getResult(),
           instrumentOperation(b, op, taintAdaptor));
}

void instrumentInstance(TaintMap &taintOf, PendingTaintMap &pendingTaints,
                        BackedgeBuilder &backedgeBuilder, StringRef taintSuffix,
                        hw::InstanceOp op, ImplicitLocOpBuilder &b);

class CellIFTInstrumentPass
    : public circt::impl::CellIFTInstrumentBase<CellIFTInstrumentPass> {
public:
  using CellIFTInstrumentBase::CellIFTInstrumentBase;
  void runOnOperation() override;

private:
  // ---- Module-level passes ----------------------------------------------
  LogicalResult rewriteModuleSignature(HWModuleOp mod, StringRef taintSuffix,
                                       ModulePortTaintInfo &portInfo);
  LogicalResult instrumentModuleBody(HWModuleOp mod,
                                     const ModulePortTaintInfo &portInfo,
                                     TaintMap &taintOf, bool taintConstants,
                                     StringRef taintSuffix);
  void rewriteOutputOp(HWModuleOp mod, const ModulePortTaintInfo &portInfo,
                       TaintMap &taintOf);
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

Value getTaint(TaintMap &taintOf, PendingTaintMap &pendingTaints,
               BackedgeBuilder &backedgeBuilder, Value v) {
  assert(isa<IntegerType>(v.getType()) && "can only taint integer values");
  auto it = taintOf.find(v);
  if (it != taintOf.end())
    return it->second;

  auto backedgeIt =
      pendingTaints.try_emplace(v, backedgeBuilder.get(v.getType())).first;
  Value placeholder = backedgeIt->second;
  taintOf[v] = placeholder;
  return placeholder;
}

void setTaint(TaintMap &taintOf, PendingTaintMap &pendingTaints, Value v,
              Value taint) {
  if (auto it = pendingTaints.find(v); it != pendingTaints.end()) {
    it->second.setValue(taint);
    pendingTaints.erase(it);
  }
  taintOf[v] = taint;
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

Value conservativeTaint(ImplicitLocOpBuilder &b, TaintMap &taintOf,
                        PendingTaintMap &pendingTaints,
                        BackedgeBuilder &backedgeBuilder, ValueRange inputs,
                        Type resTy) {
  SmallVector<Value> taintInputs;
  taintInputs.reserve(inputs.size());
  for (auto input : inputs)
    if (isa<IntegerType>(input.getType()))
      taintInputs.push_back(
          getTaint(taintOf, pendingTaints, backedgeBuilder, input));
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

// MOD (conservative): any tainted input taints the full result.
Value instrumentOperation(ImplicitLocOpBuilder &b, comb::ModUOp op,
                          comb::ModUOp::Adaptor taintAdaptor) {
  SmallVector<Value> taintInputs{taintAdaptor.getLhs(), taintAdaptor.getRhs()};
  return conservativeTaint(b, taintInputs, op.getType());
}

Value instrumentOperation(ImplicitLocOpBuilder &b, comb::ModSOp op,
                          comb::ModSOp::Adaptor taintAdaptor) {
  SmallVector<Value> taintInputs{taintAdaptor.getLhs(), taintAdaptor.getRhs()};
  return conservativeTaint(b, taintInputs, op.getType());
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

// ICMP: precise rules per supported predicate with conservative fallback.
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

  if (pred == ICmpPredicate::ceq || pred == ICmpPredicate::cne ||
      pred == ICmpPredicate::weq || pred == ICmpPredicate::wne) {
    op.emitWarning() << "falling back to conservative taint propagation for "
                     << stringifyICmpPredicate(pred) << " predicate";
    SmallVector<Value, 2> taintInputs{aT, bT};
    return conservativeTaint(b, taintInputs, op.getType());
  }

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

template <typename ShiftOp>
Value shiftTaintImprecise(ImplicitLocOpBuilder &b, Value dataT, Value amt,
                          Value amtT, Type resTy) {
  Value shiftAmtTaint = broadcast(b, orReduce(b, amtT), resTy);
  Value shiftedDataT = ShiftOp::create(b, dataT, amt);
  return comb::OrOp::create(b, shiftAmtTaint, shiftedDataT);
}

template <typename ShiftOp, typename>
Value instrumentOperation(ImplicitLocOpBuilder &b, ShiftOp op,
                          typename ShiftOp::Adaptor taintAdaptor) {
  return shiftTaintImprecise<ShiftOp>(b, taintAdaptor.getLhs(), op.getRhs(),
                                      taintAdaptor.getRhs(), op.getType());
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

void instrumentInstance(TaintMap &taintOf, PendingTaintMap &pendingTaints,
                        BackedgeBuilder &backedgeBuilder, StringRef taintSuffix,
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

    newInputs.push_back(
        getTaint(taintOf, pendingTaints, backedgeBuilder, input));
    newArgNames.push_back(
        b.getStringAttr(name.getValue().str() + taintSuffix.str()));
  }

  for (auto [result, nameAttr] :
       llvm::zip_equal(op.getResults(), op.getOutputNames())) {
    auto name = cast<StringAttr>(nameAttr);
    newResTys.push_back(result.getType());
    newResNames.push_back(name);

    if (!isa<IntegerType>(result.getType()))
      continue;

    newResTys.push_back(result.getType());
    newResNames.push_back(
        b.getStringAttr(name.getValue().str() + taintSuffix.str()));
  }

  auto newInst = hw::InstanceOp::create(
      b, op.getLoc(), newResTys, op.getInstanceNameAttr(),
      op.getModuleNameAttr(), newInputs, b.getArrayAttr(newArgNames),
      b.getArrayAttr(newResNames), op.getParametersAttr(), op.getInnerSymAttr(),
      op.getDoNotPrintAttr());

  unsigned newResIdx = 0;
  for (auto result : op.getResults()) {
    Value newResult = newInst.getResult(newResIdx++);
    if (isa<IntegerType>(result.getType())) {
      Value newTaint = newInst.getResult(newResIdx++);
      setTaint(taintOf, pendingTaints, result, newTaint);
      taintOf[newResult] = newTaint;
    }
    result.replaceAllUsesWith(newResult);
  }

  op.erase();
}

} // namespace

//===----------------------------------------------------------------------===//
// Module Signature Rewriting
//===----------------------------------------------------------------------===//

LogicalResult CellIFTInstrumentPass::rewriteModuleSignature(
    HWModuleOp mod, StringRef taintSuffix, ModulePortTaintInfo &portInfo) {
  auto *ctx = mod.getContext();
  auto portList = mod.getPortList();
  Block *body = mod.getBodyBlock();

  SmallVector<std::pair<unsigned, PortInfo>> insIns, insOuts;
  struct BlockArgInsertion {
    unsigned index;
    Type type;
    Location loc;
  };
  SmallVector<BlockArgInsertion> blockArgInsertions;
  llvm::StringSet<> portNames;
  for (auto port : portList)
    portNames.insert(port.getName());

  portInfo.taintInputSources.clear();
  portInfo.taintOutputSources.clear();

  // Insert a taint port after every integer-typed input/output.
  // Also insert block arguments for the new input ports.
  unsigned inIdx = 0, outIdx = 0;
  unsigned blockArgInsertionOffset = 0;
  for (auto port : portList) {
    if (port.isInput()) {
      if (isa<IntegerType>(port.type)) {
        std::string taintName = port.getName().str() + taintSuffix.str();
        if (!portNames.insert(taintName).second) {
          mod.emitOpError() << "cannot add taint port '" << taintName
                            << "' because the name already exists; pick a "
                               "different taint suffix";
          return failure();
        }

        PortInfo tp;
        tp.name = StringAttr::get(ctx, taintName);
        tp.type = port.type;
        tp.dir = ModulePort::Direction::Input;
        tp.loc = port.loc ? port.loc : UnknownLoc::get(ctx);
        insIns.push_back({inIdx + 1, tp});
        blockArgInsertions.push_back({inIdx + 1 + blockArgInsertionOffset,
                                      tp.type, cast<Location>(tp.loc)});
        portInfo.taintInputSources.push_back(inIdx);
        blockArgInsertionOffset++;
      }
      inIdx++;
    } else if (port.isOutput()) {
      if (isa<IntegerType>(port.type)) {
        std::string taintName = port.getName().str() + taintSuffix.str();
        if (!portNames.insert(taintName).second) {
          mod.emitOpError() << "cannot add taint port '" << taintName
                            << "' because the name already exists; pick a "
                               "different taint suffix";
          return failure();
        }

        PortInfo tp;
        tp.name = StringAttr::get(ctx, taintName);
        tp.type = port.type;
        tp.dir = ModulePort::Direction::Output;
        tp.loc = port.loc ? port.loc : UnknownLoc::get(ctx);
        insOuts.push_back({outIdx + 1, tp});
        portInfo.taintOutputSources.push_back(outIdx);
      }
      outIdx++;
    }
  }

  for (auto &insertion : blockArgInsertions)
    body->insertArgument(insertion.index, insertion.type, insertion.loc);

  mod.modifyPorts(insIns, insOuts, {}, {});
  return success();
}

//===----------------------------------------------------------------------===//
// Body Instrumentation
//===----------------------------------------------------------------------===//

LogicalResult CellIFTInstrumentPass::instrumentModuleBody(
    HWModuleOp mod, const ModulePortTaintInfo &portInfo, TaintMap &taintOf,
    bool taintConstants, StringRef taintSuffix) {
  taintOf.clear();
  Block *body = mod.getBodyBlock();
  PendingTaintMap pendingTaints;

  // Map input block args -> their taint block args.
  for (auto [taintOrdinal, sourceIdx] :
       llvm::enumerate(portInfo.taintInputSources)) {
    unsigned origArgIdx = sourceIdx + taintOrdinal;
    unsigned taintArgIdx = origArgIdx + 1;
    setTaint(taintOf, pendingTaints, body->getArgument(origArgIdx),
             body->getArgument(taintArgIdx));
  }

  ImplicitLocOpBuilder b(mod.getLoc(), mod.getContext());
  BackedgeBuilder backedgeBuilder(b, mod.getLoc());

  for (Operation &op : llvm::make_early_inc_range(*body)) {
    if (isa<hw::OutputOp>(op))
      continue;

    b.setLoc(op.getLoc());
    b.setInsertionPointAfter(&op);

    llvm::TypeSwitch<Operation *>(&op)
        .Case<seq::CompRegOp>([&](auto compreg) {
          StringAttr name;
          if (auto n = compreg.getNameAttr())
            name = b.getStringAttr(n.getValue().str() + taintSuffix.str());

          Value inputT = getTaint(taintOf, pendingTaints, backedgeBuilder,
                                  compreg.getInput());
          Value tReg;
          if (compreg.getReset()) {
            Value rstVal = getZero(b, compreg.getType());
            tReg = seq::CompRegOp::create(b, compreg.getLoc(), inputT,
                                          compreg.getClk(), compreg.getReset(),
                                          rstVal, name);
          } else {
            tReg = seq::CompRegOp::create(b, compreg.getLoc(), inputT,
                                          compreg.getClk(), name);
          }
          setTaint(taintOf, pendingTaints, compreg.getData(), tReg);
        })
        .Case<seq::FirRegOp>([&](auto firreg) {
          StringAttr name =
              b.getStringAttr(firreg.getName().str() + taintSuffix.str());

      Value nextT =
          getTaint(taintOf, pendingTaints, backedgeBuilder, firreg.getNext());
      Value tReg;
      if (firreg.hasReset()) {
        Value rstVal = getZero(b, firreg.getType());
        tReg = seq::FirRegOp::create(
            b, nextT, firreg.getClk(), name, firreg.getReset(), rstVal,
            firreg.getInnerSymAttr(), firreg.getIsAsync());
      } else {
        tReg = seq::FirRegOp::create(b, nextT, firreg.getClk(), name);
      }
      setTaint(taintOf, pendingTaints, firreg.getData(), tReg);
    })
    .Case<hw::ConstantOp>([&](auto o) {
      setTaint(taintOf, pendingTaints, o.getResult(),
              instrumentOperation(b, o, taintConstants));
    })
    .Case<comb::AndOp, comb::OrOp, comb::XorOp, comb::AddOp, comb::SubOp,
          comb::MulOp, comb::DivUOp, comb::DivSOp, comb::ModUOp,
          comb::ModSOp, comb::MuxOp, comb::ConcatOp, comb::ExtractOp,
          comb::ReplicateOp, comb::ICmpOp, comb::ShlOp, comb::ShrUOp,
          comb::ShrSOp, comb::ParityOp, comb::ReverseOp>([&](auto o) {
      instrumentSingleResultOperation(taintOf, pendingTaints,
                                      backedgeBuilder, b, o);
    })
    .Case<hw::InstanceOp>([&](auto o) {
      instrumentInstance(taintOf, pendingTaints, backedgeBuilder,
                        taintSuffix, o, b);
    })
    .Default([&](Operation *o) {
      // Conservative fallback for unknown ops with integer results.
      for (auto res : o->getResults())
        if (isa<IntegerType>(res.getType()))
          setTaint(taintOf, pendingTaints, res,
                  conservativeTaint(b, taintOf, pendingTaints,
                                    backedgeBuilder, o->getOperands(),
                                    res.getType()));
    });
}

  return backedgeBuilder.clearOrEmitError();
}

void CellIFTInstrumentPass::rewriteOutputOp(HWModuleOp mod,
                                            const ModulePortTaintInfo &portInfo,
                                            TaintMap &taintOf) {
  Block *body = mod.getBodyBlock();
  auto outputOp = cast<hw::OutputOp>(body->getTerminator());

  SmallVector<Value> newOuts;
  ImplicitLocOpBuilder b(outputOp.getLoc(), outputOp);

  unsigned nextTaintOutput = 0;
  for (unsigned origOutputIdx = 0, e = outputOp.getNumOperands();
       origOutputIdx < e; ++origOutputIdx) {
    Value origOutput = outputOp.getOperand(origOutputIdx);
    newOuts.push_back(origOutput);

    if (nextTaintOutput < portInfo.taintOutputSources.size() &&
        portInfo.taintOutputSources[nextTaintOutput] == origOutputIdx) {
      newOuts.push_back(taintOf.at(origOutput));
      ++nextTaintOutput;
    }
  }

  hw::OutputOp::create(b, outputOp.getLoc(), newOuts);
  outputOp.erase();
}

//===----------------------------------------------------------------------===//
// Main
//===----------------------------------------------------------------------===//

void CellIFTInstrumentPass::runOnOperation() {
  auto modules =
      llvm::to_vector(getOperation().getBody()->getOps<HWModuleOp>());

  std::string taintSuffix = this->taintSuffix;
  bool taintConstants = this->taintConstants;

  SmallVector<ModuleInstrumentationInfo> moduleInfos;
  moduleInfos.reserve(modules.size());

  for (auto mod : modules) {
    ModuleInstrumentationInfo info{mod, {}};
    if (failed(rewriteModuleSignature(mod, taintSuffix, info.portInfo))) {
      signalPassFailure();
      return;
    }
    moduleInfos.push_back(std::move(info));
  }

  if (failed(failableParallelForEach(
          &getContext(), moduleInfos,
          [&](ModuleInstrumentationInfo &info) -> LogicalResult {
            TaintMap taintOf;
            if (failed(instrumentModuleBody(info.mod, info.portInfo, taintOf,
                                            taintConstants, taintSuffix)))
              return failure();
            rewriteOutputOp(info.mod, info.portInfo, taintOf);
            return success();
          })))
    signalPassFailure();
}
