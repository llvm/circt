//===- FSMToSMT.cpp -----------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#include "circt/Conversion/FSMToSMT.h"
#include "circt/Dialect/Comb/CombOps.h"
#include "circt/Dialect/Comb/CombOps.h"
#include "circt/Dialect/FSM/FSMOps.h"
#include "circt/Dialect/FSM/FSMOps.h"
#include "circt/Dialect/HW/HWOps.h"
#include "circt/Dialect/HW/HWOps.h"
#include "circt/Dialect/Verif/VerifOps.h"
#include "circt/Dialect/Verif/VerifOps.h"
#include "circt/Support/BackedgeBuilder.h"
#include "circt/Support/BackedgeBuilder.h"
#include "mlir/Conversion/ReconcileUnrealizedCasts/ReconcileUnrealizedCasts.h"
#include "mlir/Dialect/SMT/IR/SMTOps.h"
#include "mlir/Dialect/SMT/IR/SMTOps.h"
#include "mlir/Dialect/SMT/IR/SMTTypes.h"
#include "mlir/Dialect/SMT/IR/SMTTypes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/Region.h"
#include "mlir/IR/Value.h"
#include "mlir/IR/ValueRange.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/DialectConversion.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/raw_ostream.h"
#include <algorithm>
#include <cassert>
#include <circt/Dialect/HW/HWTypes.h>
#include <cstdlib>
#include <memory>
#include <string>
#include <string>
#include <utility>


namespace circt {
#define GEN_PASS_DEF_CONVERTFSMTOSMT
#include "circt/Conversion/Passes.h.inc"
} // namespace circt

using namespace mlir;
using namespace circt;
using namespace fsm;

//===----------------------------------------------------------------------===//
// Convert FSM to SMT pass
//===----------------------------------------------------------------------===//

namespace {

struct LoweringConfig {
  // If true, represent all non-boolean values as !smt.bv<w>. Otherwise, use
  // !smt.int for widths > 1, and !smt.bool for width == 1.
  bool useBitVec = false;

  // If true, include a "time" parameter in the relation and increment it on
  // transitions. If false, omit it entirely.
  bool withTime = false;

  // Bit-width for time when useBitVec is true. Ignored in int mode.
  unsigned timeWidth = 5;
};


class MachineOpConverter {
public:
  MachineOpConverter(OpBuilder &builder, MachineOp machineOp,
                     const LoweringConfig &cfg)
      : machineOp(machineOp), b(builder), cfg(cfg) {}
  LogicalResult dispatch();

private:
 // Build the integer constant 2^exp.
  Value intPow2(unsigned exp, Location loc) {
    unsigned cstBits = exp + 1;
    llvm::APInt ap(cstBits, 1);
    ap = ap.shl(exp);
    auto attr = b.getIntegerAttr(b.getIntegerType(cstBits), ap);
    return b.create<smt::IntConstantOp>(loc, attr);
  }
  // Helpers (mode-aware).
  Type numTypeForWidth(unsigned w) {
    if (cfg.useBitVec)
      return b.getType<smt::BitVectorType>(w);
    // int-mode: width==1 => Bool, else Int
    if (w == 1)
      return b.getType<smt::BoolType>();
    return b.getType<smt::IntType>();
  }

  Value zeroConst(unsigned w, Location loc) {
    if (cfg.useBitVec)
      return b.create<smt::BVConstantOp>(loc, 0, w);
    if (w == 1)
      return b.create<smt::BoolConstantOp>(loc, false);
    auto attr = b.getIntegerAttr(b.getIntegerType(std::max(1u, w)), 0);
    return b.create<smt::IntConstantOp>(loc, attr);
  }

  Value oneConst(unsigned w, Location loc) {
    if (cfg.useBitVec)
      return b.create<smt::BVConstantOp>(loc, 1, w);
    if (w == 1)
      return b.create<smt::BoolConstantOp>(loc, true);
    auto attr = b.getIntegerAttr(b.getIntegerType(std::max(1u, w)), 1);
    return b.create<smt::IntConstantOp>(loc, attr);
  }

  // BV-only: (!smt.bv<1> -> !smt.bool). If already Bool, return as-is.
  static Value bv1ToBool(OpBuilder &b, Location loc, Value v) {
    if (auto bvTy = llvm::dyn_cast<smt::BitVectorType>(v.getType())) {
      if (bvTy.getWidth() == 1) {
        auto one = b.create<smt::BVConstantOp>(loc, 1, 1);
        return b.create<smt::EqOp>(loc, v, one);
      }
    }
    if (llvm::isa<smt::BoolType>(v.getType()))
      return v;
    v.getDefiningOp()->emitError()
        << "bv1ToBool expected !smt.bv<1> or !smt.bool, got " << v;
    assert(false && "bv1ToBool type mismatch");
    return v;
  }

  // BV-only: (!smt.bool -> !smt.bv<1>). If already BV<1>, return as-is.
  static Value boolToBV1(OpBuilder &b, Location loc, Value pred) {
    if (llvm::isa<smt::BoolType>(pred.getType())) {
      auto one = b.create<smt::BVConstantOp>(loc, 1, 1);
      auto zero = b.create<smt::BVConstantOp>(loc, 0, 1);
      return b.create<smt::IteOp>(loc, b.getType<smt::BitVectorType>(1), pred,
                                  one, zero);
    }
    if (auto bvTy = llvm::dyn_cast<smt::BitVectorType>(pred.getType()))
      if (bvTy.getWidth() == 1)
        return pred;
    pred.getDefiningOp()->emitError()
        << "boolToBV1 expected !smt.bool or !smt.bv<1>, got " << pred;
    assert(false && "boolToBV1 type mismatch");
    return pred;
  }

  // Mode-aware: !smt.bool -> numeric (BV<1> or Int(0/1))
  Value boolToNumeric(Value v, Location loc) {
    if (!llvm::isa<smt::BoolType>(v.getType()))
      return v;
    if (cfg.useBitVec)
      return boolToBV1(b, loc, v);
    // int mode: ite(bool, 1, 0)
    auto t1 = oneConst(/*w*/ 2, loc);  // width ignored in int mode
    auto t0 = zeroConst(/*w*/ 2, loc); // width ignored in int mode
    // In Int mode, Ite result type is !smt.int if arms are Int
    return b.create<smt::IteOp>(loc, b.getType<smt::IntType>(), v, t1, t0);
  }

  // Mode-aware: numeric (BV<1> or Int) -> Bool
  Value numericToBool(Value v, Location loc) {
    if (llvm::isa<smt::BoolType>(v.getType()))
      return v;
    if (auto bvTy = llvm::dyn_cast<smt::BitVectorType>(v.getType())) {
      if (bvTy.getWidth() == 1) {
        auto one = b.create<smt::BVConstantOp>(loc, 1, 1);
        return b.create<smt::EqOp>(loc, v, one);
      }
      v.getDefiningOp()->emitError()
          << "expected !smt.bv<1> for numericToBool, got " << v;
      assert(false && "non-1-width BV to bool");
    }
    // int mode: v != 0
    auto z = oneConst(/*w*/ 2, loc);
    return b.create<smt::EqOp>(loc, v, z);
  }
  
  // Type width computation for lowering aggregates.
  static unsigned getPackedBitWidth(Type t) {
    if (auto intTy = llvm::dyn_cast<IntegerType>(t))
      return intTy.getIntOrFloatBitWidth();

    if (auto bvTy = llvm::dyn_cast<smt::BitVectorType>(t))
      return bvTy.getWidth();

    if (auto arrTy = llvm::dyn_cast<hw::ArrayType>(t)) {
      unsigned elemW = getPackedBitWidth(arrTy.getElementType());
      return elemW * arrTy.getNumElements();
    }

    if (auto structTy = llvm::dyn_cast<hw::StructType>(t)) {
      unsigned w = 0;
      for (auto elem : structTy.getElements())
        w += getPackedBitWidth(elem.type);
      return w;
    }

    llvm::errs() << "Unsupported type for bitwidth computation in FSMToSMT: "
                 << t << "\n";
    assert(false && "Unsupported type in getPackedBitWidth");
    return 0;
  }

  smt::BVCmpPredicate getSmtBVPred(comb::ICmpPredicate p) {
    switch (p) {
    case comb::ICmpPredicate::slt:
      return smt::BVCmpPredicate::slt;
    case comb::ICmpPredicate::sle:
      return smt::BVCmpPredicate::sle;
    case comb::ICmpPredicate::sgt:
      return smt::BVCmpPredicate::sgt;
    case comb::ICmpPredicate::sge:
      return smt::BVCmpPredicate::sge;
    case comb::ICmpPredicate::ult:
      return smt::BVCmpPredicate::ult;
    case comb::ICmpPredicate::ule:
      return smt::BVCmpPredicate::ule;
    case comb::ICmpPredicate::ugt:
      return smt::BVCmpPredicate::ugt;
    case comb::ICmpPredicate::uge:
      return smt::BVCmpPredicate::uge;
    }
    assert(false && "unsupported comparison predicate");
  }
  

  // In int mode, map both signed/unsigned preds to Int preds (best-effort).
  smt::IntPredicate getSmtIntPred(comb::ICmpPredicate p) {
    switch (p) {
    case comb::ICmpPredicate::slt:
    case comb::ICmpPredicate::ult:
      return smt::IntPredicate::lt;
    case comb::ICmpPredicate::sle:
    case comb::ICmpPredicate::ule:
      return smt::IntPredicate::le;
    case comb::ICmpPredicate::sgt:
    case comb::ICmpPredicate::ugt:
      return smt::IntPredicate::gt;
    case comb::ICmpPredicate::sge:
    case comb::ICmpPredicate::uge:
      return smt::IntPredicate::ge;
    default:
      assert(false && "eq/ne handled elsewhere");
    }
  }

  // Build SMT for a comb op (mode-aware).
  Value getCombValue(Operation &op, Location &loc,
                     SmallVector<Value> args) {
    auto toBV = [&](Value v) -> Value {
      if (isa<smt::IntType>(v.getType()))
        return b.create<smt::Int2BVOp>(loc, b.getType<smt::BitVectorType>(64), v);
      if (auto bvTy = llvm::dyn_cast<smt::BitVectorType>(v.getType()))
        return v;
      if (llvm::isa<smt::BoolType>(v.getType()))
        return boolToBV1(b, loc, v);
    };

    auto toInt = [&](Value v) -> Value {
      if (llvm::isa<smt::IntType>(v.getType()))
        return v;
      if (llvm::isa<smt::BoolType>(v.getType()))
        return boolToNumeric(v, loc); // ite(bool,1,0) -> Int
      op.emitError() << "expected SMT Int or Bool operand, got " << v;
      assert(false && "unexpected SMT operand type");
      return v;
    };

    auto toBool = [&](Value v) -> Value {
      if (llvm::isa<smt::BoolType>(v.getType()))
        return v;
      return numericToBool(v, loc);
    };

    auto bvWidth = [&](Value v) -> int {
      if (auto bvTy = llvm::dyn_cast<smt::BitVectorType>(v.getType()))
        return bvTy.getWidth();
      if (llvm::isa<smt::BoolType>(v.getType()))
        return 1;
      op.emitError() << "expected SMT BV or Bool operand, got " << v;
      assert(false && "unexpected SMT operand type");
      return 1;
    };

    // widths for BV ops only (for result typing)
    SmallVector<int> widths;
    if (cfg.useBitVec) {
      for (auto arg : args) {
        if (auto bvTy = llvm::dyn_cast<smt::BitVectorType>(arg.getType())) {
          widths.push_back(bvTy.getWidth());
        } else if (llvm::isa<smt::BoolType>(arg.getType())) {
          widths.push_back(1);
        } else {
          op.emitError() << "getCombValue received a non-SMT value: " << arg;
          assert(false && "Non-SMT value passed to getCombValue");
        }
      }
    }

    // comb.add
    if (auto addOp = dyn_cast<comb::AddOp>(op)) {
      // bitvec
      if (cfg.useBitVec) {
        return b.create<smt::BVAddOp>(loc, b.getType<smt::BitVectorType>(widths[0]),
                                      args);
      }
   // int (wraparound: mod 2^w)
   SmallVector<Value> ops;
   for (auto v : args) ops.push_back(toInt(v));
   Value sum = b.create<smt::IntAddOp>(loc, ops);
   unsigned w = cast<IntegerType>(addOp.getType()).getIntOrFloatBitWidth();
   Value modulus = intPow2(w, loc); // 2^w as !smt.int
   return b.create<smt::IntModOp>(loc, sum, modulus);
    }

    // comb.sub
    if (auto subOp = dyn_cast<comb::SubOp>(op)) {
      // bitvec
      if (cfg.useBitVec) {
        Value rhs = toBV(args[1]);
        auto rhsTy = llvm::cast<smt::BitVectorType>(rhs.getType());
        Value neg = b.create<smt::BVNegOp>(loc, rhsTy, rhs);
        return b.create<smt::BVAddOp>(loc,
               b.getType<smt::BitVectorType>(widths[0]), args[0], neg);
      }
      // int ((2^w + a) - b) % 2^w
      SmallVector<Value> ops;
      for (auto v : args) ops.push_back(toInt(v));
      unsigned w = cast<IntegerType>(subOp.getType()).getIntOrFloatBitWidth();
      Value modulus = intPow2(w, loc); 
      Value add = b.create<smt::IntSubOp>(loc, modulus, ops[0]);
      Value sub = b.create<smt::IntSubOp>(loc, add, ops[1]);
      return b.create<smt::IntModOp>(loc, sub, modulus);
    }

    // comb.mul
    if (auto mulOp = dyn_cast<comb::MulOp>(op)) {
      // bitvec
      if (cfg.useBitVec) {
        return b.create<smt::BVMulOp>(loc, b.getType<smt::BitVectorType>(widths[0]),
                                      args);
      }
      // int 
      SmallVector<Value> ops;
      for (auto v : args) ops.push_back(toInt(v));
      auto mul = b.create<smt::IntMulOp>(loc, ops);
      unsigned w = cast<IntegerType>(mulOp.getType()).getIntOrFloatBitWidth();
      Value modulus = intPow2(w, loc); 
      return b.create<smt::IntModOp>(loc, mul, modulus);
    }

    // comb.and (boolean and for i1)
    if (auto andOp = dyn_cast<comb::AndOp>(op)) {
      if (args.size() == 1)
        return args[0];
      // bitvec 
      if (cfg.useBitVec) {
        Value result = toBV(args[0]);
        for (size_t i = 1; i < args.size(); ++i)
          result = b.create<smt::BVAndOp>(loc, result, toBV(args[i]));
        return result;
      }
      // int 
      int width = op.getOperand(0).getType().getIntOrFloatBitWidth();
      if (width== 1){
        SmallVector<Value> bools;
        for (auto v : args) bools.push_back(toBool(v));
        Value result = bools[0];
        for (size_t i = 1; i < bools.size(); ++i) result = b.create<smt::AndOp>(loc, result, bools[i]);
        return result;
      } 
      SmallVector<Value> convertedOps;
      for (auto v : args){
        if ((isa<smt::IntType>(v.getType())))
          convertedOps.push_back(toBV(v));
        else 
          llvm::outs() << "\n\nunsupported comb.and op: " << v;
      }
      Value result = convertedOps[0];
      for (size_t i = 1; i < convertedOps.size(); ++i)
        result = b.create<smt::BVAndOp>(loc, result, convertedOps[i]);
      auto boolRes = b.create<smt::BV2IntOp>(loc, result);
      return boolRes;
    }

    // comb.or (boolean or for i1)
    if (auto orOp = dyn_cast<comb::OrOp>(op)) {
      // bitvec
      if (cfg.useBitVec) {
        Value result = toBV(args[0]);
        for (size_t i = 1; i < args.size(); ++i)
          result = b.create<smt::BVOrOp>(loc, result, toBV(args[i]));
        return result;
      }
      // int
      int width = op.getOperand(0).getType().getIntOrFloatBitWidth();
      if (width== 1){
        SmallVector<Value> bools;
        for (auto v : args) bools.push_back(toBool(v));
        Value result = bools[0];
        for (size_t i = 1; i < bools.size(); ++i)
          result = b.create<smt::OrOp>(loc, result, bools[i]);
        return result;
      } 
      SmallVector<Value> convertedOps;
      for (auto v : args){
        if ((isa<smt::IntType>(v.getType())))
          convertedOps.push_back(toBV(v));
        else 
          llvm::outs() << "\n\nunsupported comb.and op: " << v;
      }
      Value result = convertedOps[0];
      for (size_t i = 1; i < convertedOps.size(); ++i)
        result = b.create<smt::BVOrOp>(loc, result, convertedOps[i]);
      auto boolRes = b.create<smt::BV2IntOp>(loc, result);
      return boolRes;
    }

    // comb.xor (boolean xor for i1)
    if (auto xorOp = dyn_cast<comb::XorOp>(op)) {
      // bitvec
      if (cfg.useBitVec) {
        Value result = toBV(args[0]);
        for (size_t i = 1; i < args.size(); ++i)
          result = b.create<smt::BVXOrOp>(loc, result, toBV(args[i]));
        return result;
      }
      // int
      int width = op.getOperand(0).getType().getIntOrFloatBitWidth();
      if (width== 1){
        SmallVector<Value> bools;
        for (auto v : args) bools.push_back(toBool(v));
        Value result = bools[0];
        for (size_t i = 1; i < bools.size(); ++i)
          result = b.create<smt::XOrOp>(loc, result, bools[i]);
        return result;
      } 
      SmallVector<Value> convertedOps;
      for (auto v : args){
        if ((isa<smt::IntType>(v.getType())))
          convertedOps.push_back(toBV(v));
        else 
          llvm::outs() << "\n\nunsupported comb.and op: " << v;
      }
      Value result = convertedOps[0];
      for (size_t i = 1; i < convertedOps.size(); ++i)
        result = b.create<smt::BVXOrOp>(loc, result, convertedOps[i]);
      auto boolRes = b.create<smt::BV2IntOp>(loc, result);
      return boolRes;
    }

    // comb.mux
    if (auto mux = dyn_cast<comb::MuxOp>(op)) {
      assert(args.size() == 3 && "MuxOp should have 3 arguments");
      Value condBool = toBool(args[0]);
      // Return type is the type of data arms.
      Type resTy = args[1].getType();
      return b.create<smt::IteOp>(loc, resTy, condBool, args[1], args[2]);
    }
    
    // comb.concat
    if (auto concatOp = dyn_cast<comb::ConcatOp>(op)) {
      // bitvec
      if (cfg.useBitVec) {
        Value acc = toBV(args[0]);
        int accW = bvWidth(acc);
        for (size_t i = 1; i < args.size(); ++i) {
          Value next = toBV(args[i]);
          int nextW = bvWidth(next);
          auto resTy = b.getType<smt::BitVectorType>(accW + nextW);
          acc = b.create<smt::ConcatOp>(loc, resTy, acc, next);
          accW += nextW;
        }
        return acc;
      }
      // int: concatenate by "shift-and-add":
      // If acc encodes the left part and next encodes the right part,
      // concat(acc, next) = acc * 2^(width(next)) + next
      // We compute 2^width(next) as a constant Int, since widths are known statically.
      SmallVector<unsigned> opWidths;
      opWidths.reserve(concatOp->getNumOperands());
      for (Value orig : concatOp->getOperands())
        opWidths.push_back(getPackedBitWidth(orig.getType()));
      Value acc = toInt(args[0]);
      for (size_t i = 1; i < args.size(); ++i) {
        unsigned nextW = opWidths[i];
        // Build the Int constant 2^nextW.
        // Use bitwidth (nextW + 1) to be sufficient to hold the value.
        unsigned cstBits = nextW + 1;
        llvm::APInt twoPow(cstBits, 1);
        twoPow = twoPow.shl(nextW);
        auto powAttr = b.getIntegerAttr(b.getIntegerType(cstBits), twoPow);
        Value powC = b.create<smt::IntConstantOp>(loc, powAttr);
        // acc = acc * 2^nextW + toInt(args[i])
        Value scaled = b.create<smt::IntMulOp>(loc, SmallVector<Value>{acc, powC});
        Value nextInt = toInt(args[i]);
        acc = b.create<smt::IntAddOp>(loc, SmallVector<Value>{scaled, nextInt});
      }
      return acc;
    }

    // comb.extract
    if (auto extOp = dyn_cast<comb::ExtractOp>(op)) {
      unsigned low = extOp.getLowBit();
      unsigned width = extOp.getType().getIntOrFloatBitWidth();
      // bitvec 
      if (cfg.useBitVec) {
        auto resTy = b.getType<smt::BitVectorType>(width);
        return b.create<smt::ExtractOp>(loc, resTy, low, toBV(args.front()));
      }
      // int: extract(x, low, width) = ((x div 2^low) mod 2^width)
      Value xInt = toInt(args.front());
      // q = x div 2^low  (skip division if low == 0)
      Value q = xInt;
      if (low != 0)
        q = b.create<smt::IntDivOp>(loc, xInt, intPow2(low, loc));
      if (width == 1) {
        // Return a Bool for i1. Compute the selected bit as Int 0/1 and convert.
        Value bitInt = b.create<smt::IntModOp>(loc, q, intPow2(1, loc)); // mod 2
        return numericToBool(bitInt, loc); // 0 -> false, 1 -> true
      }
      // General case: result is Int in [0, 2^width)
      Value maskRange = intPow2(width, loc); // 2^width
      return b.create<smt::IntModOp>(loc, q, maskRange);
    }

    // comb.replicate
    if (auto repOp = dyn_cast<comb::ReplicateOp>(op)) {
      // bitvec
      if (cfg.useBitVec){
        unsigned count = repOp.getMultiple();
        Value in = toBV(args[0]);
        return b.create<smt::RepeatOp>(loc, count, in);
      }
      // int: we only support replicate of i1, which is lowered as ite(cond, 2^count-1, 0)
      if (repOp.getOperand().getType().getIntOrFloatBitWidth() == 1){
        unsigned count = repOp.getMultiple();
        auto lhs =  b.create<smt::IntConstantOp>(loc, b.getI32IntegerAttr(2 ^ count - 1));
        auto rhs =  b.create<smt::IntConstantOp>(loc, b.getI32IntegerAttr(0));
        auto cond= toBool(args[0]);
        return b.create<smt::IteOp>(loc, b.getType<smt::IntType>(), cond, lhs, rhs);
      } 
      op.emitError() << "replicate is only unsupported in int mode for width == 1";
      assert(false && "replicate needs 1-long bit-vector");
    }

    // comb.shru
    if (comb::ShrUOp shruOp = dyn_cast<comb::ShrUOp>(op)) {
      // bitvec 
      SmallVector<Value> bvArgs; 
      for (auto a : args) bvArgs.push_back(toBV(a));
      auto bvOp = b.create<smt::BVLShrOp>(loc, bvArgs);
      if (cfg.useBitVec) {
        return bvOp;
      }
      // int 
      return b.create<smt::BV2IntOp>(loc, bvOp->getResult(0));
      // return b.create<smt::IntConstantOp>(loc, b.getI32IntegerAttr(0));
    }
  
    // comb.icmp
    if (auto icmp = dyn_cast<comb::ICmpOp>(op)) {
      auto pred = icmp.getPredicate();
      // eq/ne first
      if (pred == comb::ICmpPredicate::eq) {
        // Generic eq works for both Int/BV; result Bool.
        Value eq = b.create<smt::EqOp>(loc, args);
        if (cfg.useBitVec) {
          return boolToBV1(b, icmp.getLoc(), eq);
        }
        return eq; // Bool in int mode
      }
      if (pred == comb::ICmpPredicate::ne) {
        auto dis = b.create<smt::DistinctOp>(loc, args);
        if (cfg.useBitVec) {
          return boolToBV1(b, icmp.getLoc(), dis);
        }
        return dis; // Bool in int mode
      }
      // ordered comparisons
      if (cfg.useBitVec) {
        auto p = getSmtBVPred(pred);
        return b.create<smt::BVCmpOp>(loc, p, toBV(args[0]), toBV(args[1]));
      }
      auto p = getSmtIntPred(pred);
      return b.create<smt::IntCmpOp>(loc, p, toInt(args[0]), toInt(args[1]));
    }

    llvm::outs() << "\n\nunsupported comb op: " << op;
    assert(false && "unsupported comb operation");
    return Value();
  }

  Value getSmtValue(Value v,
                    const SmallVector<std::pair<Value, Value>> &fsmArgVals,
                    Location &loc) {
    for (auto fav : fsmArgVals)
      if (v == fav.first)
        return fav.second;

    if (v.getDefiningOp()->getName().getDialect()->getNamespace() == "comb") {
      SmallVector<Value> combArgs;
      for (auto arg : v.getDefiningOp()->getOperands()) {
        auto lowered = getSmtValue(arg, fsmArgVals, loc);
        combArgs.push_back(lowered);
      }
      return getCombValue(*v.getDefiningOp(), loc, combArgs);
    }

    if (auto cst = dyn_cast<hw::ConstantOp>(v.getDefiningOp())) {
      // Lower constants based on mode and bitwidth.
      auto ap = cst.getValue();
      unsigned w = ap.getBitWidth();
      if (cfg.useBitVec)
        return b.create<smt::BVConstantOp>(loc, ap);
      // int mode: width==1 => Bool, else Int
      if (w == 1)
        return b.create<smt::BoolConstantOp>(loc, ap != 0);
      auto attr = b.getIntegerAttr(b.getIntegerType(w), ap);
      return b.create<smt::IntConstantOp>(loc, attr);
    }

    llvm::outs() << "\n\nunsupported getSmtValue op: " << v;
    return v;
  }

  struct Transition {
    int from;
    int to;
    bool hasGuard = false, hasAction = false, hasOutput = false;
    Region *guard = nullptr, *action = nullptr, *output = nullptr;
  };

  Transition parseTransition(fsm::TransitionOp t, int from,
                             SmallVector<std::string> &states, Location &loc) {
    std::string nextState = t.getNextState().str();
    Transition tr = {.from = from, .to = insertStates(states, nextState)};
    if (!t.getGuard().empty()) {
      tr.hasGuard = true;
      tr.guard = &t.getGuard();
    }
    if (!t.getAction().empty()) {
      tr.hasAction = true;
      tr.action = &t.getAction();
    }
    return tr;
  }

  static int insertStates(SmallVector<std::string> &states, llvm::StringRef st) {
    for (auto [id, s] : llvm::enumerate(states))
      if (s == st)
        return id;
    states.push_back(st.str());   // materialize once, stored in vector
    return states.size() - 1;
  }

  Region *getOutputRegion(
      SmallVector<std::pair<Region *, int>> outputOfStateId, int stateId) {
    for (auto oid : outputOfStateId)
      if (stateId == oid.second)
        return oid.first;
    abort();
  }

  MachineOp machineOp;
  OpBuilder &b;
  LoweringConfig cfg;
};



// Implementation.

LogicalResult MachineOpConverter::dispatch() {
  b.setInsertionPoint(machineOp);
  auto loc = machineOp.getLoc();
  auto machineArgs = machineOp.getArguments();

  SmallVector<Type> argVarTypes; // The SMT types used to quantify.
  SmallVector<Value> argVars;    // The original FSM Values in order: args, outs init, vars

  int numArgs = 0;
  int numOut = 0;

  TypeRange typeRange;
  ValueRange valueRange;

  auto solver = b.create<smt::SolverOp>(loc, typeRange, valueRange);
  solver.getBodyRegion().emplaceBlock();
  b.setInsertionPointToStart(solver.getBody());

  // Arguments.
  for (auto a : machineArgs) {
    unsigned w = getPackedBitWidth(a.getType());
    argVarTypes.push_back(numTypeForWidth(w));
    argVars.push_back(a);
    numArgs++;
  }

  // Outputs (introduce symbolic slots initialized from output region).
  if (!machineOp.getResultTypes().empty()) {
    for (auto o : machineOp.getResultTypes()) {
      unsigned w = getPackedBitWidth(o);
      argVarTypes.push_back(numTypeForWidth(w));
      auto ov = zeroConst(w, loc);
      argVars.push_back(ov);
      numOut++;
    }
  }

  // Variables (and record initial values).
  SmallVector<llvm::APInt> varInitValues;
  for (auto variableOp : machineOp.front().getOps<fsm::VariableOp>()) {
    unsigned w = getPackedBitWidth(variableOp.getType());
    argVarTypes.push_back(numTypeForWidth(w));
    auto initVal = variableOp.getInitValueAttr();
    if (auto intAttr = dyn_cast<IntegerAttr>(initVal))
      varInitValues.push_back(intAttr.getValue());
    else
      varInitValues.emplace_back(1, 0); // default false if missing
    argVars.push_back(variableOp->getOpResult(0));
  }
  size_t numVars = varInitValues.size();

  // Optional time parameter (last).
  if (cfg.withTime) {
    if (cfg.useBitVec)
      argVarTypes.push_back(b.getType<smt::BitVectorType>(cfg.timeWidth));
    else
      argVarTypes.push_back(b.getType<smt::IntType>());
  }

  SmallVector<MachineOpConverter::Transition> transitions;
  SmallVector<Value> stateFunctions;

  SmallVector<std::string> states;
  SmallVector<std::pair<Region *, int>> outputOfStateId;

  // States set.
  std::string initialState = machineOp.getInitialState().str();
  insertStates(states, initialState);

  // Declare one predicate per state: F_state(args, outs, vars, [time]) -> Bool
  for (auto stateOp : machineOp.front().getOps<fsm::StateOp>()) {
    mlir::StringAttr funName = b.getStringAttr(("F_" + stateOp.getName().str()));
    auto range = b.getType<smt::BoolType>();
    auto funTy = b.getType<smt::SMTFuncType>(argVarTypes, range);
    smt::DeclareFunOp acFun = b.create<smt::DeclareFunOp>(loc, funTy, funName);
    stateFunctions.push_back(acFun);
    auto fromState = insertStates(states, stateOp.getName().str());
    outputOfStateId.push_back({&stateOp.getOutput(), fromState});
  }

  // Collect transitions.
  for (auto stateOp : machineOp.front().getOps<fsm::StateOp>()) {
    std::string stateName = stateOp.getName().str();
    auto fromState = insertStates(states, stateName);
    if (!stateOp.getTransitions().empty()) {
      for (auto tr :
           stateOp.getTransitions().front().getOps<fsm::TransitionOp>()) {
        auto t = parseTransition(tr, fromState, states, loc);
        if (!stateOp.getOutput().empty()) {
          t.hasOutput = true;
          t.output = getOutputRegion(outputOfStateId, t.to);
        } else {
          t.hasOutput = false;
        }
        transitions.push_back(t);
      }
    }
  }

  // Gather verif.assert properties we find when visiting output regions.
  struct PendingAssertion {
    int stateId;
    Value predicateFsm; // in FSM domain; lower later in its own forall
  };
  SmallVector<PendingAssertion> assertions;

  // Initial condition: initialize outputs/vars (from output region/initial
  // values) and assert F_init(...). If withTime, guard with (time == 0).
  auto forallInit = b.create<smt::ForallOp>(
      loc, argVarTypes,
      [&](OpBuilder &b, Location loc, SmallVector<Value> forallArgs) -> Value {
        SmallVector<Value> initArgs;         // args to F_0
        SmallVector<Value> outputSmtValues;  // computed outputs at init state
        SmallVector<Value> initVarValues;    // SMT constants for vars

        auto initOutputReg = getOutputRegion(outputOfStateId, 0);

        // Build var init constants in SMT order (variables occupy positions
        // [numArgs + numOut, numArgs + numOut + numVars)).
        for (auto [i, a] : llvm::enumerate(forallArgs)) {
          if (int(i) >= numOut + numArgs &&
              int(i) < int(numOut + numArgs + numVars)) {
            size_t varIdx = i - (numOut + numArgs);
            auto ap = varInitValues[varIdx];

            // Determine SMT type for this quantified symbol.
            Type qt = argVarTypes[i];
            if (auto bvTy = llvm::dyn_cast<smt::BitVectorType>(qt)) {
              assert(ap.getBitWidth() == bvTy.getWidth() &&
                     "init width mismatch for variable");
              initVarValues.push_back(b.create<smt::BVConstantOp>(loc, ap));
            } else if (llvm::isa<smt::BoolType>(qt)) {
              initVarValues.push_back(b.create<smt::BoolConstantOp>(
                  loc, ap != 0));
            } else {
              // IntType
              auto attr =
                  b.getIntegerAttr(b.getIntegerType(ap.getBitWidth()), ap);
              initVarValues.push_back(b.create<smt::IntConstantOp>(loc, attr));
            }
          }
        }

        // Evaluate output region at init to compute outputs (if present).
        if (!initOutputReg->empty()) {
          SmallVector<std::pair<Value, Value>> avToSmt;
          for (auto [i, a] : llvm::enumerate(argVars)) {
            if (int(i) >= numOut + numArgs &&
                int(i) < int(numOut + numArgs + numVars)) {
              avToSmt.push_back({a, initVarValues[i - numOut - numArgs]});
            } else {
              avToSmt.push_back({a, forallArgs[i]});
            }
          }

          for (auto &op : initOutputReg->getOps()) {
            if (auto outOp = dyn_cast<fsm::OutputOp>(op)) {
              for (auto outs : outOp->getOperands()) {
                bool found = false;
                for (auto [i, fav] : llvm::enumerate(avToSmt)) {
                  if (outs == fav.first && i < numArgs) {
                    outputSmtValues.push_back(forallArgs[i]);
                    found = true;
                  }
                }
                if (!found) {
                  auto v = getSmtValue(outs, avToSmt, loc);
                  outputSmtValues.push_back(v);
                }
              }
            }
          }
        }

        // Build initArgs in the same order as argVarTypes: args, outs, vars,
        // [time]
        for (auto [i, a] : llvm::enumerate(forallArgs)) {
          if (int(i) >= numArgs && int(i) < numOut + numArgs &&
              int(i) < int(numOut + numArgs + numVars)) {
            initArgs.push_back(outputSmtValues[i - numArgs]); // outs
          } else if (int(i) >= numOut + numArgs &&
                     int(i) < int(numOut + numArgs + numVars)) {
            initArgs.push_back(initVarValues[i - numOut - numArgs]); // vars
          } else {
            initArgs.push_back(a); // args (and time if present)
          }
        }

        auto inInit = b.create<smt::ApplyFuncOp>(loc, stateFunctions[0],
                                                 initArgs);

        if (cfg.withTime) {
          // time is the last forall arg
          Value zeroTime;
              // cfg.useBitVec ? b.create<smt::BVConstantOp>(loc, 0, cfg.timeWidth)
              //               : b.create<smt::IntConstantOp>(
              //                     loc, b.getI32IntegerAttr(0));
          if (cfg.useBitVec) {
            zeroTime = b.create<smt::BVConstantOp>(loc, 0, cfg.timeWidth);
          } else {
            zeroTime = b.create<smt::IntConstantOp>(loc, b.getI32IntegerAttr(0));
          }
          
          Value atZero = b.create<smt::EqOp>(loc, forallArgs.back(), zeroTime);

          return b.create<smt::ImpliesOp>(loc, atZero, inInit);
        }

        return inInit;
      });

  b.create<smt::AssertOp>(loc, forallInit);

  // Transition semantics.
  for (auto [id1, t1] : llvm::enumerate(transitions)) {
    auto action = [&](SmallVector<Value> actionArgs)
        -> SmallVector<Value> {
      // actionArgs are the current tuple (args, outs, vars, [time]).
      SmallVector<Value> outputSmtValues;

      if (t1.hasOutput) {
        SmallVector<std::pair<Value, Value>> avToSmt;
        for (auto [id, av] : llvm::enumerate(argVars))
          avToSmt.push_back({av, actionArgs[id]});
        for (auto &op : t1.output->getOps()) {
          if (auto outOp = dyn_cast<fsm::OutputOp>(op)) {
            for (auto outs : outOp->getOperands()) {
              auto v = getSmtValue(outs, avToSmt, loc);
              outputSmtValues.push_back(v);
            }
          }
          if (auto a = dyn_cast<verif::AssertOp>(op)) {
            // Store original FSM value; defer lowering.
            assertions.push_back({t1.to, a.getOperand(0)});
          }
        }
      }

      SmallVector<std::pair<Value, Value>> avToSmt;
      SmallVector<Value> updatedSmtValues;

      for (auto [id, av] : llvm::enumerate(argVars))
        avToSmt.push_back({av, actionArgs[id]});

      if (t1.hasAction) {
        // Copy current values; override with updates where present.
        for (auto [j, uv] : llvm::enumerate(avToSmt)) {
          bool found = false;
          for (auto &op : t1.action->getOps()) {
            if (auto upd = dyn_cast<fsm::UpdateOp>(op)) {
              if (upd->getOperand(0) == uv.first) {
                auto nv = getSmtValue(upd->getOperand(1), avToSmt, loc);
                updatedSmtValues.push_back(nv);
                found = true;
              }
            }
          }
          if (!found)
            updatedSmtValues.push_back(uv.second);
        }
      } else {
        for (auto [j, uv] : llvm::enumerate(avToSmt))
          updatedSmtValues.push_back(uv.second);
      }

      // Overwrite outputs from output region synthesis.
      for (auto [i, ov] : llvm::enumerate(outputSmtValues))
        updatedSmtValues[numArgs + i] = ov;

      // Time update (if present).
      if (cfg.withTime) {
        Value nextTime;
        if (cfg.useBitVec) {
          auto oneT = b.create<smt::BVConstantOp>(loc, 1, cfg.timeWidth);
          auto resTy = b.getType<smt::BitVectorType>(cfg.timeWidth);
          nextTime = b.create<smt::BVAddOp>(
              loc, resTy, SmallVector<Value>{actionArgs.back(), oneT});
        } else {
          auto one = b.getI32IntegerAttr(1);
          nextTime = b.create<smt::IntAddOp>(
              loc, SmallVector<Value>{actionArgs.back(),
                                      b.create<smt::IntConstantOp>(loc, one)});
        }
        updatedSmtValues.push_back(nextTime);
      }

      return updatedSmtValues;
    };

    auto guard1 = [&](SmallVector<Value> guardArgs) -> Value {
      if (t1.hasGuard) {
        SmallVector<std::pair<Value, Value>> avToSmt;
        for (auto [av, a] : llvm::zip(argVars, guardArgs))
          avToSmt.push_back({av, a});
        for (auto &op : t1.guard->getOps())
          if (auto retOp = dyn_cast<fsm::ReturnOp>(op)) {
            auto gVal = getSmtValue(retOp->getOperand(0), avToSmt, loc);
            // If numeric-1-bit, convert to Bool.
            if (auto bvTy = llvm::dyn_cast<smt::BitVectorType>(gVal.getType()))
              if (bvTy.getWidth() == 1) {
                auto one = b.create<smt::BVConstantOp>(loc, 1, 1);
                return b.create<smt::EqOp>(loc, gVal, one);
              }
            return gVal; // Already Bool
          }
      }
      return b.create<smt::BoolConstantOp>(loc, true);
    };

    // For each transition, assert:
    // Forall (argsNew,argsOld,others): F_from(argsOld, ...) AND guard(argsOld, ...)
    //    => F_to(argsNew, outputs/vars updated, [time+1])
    SmallVector<Type> forallTypes;
    for (auto [id, ty] : llvm::enumerate(argVarTypes)) {
      if (id < numArgs) {
        forallTypes.push_back(ty);
        forallTypes.push_back(ty);
      } else {
        forallTypes.push_back(ty);
      }
    }

    auto forall = b.create<smt::ForallOp>(
        loc, forallTypes,
        [&](OpBuilder &b, Location loc, ValueRange forallDoubleInputs) -> Value {
          SmallVector<Value> startingStateArgs;
          SmallVector<Value> arrivingStateArgs;
          for (auto [idx, fdi] : llvm::enumerate(forallDoubleInputs)) {
            if (idx < static_cast<size_t>(numArgs * 2)) {
              if (idx % 2 == 1)
                startingStateArgs.push_back(fdi);
              else
                arrivingStateArgs.push_back(fdi);
            } else {
              startingStateArgs.push_back(fdi);
              arrivingStateArgs.push_back(fdi);
            }
          }

          auto inFrom = b.create<smt::ApplyFuncOp>(loc, stateFunctions[t1.from],
                                                   startingStateArgs);
          auto actionedArgs = action(startingStateArgs);
          for (auto [ida, aa] : llvm::enumerate(actionedArgs))
            if (ida < numArgs)
              actionedArgs[ida] = arrivingStateArgs[ida];

          auto rhs =
              b.create<smt::ApplyFuncOp>(loc, stateFunctions[t1.to],
                                         actionedArgs);
          auto guard = guard1(startingStateArgs);
          auto lhs = b.create<smt::AndOp>(loc, inFrom, guard);
          return b.create<smt::ImpliesOp>(loc, lhs, rhs);
        });

    b.create<smt::AssertOp>(loc, forall);
  }

  // Lower each captured verif.assert as a safety clause:
  // Forall x. F_state(x) => predicate(x)
  for (auto &pa : assertions) {
    auto forall = b.create<smt::ForallOp>(
        loc, argVarTypes,
        [&](OpBuilder &b, Location loc, ValueRange forallInputs) {
          SmallVector<std::pair<Value, Value>> avToSmt;
          for (auto [i, av] : llvm::enumerate(argVars))
            avToSmt.push_back({av, forallInputs[i]});

          Value predVal = getSmtValue(pa.predicateFsm, avToSmt, loc);
          Value predBool =
              cfg.useBitVec ? bv1ToBool(b, loc, predVal) : numericToBool(predVal, loc);

          Value inState = b.create<smt::ApplyFuncOp>(
              loc, stateFunctions[pa.stateId], forallInputs);

          return b.create<smt::ImpliesOp>(loc, inState, predBool);
        });
    b.create<smt::AssertOp>(loc, forall);
  }

  b.create<smt::YieldOp>(loc, typeRange, valueRange);
  machineOp.erase();
  return success();
}

struct FSMToSMTPass
    : public circt::impl::ConvertFSMToSMTBase<FSMToSMTPass> {
  void runOnOperation() override;
};

void FSMToSMTPass::runOnOperation() {
  auto module = getOperation();
  OpBuilder b(module);

  auto machineOps = to_vector(module.getOps<fsm::MachineOp>());
  if (machineOps.empty()) {
    // markAllAnalysesPreserved();
    return;
  }

  // Read options from the generated base. Defaults:
  // withTime (false), bitVecOrInt ("bitVec").
  LoweringConfig cfg;
  cfg.withTime = withTime;
  // Normalize option string.
  // if (mode == "int" || mode == "ints" || mode == "integer")
  //   cfg.useBitVec = false;
  // else
  //   cfg.useBitVec = true; // default

  // Optional: set time width; keep 5 as stable default (can be parameterized later).
  cfg.timeWidth = 5;

  for (auto machine : llvm::make_early_inc_range(module.getOps<MachineOp>())) {
    MachineOpConverter converter(b, machine, cfg);
    if (failed(converter.dispatch())) {
      signalPassFailure();
      return;
    }
    // Clean up dangling HW constants left around.
    module.walk([&](circt::hw::ConstantOp cst) { cst.erase(); });
  }
}

} // namespace

std::unique_ptr<mlir::Pass> circt::createConvertFSMToSMTPass() {
  return std::make_unique<FSMToSMTPass>();
}