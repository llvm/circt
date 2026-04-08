//===- SynthVisitors.h - Synth Boolean Logic Helpers ------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines lightweight visitors and helpers for the boolean logic ops
// commonly manipulated by Synth transforms.
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_DIALECT_SYNTH_SYNTHVISITORS_H
#define CIRCT_DIALECT_SYNTH_SYNTHVISITORS_H

#include "circt/Dialect/Comb/CombOps.h"
#include "circt/Dialect/Synth/SynthOps.h"
#include "llvm/ADT/APInt.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/ErrorHandling.h"
#include <type_traits>

namespace circt {
namespace synth {

enum class BooleanLogicKind { And, Or, Xor, Majority };

template <typename ResultType = void, typename VisitorT, typename DefaultT>
ResultType dispatchBooleanLogicVisitor(Operation *op, VisitorT &&visitor,
                                       DefaultT &&defaultVisitor) {
  return llvm::TypeSwitch<Operation *, ResultType>(op)
      .template Case<aig::AndInverterOp, mig::MajorityInverterOp, comb::AndOp,
                     comb::OrOp, comb::XorOp>(
          [&](auto logicOp) -> ResultType { return visitor(logicOp); })
      .Default([&](Operation *otherOp) -> ResultType {
        return defaultVisitor(otherOp);
      });
}

inline bool isBooleanLogicOp(Operation *op) {
  return dispatchBooleanLogicVisitor<bool>(
      op, [&](auto) { return true; }, [](Operation *) { return false; });
}

inline BooleanLogicKind getBooleanLogicKind(Operation *op) {
  return dispatchBooleanLogicVisitor<BooleanLogicKind>(
      op,
      [](auto logicOp) {
        using OpTy = std::decay_t<decltype(logicOp)>;
        if constexpr (std::is_same_v<OpTy, aig::AndInverterOp> ||
                      std::is_same_v<OpTy, comb::AndOp>)
          return BooleanLogicKind::And;
        if constexpr (std::is_same_v<OpTy, comb::OrOp>)
          return BooleanLogicKind::Or;
        if constexpr (std::is_same_v<OpTy, comb::XorOp>)
          return BooleanLogicKind::Xor;
        if constexpr (std::is_same_v<OpTy, mig::MajorityInverterOp>)
          return BooleanLogicKind::Majority;
        llvm_unreachable("unexpected boolean logic op");
      },
      [](Operation *op) -> BooleanLogicKind {
        op->emitOpError("unsupported boolean logic node");
        llvm_unreachable("unsupported boolean logic node");
      });
}

template <typename CallbackT>
void forEachBooleanLogicOperand(Operation *op, CallbackT &&callback) {
  dispatchBooleanLogicVisitor<void>(
      op,
      [&](auto logicOp) {
        using OpTy = std::decay_t<decltype(logicOp)>;
        if constexpr (std::is_same_v<OpTy, aig::AndInverterOp> ||
                      std::is_same_v<OpTy, mig::MajorityInverterOp>) {
          for (auto [input, inverted] :
               llvm::zip_equal(logicOp.getInputs(), logicOp.getInverted()))
            callback(input, inverted);
        } else {
          for (Value input : logicOp.getInputs())
            callback(input, false);
        }
      },
      [](Operation *op) {
        op->emitOpError("unsupported boolean logic node");
        llvm_unreachable("unsupported boolean logic node");
      });
}

inline llvm::APInt evaluateBooleanLogicOp(Operation *op,
                                          llvm::ArrayRef<llvm::APInt> inputs) {
  return dispatchBooleanLogicVisitor<llvm::APInt>(
      op,
      [&](auto logicOp) {
        using OpTy = std::decay_t<decltype(logicOp)>;
        if constexpr (std::is_same_v<OpTy, aig::AndInverterOp> ||
                      std::is_same_v<OpTy, mig::MajorityInverterOp>) {
          return logicOp.evaluate(inputs);
        } else if constexpr (std::is_same_v<OpTy, comb::AndOp>) {
          llvm::APInt result =
              llvm::APInt::getAllOnes(inputs.front().getBitWidth());
          for (const auto &input : inputs)
            result &= input;
          return result;
        } else if constexpr (std::is_same_v<OpTy, comb::OrOp>) {
          llvm::APInt result =
              llvm::APInt::getZero(inputs.front().getBitWidth());
          for (const auto &input : inputs)
            result |= input;
          return result;
        } else if constexpr (std::is_same_v<OpTy, comb::XorOp>) {
          llvm::APInt result =
              llvm::APInt::getZero(inputs.front().getBitWidth());
          for (const auto &input : inputs)
            result ^= input;
          return result;
        }
        llvm_unreachable("unexpected boolean logic op");
      },
      [](Operation *logicOp) -> llvm::APInt {
        logicOp->emitOpError("unsupported boolean logic node");
        llvm_unreachable("unsupported boolean logic node");
      });
}

} // namespace synth
} // namespace circt

#endif // CIRCT_DIALECT_SYNTH_SYNTHVISITORS_H
