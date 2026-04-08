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

#include "circt/Dialect/Synth/SynthOps.h"
#include "llvm/ADT/APInt.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/Support/ErrorHandling.h"
#include <optional>

namespace circt {
namespace synth {

enum class BooleanLogicKind { And, Xor, Majority };

inline bool isBooleanLogicOp(Operation *op) {
  return isa<BooleanLogicOpInterface>(op);
}

inline std::optional<BooleanLogicKind> getBooleanLogicKind(Operation *op) {
  if (isa<AndInverterOp>(op))
    return BooleanLogicKind::And;
  if (isa<XorInverterOp>(op))
    return BooleanLogicKind::Xor;
  if (isa<MajorityInverterOp>(op))
    return BooleanLogicKind::Majority;
  return std::nullopt;
}

template <typename ResultT, typename VisitorT, typename FallbackT>
ResultT dispatchBooleanLogicVisitor(Operation *op, VisitorT &&visit,
                                    FallbackT &&fallback) {
  if (auto andOp = dyn_cast<AndInverterOp>(op))
    return visit(andOp);
  if (auto xorOp = dyn_cast<XorInverterOp>(op))
    return visit(xorOp);
  if (auto majOp = dyn_cast<MajorityInverterOp>(op))
    return visit(majOp);
  return fallback(op);
}

inline BooleanLogicOpInterface getBooleanLogicOpInterface(Operation *op) {
  auto logicOp = dyn_cast<BooleanLogicOpInterface>(op);
  if (logicOp)
    return logicOp;
  op->emitOpError("unsupported boolean logic node");
  llvm_unreachable("unsupported boolean logic node");
}

template <typename CallbackT>
void forEachBooleanLogicOperand(Operation *op, CallbackT &&callback) {
  auto logicOp = getBooleanLogicOpInterface(op);
  for (unsigned i = 0, e = logicOp.getNumLogicInputs(); i < e; ++i)
    callback(logicOp.getLogicInput(i), logicOp.isLogicInputInverted(i));
}

inline llvm::APInt evaluateBooleanLogicOp(Operation *op,
                                          llvm::ArrayRef<llvm::APInt> inputs) {
  return getBooleanLogicOpInterface(op).evaluateBooleanLogic(inputs);
}

inline llvm::APInt getSingleBitTruthTable(Operation *op) {
  return getBooleanLogicOpInterface(op).getSingleBitTruthTable();
}

} // namespace synth
} // namespace circt

#endif // CIRCT_DIALECT_SYNTH_SYNTHVISITORS_H
