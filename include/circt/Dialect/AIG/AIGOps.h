//===- AIGOps.h - AIG Op Definitions ----------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_DIALECT_AIG_AIGOPS_H
#define CIRCT_DIALECT_AIG_AIGOPS_H

#include "circt/Support/LLVM.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/Operation.h"
#include "mlir/Interfaces/ControlFlowInterfaces.h"
#include "mlir/Interfaces/FunctionInterfaces.h"
#include "mlir/Interfaces/InferTypeOpInterface.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Rewrite/PatternApplicator.h"

#include "circt/Dialect/AIG/AIGDialect.h"

#define GET_OP_CLASSES
#include "circt/Dialect/AIG/AIG.h.inc"

namespace circt {
namespace aig {
struct AndInverterVariadicOpConversion
    : mlir::OpRewritePattern<circt::aig::AndInverterOp> {
  using OpRewritePattern<circt::aig::AndInverterOp>::OpRewritePattern;
  mlir::LogicalResult
  matchAndRewrite(circt::aig::AndInverterOp op,
                  mlir::PatternRewriter &rewriter) const override;
};
} // namespace aig
} // namespace circt

#endif // CIRCT_DIALECT_AIG_AIGOPS_H
