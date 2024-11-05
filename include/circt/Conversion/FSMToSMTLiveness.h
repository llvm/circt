//===- FSMTOSMTLIVENESS.h - FSM to SMT conversions ------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_CONVERSION_FSMTOSMTLIVENESS_FSMTOSMTLIVENESS_H
#define CIRCT_CONVERSION_FSMTOSMTLIVENESS_FSMTOSMTLIVENESS_H

#include <memory>

namespace mlir {
class Pass;
} // namespace mlir

namespace circt {

#define GEN_PASS_DECL_CONVERTFSMTOSMTLIVENESS
#include "circt/Conversion/Passes.h.inc"


std::unique_ptr<mlir::Pass> createConvertFSMToSMTLivenessPass();
} // namespace circt

#endif // CIRCT_CONVERSION_FSMTOSMTLIVENESS_FSMTOSMTLIVENESS_H
