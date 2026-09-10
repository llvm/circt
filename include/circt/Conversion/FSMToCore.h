//===- FSMToCore.h - FSM to Core conversions --------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_CONVERSION_FSMTOCORE_FSMTOCORE_H
#define CIRCT_CONVERSION_FSMTOCORE_FSMTOCORE_H

#include <memory>

namespace mlir {
class Pass;
} // namespace mlir

namespace circt {

#define GEN_PASS_DECL_CONVERTFSMTOCORE
#include "circt/Conversion/Passes.h.inc"

std::unique_ptr<mlir::Pass> createConvertFSMToCorePass();
} // namespace circt

#endif // CIRCT_CONVERSION_FSMTOCORE_FSMTOCORE_H
