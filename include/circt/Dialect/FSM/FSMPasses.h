//===- FSMPasses.h - FSM pass entry points ----------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_DIALECT_FSM_FSMPASSES_H
#define CIRCT_DIALECT_FSM_FSMPASSES_H

#include "mlir/Pass/Pass.h"

#include <memory>

namespace mlir {
class ModuleOp;
template <typename T>
class OperationPass;
} // namespace mlir

namespace circt {
namespace fsm {

std::unique_ptr<mlir::Pass> createPrintStateGraphPass();

/// Generate the code for registering passes.
#define GEN_PASS_REGISTRATION
#include "circt/Dialect/FSM/FSMPasses.h.inc"

} // namespace fsm
} // namespace circt

#endif // CIRCT_DIALECT_FSM_FSMPASSES_H
