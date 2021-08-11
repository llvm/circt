//===- FSMToStandard.h - FSM to Standard conversions ------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_CONVERSION_FSMTOSTANDARD_FSMTOSTANDARD_H
#define CIRCT_CONVERSION_FSMTOSTANDARD_FSMTOSTANDARD_H

#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include <memory>

namespace mlir {
class Pass;
} // namespace mlir

namespace circt {
std::unique_ptr<mlir::Pass> createConvertFSMToStandardPass();
} // namespace circt

#endif // CIRCT_CONVERSION_FSMTOSTANDARD_FSMTOSTANDARD_H
