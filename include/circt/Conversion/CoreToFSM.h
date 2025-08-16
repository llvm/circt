//===- CoreToFSM.h - Core to FSM conversions --------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_CONVERSION_CORETOFSM_H
#define CIRCT_CONVERSION_CORETOFSM_H

#include <memory>

namespace mlir {
class Pass;
class ModuleOp;
} // namespace mlir

namespace circt {

std::unique_ptr<mlir::Pass> createConvertCoreToFSMPass();

#define GEN_PASS_DECL_CONVERTCORETOFSM
#include "circt/Conversion/Passes.h.inc"

} // namespace circt

#endif // CIRCT_CONVERSION_CORETOFSM_H