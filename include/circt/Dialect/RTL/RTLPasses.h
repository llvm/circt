//===- RTLPasses.h - RTL pass entry points -----------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This header file defines prototypes that expose pass constructors.
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_DIALECT_RTL_RTLPASSES_H
#define CIRCT_DIALECT_RTL_RTLPASSES_H

#include <memory>

namespace mlir {
class ModuleOp;
template <typename T>
class OperationPass;
} // namespace mlir

namespace circt {
namespace rtl {

std::unique_ptr<mlir::Pass> createRTLGreyBoxerPass();

/// Generate the code for registering passes.
#define GEN_PASS_REGISTRATION
#include "circt/Dialect/RTL/RTLPasses.h.inc"

} // namespace rtl
} // namespace circt

#endif // CIRCT_DIALECT_RTL_RTLPASSES_H
