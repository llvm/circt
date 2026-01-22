//===- LTLPasses.h - LTL pass entry points ----------------------*- C++ -*-===//
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

#ifndef CIRCT_DIALECT_LTL_LTLPASSES_H
#define CIRCT_DIALECT_LTL_LTLPASSES_H

#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassRegistry.h"

namespace circt {
namespace ltl {

#define GEN_PASS_DECL
#include "circt/Dialect/LTL/LTLPasses.h.inc"

/// Generate the code for registering passes.
#define GEN_PASS_REGISTRATION
#include "circt/Dialect/LTL/LTLPasses.h.inc"

} // namespace ltl
} // namespace circt

#endif // CIRCT_DIALECT_LTL_LTLPASSES_H
