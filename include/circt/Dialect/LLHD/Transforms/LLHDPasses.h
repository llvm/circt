//===- Passes.h - LLHD pass entry points ------------------------*- C++ -*-===//
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

#ifndef CIRCT_DIALECT_LLHD_TRANSFORMS_PASSES_H
#define CIRCT_DIALECT_LLHD_TRANSFORMS_PASSES_H

#include "circt/Support/LLVM.h"
#include "mlir/Pass/Pass.h"
#include <memory>

namespace circt {
namespace llhd {

#define GEN_PASS_DECL
#define GEN_PASS_REGISTRATION
#include "circt/Dialect/LLHD/Transforms/LLHDPasses.h.inc"

} // namespace llhd
} // namespace circt

#endif // CIRCT_DIALECT_LLHD_TRANSFORMS_PASSES_H
