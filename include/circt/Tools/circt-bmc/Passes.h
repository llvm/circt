//===- Passes.h - Pass Entrypoints ------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This header file defines prototypes for CIRCT BMC transformation passes.
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_TOOLS_CIRCT_BMC_PASSES_H
#define CIRCT_TOOLS_CIRCT_BMC_PASSES_H

#include "mlir/Pass/Pass.h"

namespace circt {

//===----------------------------------------------------------------------===//
// Registration
//===----------------------------------------------------------------------===//

/// Generate the code for registering passes.
#define GEN_PASS_DECL_LOWERTOBMC
#define GEN_PASS_DECL_EXTERNALIZEREGISTERS
#define GEN_PASS_REGISTRATION
#include "circt/Tools/circt-bmc/Passes.h.inc"

} // namespace circt

#endif // CIRCT_TOOLS_CIRCT_BMC_PASSES_H
