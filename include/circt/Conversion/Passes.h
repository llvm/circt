//===- Passes.h - Conversion Pass Construction and Registration -*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This fle contains the declarations to register conversion passes.
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_CONVERSION_PASSES_H
#define CIRCT_CONVERSION_PASSES_H

#include "circt/Conversion/FIRRTLToRTL/FIRRTLToRTL.h"
#include "mlir/Pass/PassRegistry.h"

namespace circt {

// Generate the code for registering conversion passes.
#define GEN_PASS_REGISTRATION
#include "circt/Conversion/Passes.h.inc"

} // namespace circt

#endif // CIRCT_CONVERSION_PASSES_H
