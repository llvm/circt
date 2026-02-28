//===- Passes.h - Pass Entrypoints ------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This header file defines prototypes for CIRCT LEC transformation passes.
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_TOOLS_CIRCT_LEC_PASSES_H
#define CIRCT_TOOLS_CIRCT_LEC_PASSES_H

#include "mlir/Pass/Pass.h"

namespace circt {

namespace lec {
enum class InsertAdditionalModeEnum {
  /// Don't insert any LLVM code.
  None,

  /// Insert LLVM code to report the LEC result from an SMT solver.
  Reporting,

  /// Insert a main function for AOT compilation.
  Main,
};
}

/// Generate the code for registering passes.
#define GEN_PASS_DECL_CONSTRUCTLEC
#define GEN_PASS_REGISTRATION
#include "circt/Tools/circt-lec/Passes.h.inc"

} // namespace circt

#endif // CIRCT_TOOLS_CIRCT_LEC_PASSES_H
