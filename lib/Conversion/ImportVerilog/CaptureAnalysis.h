//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Pre-pass over the slang AST to determine which non-local, non-global
// variables each function captures, either directly or transitively through
// calls to other functions.
//
// This information is needed before any MLIR conversion happens so that
// function declarations can be created with the correct signature (including
// extra capture parameters) upfront, enabling a clean two-phase
// declare-then-define approach for functions.
//
//===----------------------------------------------------------------------===//

// NOLINTNEXTLINE(llvm-header-guard)
#ifndef CONVERSION_IMPORTVERILOG_CAPTUREANALYSIS_H
#define CONVERSION_IMPORTVERILOG_CAPTUREANALYSIS_H

#include "circt/Support/LLVM.h"
#include "llvm/ADT/SetVector.h"

namespace slang {
namespace ast {
class RootSymbol;
class SubroutineSymbol;
class ValueSymbol;
} // namespace ast
} // namespace slang

namespace circt {
namespace ImportVerilog {

/// The result of capture analysis: for each function, the set of non-local,
/// non-global variable symbols that the function captures directly or
/// transitively through calls.
using CaptureMap = DenseMap<const slang::ast::SubroutineSymbol *,
                            SmallSetVector<const slang::ast::ValueSymbol *, 4>>;

/// Analyze the AST rooted at `root` to determine which variables each function
/// captures. A variable is considered captured by a function if it is
/// referenced inside the function's body (or transitively through called
/// functions) and is neither local to the function nor a global variable
/// (package-scope or compilation-unit-scope variables that are lowered via
/// `get_global_signal`).
CaptureMap analyzeFunctionCaptures(const slang::ast::RootSymbol &root);

} // namespace ImportVerilog
} // namespace circt

#endif // CONVERSION_IMPORTVERILOG_CAPTUREANALYSIS_H
