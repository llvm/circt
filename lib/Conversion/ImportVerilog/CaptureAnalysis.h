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
class InstanceSymbol;
class HierarchicalReference;
enum class SymbolKind : int;
} // namespace ast
} // namespace slang

namespace circt {
namespace ImportVerilog {

/// Return true if symbols of this kind are elaboration-time constants. They
/// are materialized inline instead of being captured or ported.
bool isCompileTimeConstant(slang::ast::SymbolKind kind);

/// Return the first instance on a hierarchical reference path, i.e. the
/// instance the reference enters the hierarchy through.
const slang::ast::InstanceSymbol *
getRootInstance(const slang::ast::HierarchicalReference &ref);

/// The result of capture analysis: for each function, the set of non-local,
/// non-global variable symbols that the function captures directly or
/// transitively through calls.
using CaptureMap = DenseMap<const slang::ast::SubroutineSymbol *,
                            SmallSetVector<const slang::ast::ValueSymbol *, 4>>;

/// A function whose hierarchical reference resolves to one symbol reachable
/// through more than one instance. Lowering cannot pick an instance safely.
struct AmbiguousHierCapture {
  const slang::ast::SubroutineSymbol *function;
  const slang::ast::ValueSymbol *symbol;
};

/// Analyze the AST rooted at `root` to determine which variables each
/// function captures: symbols referenced inside the function's body (directly
/// or through called functions) that are neither local nor global, including
/// hierarchical references to other instances. Captures that cannot be
/// resolved to a single instance are reported in `ambiguous`.
CaptureMap
analyzeFunctionCaptures(const slang::ast::RootSymbol &root,
                        SmallVectorImpl<AmbiguousHierCapture> &ambiguous);

} // namespace ImportVerilog
} // namespace circt

#endif // CONVERSION_IMPORTVERILOG_CAPTUREANALYSIS_H
