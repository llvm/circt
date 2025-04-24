
//===- Passes.h - Verif pass entry points ------------------------*- C++-*-===//
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

#ifndef CIRCT_DIALECT_VERIF_VERIFPASSES_H
#define CIRCT_DIALECT_VERIF_VERIFPASSES_H

#include "circt/Dialect/HW/HWDialect.h"
#include "circt/Dialect/SV/SVDialect.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassRegistry.h"
#include <memory>

namespace circt {
namespace verif {
class FormalOp;
class RequireLike;

/// Ways to lower symbolic values.
enum class SymbolicValueLowering {
  /// Lower to instances of an external module.
  ExtModule,
  /// Lower to wire declarations with a `(* anyseq *)` attribute.
  Yosys,
};

/// Construct the command line options to pick one of the symbolic value
/// lowerings.
static inline llvm::cl::ValuesClass symbolicValueLoweringCLValues() {
  return llvm::cl::values(
      clEnumValN(SymbolicValueLowering::ExtModule, "extmodule",
                 "Lower to instances of an external module"),
      clEnumValN(SymbolicValueLowering::Yosys, "yosys",
                 "Lower to `(* anyseq *)` wire declarations"));
}

#define GEN_PASS_DECL
#define GEN_PASS_REGISTRATION
#include "circt/Dialect/Verif/Passes.h.inc"

} // namespace verif
} // namespace circt

#endif // CIRCT_DIALECT_VERIF_VERIFPASSES_H
