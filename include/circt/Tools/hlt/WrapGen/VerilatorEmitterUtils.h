//===- VerilatorEmitterUtils.h - Verilator emission utilities -------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file contains functions for emitting verilator-compatible types from
// MLIR types.
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_TOOLS_HLT_WRAPGEN_VERILATOREMITTERUTILS_H
#define CIRCT_TOOLS_HLT_WRAPGEN_VERILATOREMITTERUTILS_H

#include "mlir/IR/BuiltinOps.h"
#include "mlir/Support/IndentedOstream.h"

using namespace mlir;

namespace circt {
namespace hlt {

/// Outputs Verilator types based on MLIR integer types.
LogicalResult emitVerilatorType(llvm::raw_ostream &os, Location loc, Type type,
                                Optional<StringRef> varName = {});

} // namespace hlt
} // namespace circt

#endif // CIRCT_TOOLS_HLT_WRAPGEN_VERILATOREMITTERUTILS_H
