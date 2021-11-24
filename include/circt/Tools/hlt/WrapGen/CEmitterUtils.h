//===- CEmitterUtils.h - C emission utilities -----------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file contains functions for emitting C types from MLIR types.
// Most of these functions are copied from TranslateToCpp, which (unfortunately)
// are not publicly available.
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_TOOLS_HLT_WRAPGEN_CEMITTERUTILS_H
#define CIRCT_TOOLS_HLT_WRAPGEN_CEMITTERUTILS_H

#include "mlir/IR/BuiltinOps.h"
#include "mlir/Support/IndentedOstream.h"

using namespace mlir;

namespace circt {
namespace hlt {

LogicalResult emitType(llvm::raw_ostream &os, Location loc, Type type,
                       Optional<StringRef> variable = {});
LogicalResult emitTupleType(llvm::raw_ostream &os, Location loc,
                            TypeRange types);
LogicalResult emitTypes(llvm::raw_ostream &os, Location loc, TypeRange types);

} // namespace hlt
} // namespace circt

#endif // CIRCT_TOOLS_HLT_WRAPGEN_CEMITTERUTILS_H
