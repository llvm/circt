//===- MemToRegOfVec.h - Shared comb-mem conversion API ---------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Shared helpers used by MemToRegOfVec and FullReset to convert combinational
// memories into register vectors.
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_DIALECT_FIRRTL_TRANSFORMS_MEMTOREGOFVEC_H
#define CIRCT_DIALECT_FIRRTL_TRANSFORMS_MEMTOREGOFVEC_H

#include "circt/Dialect/FIRRTL/FIRRTLOps.h"
#include "llvm/ADT/ArrayRef.h"

namespace circt {
namespace firrtl {

/// Convert combinational memories in `mod` into register vectors.
/// Sequential memories are left unchanged. Increments `numConverted` for each
/// converted memory.
void convertCombMemsToRegOfVec(FModuleOp mod, bool ignoreReadEnable,
                               unsigned &numConverted);

/// Convert combinational memories in each module in `mods`.
void convertCombMemsToRegOfVec(ArrayRef<FModuleOp> mods, bool ignoreReadEnable,
                               unsigned &numConverted);

} // namespace firrtl
} // namespace circt

#endif // CIRCT_DIALECT_FIRRTL_TRANSFORMS_MEMTOREGOFVEC_H
