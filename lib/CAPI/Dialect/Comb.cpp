//===- Comb.cpp - C interface for the Comb dialect ------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "circt-c/Dialect/Comb.h"
#include "circt/Dialect/Comb/CombDialect.h"
#include "circt/Dialect/Comb/CombPasses.h"
#include "mlir/CAPI/IR.h"
#include "mlir/CAPI/Registration.h"
#include "mlir/CAPI/Support.h"

void registerCombPasses() { circt::comb::registerPasses(); }

MLIR_DEFINE_CAPI_DIALECT_REGISTRATION(Combinational, comb,
                                      circt::comb::CombDialect)
