//===- SMT.cpp - C interface for the SMT dialect ----------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "circt-c/Dialect/SMT.h"
#include "circt/Dialect/SMT/SMTDialect.h"

#include "mlir/CAPI/Registration.h"

MLIR_DEFINE_CAPI_DIALECT_REGISTRATION(SMT, smt, circt::smt::SMTDialect)
