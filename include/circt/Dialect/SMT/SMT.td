//===- SMT.td - SMT dialect definition ---------------------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_DIALECT_SMT_SMT_TD
#define CIRCT_DIALECT_SMT_SMT_TD

include "mlir/IR/OpBase.td"

include "circt/Dialect/SMT/SMTAttributes.td"
include "circt/Dialect/SMT/SMTDialect.td"
include "circt/Dialect/SMT/SMTTypes.td"
include "circt/Dialect/SMT/SMTOps.td"
include "circt/Dialect/SMT/SMTArrayOps.td"
include "circt/Dialect/SMT/SMTBitVectorOps.td"

#endif // CIRCT_DIALECT_SMT_SMT_TD
