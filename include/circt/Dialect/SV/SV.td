//===- SV.td - SystemVerilog dialect definition ------------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This is the top level file for the SV dialect.
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_DIALECT_SV_SV
#define CIRCT_DIALECT_SV_SV

include "circt/Dialect/SV/SVDialect.td"
include "circt/Dialect/SV/SVAttributes.td"
include "circt/Dialect/SV/SVEnums.td"
include "circt/Dialect/SV/SVTypes.td"
include "circt/Dialect/SV/SVExpressions.td"
include "circt/Dialect/SV/SVInOutOps.td"
include "circt/Dialect/SV/SVGenerate.td"
include "circt/Dialect/SV/SVStatements.td"
include "circt/Dialect/SV/SVVerification.td"
include "circt/Dialect/SV/SVTypeDecl.td"

#endif // CIRCT_DIALECT_SV_SV
