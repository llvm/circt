//===- Datapath.td - Datapath dialect definition --------------------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This is the top level file for the Datapath dialect.
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_DIALECT_DATAPATH_DATAPATH_TD
#define CIRCT_DIALECT_DATAPATH_DATAPATH_TD

include "mlir/IR/OpBase.td"

include "circt/Dialect/Datapath/DatapathDialect.td"
// include "circt/Dialect/Datapath/DatapathTypes.td"
include "circt/Dialect/Datapath/DatapathOps.td"

#endif // CIRCT_DIALECT_DATAPATH_DATAPATH_TD
