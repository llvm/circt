//===- DatapathOps.td - Datapath dialect ops ---------------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This describes the MLIR ops for datapath logic.
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_DIALECT_DATAPATH_OPS_TD
#define CIRCT_DIALECT_DATAPATH_OPS_TD

include "circt/Dialect/Datapath/DatapathDialect.td"
include "mlir/Interfaces/InferTypeOpInterface.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

include "circt/Dialect/HW/HWTypes.td"

// Base class for the operation in this dialect.
class DatapathOp<string mnemonic, list<Trait> traits = []> :
    Op<DatapathDialect, mnemonic, traits>;


#endif // CIRCT_DIALECT_DATAPATH_OPS_TD
