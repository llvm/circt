//===- SSPOps.td - SSP operation definitions ---------------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines the SSP (static scheduling problem) dialect operations.
//
//===----------------------------------------------------------------------===//

include "mlir/IR/RegionKindInterface.td"
include "mlir/IR/SymbolInterfaces.td"

def InstanceOp : Op<SSPDialect, "instance",
    [IsolatedFromAbove, SymbolTable,
    NoRegionArguments, NoTerminator, SingleBlock,
    DeclareOpInterfaceMethods<RegionKindInterface>]> {
  
  let summary = "Foo";
  let description = [{
    TODO
  }];

  let arguments = (ins StrAttr:$problemName);
  let regions = (region SizedRegion<1>: $body);
  let assemblyFormat = "$problemName $body attr-dict";
}
