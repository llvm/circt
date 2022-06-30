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

class SSPOp<string mnemonic, list<Trait> traits = []> :
  Op<SSPDialect, mnemonic, traits>;

def InstanceOp : SSPOp<"instance",
    [IsolatedFromAbove, SymbolTable, NoRegionArguments]
    # GraphRegionNoTerminator.traits> {
  let summary = "Instance of a static scheduling problem";
  let description = [{ TODO }];

  let arguments = (ins StrAttr:$instanceName, StrAttr:$problemName, OptionalAttr<ArrayAttr>:$properties);
  let regions = (region SizedRegion<1>: $body);
  let assemblyFormat = "$instanceName `of` $problemName $properties $body attr-dict";
}

def OperatorTypeOp : SSPOp<"operator_type",
    [Symbol, HasParent<"InstanceOp">]> {
  let summary = "Abstraction of the target architecture";
  let description = [{ TODO }];

  let arguments = (ins SymbolNameAttr:$sym_name, OptionalAttr<ArrayAttr>:$properties);
  let assemblyFormat = "$sym_name $properties attr-dict";
}

def OperationOp : SSPOp<"operation",
    [Symbol, HasParent<"InstanceOp">,
    DeclareOpInterfaceMethods<SymbolUserOpInterface>]> {
  let summary = "Vertex in the dependence graph";
  let description = [{ TODO }];

  let arguments = (ins SymbolNameAttr:$sym_name,
                       Variadic<NoneType>:$operands,
                       DependenceArrayAttr:$dependences,
                       OptionalAttr<ArrayAttr>:$properties);
  let results = (outs Variadic<NoneType>:$results);

  let hasCustomAssemblyFormat = 1;
  let hasVerifier = 1;
}
