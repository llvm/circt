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

include "mlir/IR/OpAsmInterface.td"
include "mlir/IR/RegionKindInterface.td"
include "mlir/IR/SymbolInterfaces.td"

class SSPOp<string mnemonic, list<Trait> traits = []> :
  Op<SSPDialect, mnemonic, traits>;

def InstanceOp : SSPOp<"instance",
    [IsolatedFromAbove, SymbolTable, NoRegionArguments, OpAsmOpInterface]
    # GraphRegionNoTerminator.traits> {
  let summary = "Instance of a static scheduling problem";
  let description = [{ TODO }];

  let arguments = (ins StrAttr:$instanceName, StrAttr:$problemName, OptionalAttr<ArrayAttr>:$properties);
  let regions = (region SizedRegion<1>: $body);
  let assemblyFormat = "$instanceName `of` $problemName custom<Properties>($properties) $body attr-dict";
  
  let extraClassDeclaration = [{
    static ::llvm::StringRef getDefaultDialect() { return "ssp"; }
  }];

  let skipDefaultBuilders = true;
  let builders = [
    OpBuilder<(ins "::mlir::StringAttr":$instanceName,
                   "::mlir::StringAttr":$problemName,
                   CArg<"::mlir::ArrayAttr", "::mlir::ArrayAttr()">:$properties), [{
      $_state.addAttribute($_builder.getStringAttr("instanceName"), instanceName);
      $_state.addAttribute($_builder.getStringAttr("problemName"), problemName);
      if (properties)
        $_state.addAttribute($_builder.getStringAttr("properties"), properties);
      ::mlir::Region* region = $_state.addRegion();
      region->push_back(new ::mlir::Block());
    }]>
  ];
}

def OperatorTypeOp : SSPOp<"operator_type",
    [Symbol, HasParent<"InstanceOp">]> {
  let summary = "Abstraction of the target architecture";
  let description = [{ TODO }];

  let arguments = (ins SymbolNameAttr:$sym_name, OptionalAttr<ArrayAttr>:$properties);
  let assemblyFormat = "$sym_name custom<Properties>($properties) attr-dict";
}

def OperationOp : SSPOp<"operation",
    [Symbol, HasParent<"InstanceOp">,
    DeclareOpInterfaceMethods<SymbolUserOpInterface>]> {
  let summary = "Vertex in the dependence graph";
  let description = [{ TODO }];

  let arguments = (ins Variadic<NoneType>:$operands,
                       OptionalAttr<SymbolNameAttr>:$sym_name,
                       OptionalAttr<DependenceArrayAttr>:$dependences,
                       OptionalAttr<ArrayAttr>:$properties);
  let results = (outs Variadic<NoneType>:$results);

  let hasCustomAssemblyFormat = 1;
  let hasVerifier = 1;

  let extraClassDeclaration = [{
    static bool isOptionalSymbol() { return true; }
  }];

  let skipDefaultBuilders = true;
  let builders = [
    OpBuilder<(ins "unsigned":$numResults,
                   "::mlir::ValueRange":$operands,
                   CArg<"::mlir::StringAttr", "::mlir::StringAttr()">:$sym_name,
                   CArg<"::mlir::ArrayAttr", "::mlir::ArrayAttr()">:$dependences,
                   CArg<"::mlir::ArrayAttr", "::mlir::ArrayAttr()">:$properties), [{
      $_state.addTypes(::llvm::SmallVector<::mlir::Type>(numResults, $_builder.getNoneType()));
      $_state.addOperands(operands);
      if (sym_name)
        $_state.addAttribute(::mlir::SymbolTable::getSymbolAttrName(), sym_name);
      if (dependences)
        $_state.addAttribute($_builder.getStringAttr("dependences"), dependences);
      if (properties)
        $_state.addAttribute($_builder.getStringAttr("properties"), properties);
    }]>
  ];
}
