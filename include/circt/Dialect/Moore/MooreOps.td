//===- MooreOps.td - Moore dialect operations --------------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_DIALECT_MOORE_MOOREOPS
#define CIRCT_DIALECT_MOORE_MOOREOPS

include "circt/Dialect/Moore/MooreDialect.td"
include "circt/Dialect/Moore/MooreTypes.td"
include "mlir/IR/OpBase.td"
include "mlir/IR/RegionKindInterface.td"
include "mlir/IR/SymbolInterfaces.td"
include "mlir/Interfaces/InferTypeOpInterface.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

// Base class for the operations in this dialect.
class MooreOp<string mnemonic, list<Trait> traits = []> :
  Op<MooreDialect, mnemonic, traits>;

// Base class for the MIR operations in this dialect.
class MIROp<string mnemonic, list<Trait> traits = []> :
  MooreOp<"mir." # mnemonic, traits>;

include "circt/Dialect/Moore/MIRExpressions.td"

//===----------------------------------------------------------------------===//
// Structure
//===----------------------------------------------------------------------===//

def SVModuleOp : MooreOp<"module", [
  IsolatedFromAbove,
  Symbol,
  NoTerminator,
  SingleBlock
]> {
  let summary = "A module definition";
  let description = [{
    The `moore.module` operation represents a SystemVerilog module, including
    its name, port list, and the constituent parts that make up its body. The
    module's body is an SSACFG region, since declarations within SystemVerilog
    modules generally have to appear before their uses, and dedicated assignment
    operators are used to make connections after declarations.

    See IEEE 1800-2017 § 3.3 "Modules" and § 23.2 "Module definitions".
  }];

  let arguments = (ins SymbolNameAttr:$sym_name);
  let regions = (region SizedRegion<1>:$bodyRegion);
  let assemblyFormat = [{
    $sym_name attr-dict-with-keyword $bodyRegion
  }];
}

def InstanceOp : MooreOp<"instance", [
  DeclareOpInterfaceMethods<SymbolUserOpInterface>
]> {
  let summary = "Create an instance of a module";
  let description = [{
    The `moore.instance` operation instantiates a `moore.module` operation.

    See IEEE 1800-2017 § 23.3 "Module instances".
  }];

  let arguments = (ins StrAttr:$instanceName,
                       FlatSymbolRefAttr:$moduleName);
  let assemblyFormat = [{
    $instanceName $moduleName attr-dict
  }];
}

#endif // CIRCT_DIALECT_MOORE_MOOREOPS
