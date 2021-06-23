//===- Calyx.td - Calyx dialect definition -----------------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This is the top level file for the Calyx dialect.
//
//===----------------------------------------------------------------------===//

#ifndef CALYX_TD
#define CALYX_TD

include "mlir/IR/OpBase.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/IR/OpAsmInterface.td"
include "mlir/IR/SymbolInterfaces.td"

def CalyxDialect : Dialect {
  let name = "calyx";
  let summary = "Types and operations for the Calyx dialect";
  let description = [{
    Calyx is an intermediate language and infrastructure for building
    compilers that generate custom hardware accelerators. For more
    information, visit the
    [documentation](https://capra.cs.cornell.edu/calyx/).
  }];
  let extraClassDeclaration = [{
    /// Register all Calyx types.
    void registerTypes();
    /// Register all Calyx attributes.
    void registerAttributes();
  }];
  let cppNamespace = "::circt::calyx";
}

/// Base class for the operation in this dialect.
class CalyxOp<string mnemonic, list<OpTrait> traits = []> :
  Op<CalyxDialect, mnemonic, traits>;

/// Base class for Calyx containers.
class CalyxContainer<string mnemonic, list<OpTrait> traits = []> :
  Op<CalyxDialect, mnemonic, traits> {
  let assemblyFormat = "$body attr-dict";
  let regions = (region SizedRegion<1>: $body);
}

include "circt/Dialect/Calyx/CalyxStructure.td"

#endif // CALYX_TD
