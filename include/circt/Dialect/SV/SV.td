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

#ifndef SV_TD
#define SV_TD

include "mlir/IR/OpBase.td"
include "mlir/IR/SymbolInterfaces.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

def SVDialect : Dialect {
  let name = "sv";

  let summary = "Types and operations for SV dialect";
  let description = [{
    This dialect defines the `sv` dialect, which represents various
    SystemVerilog-specific constructs in an AST-like representation.
  }];

  let cppNamespace = "::circt::sv";
}

// Base class for the operation in this dialect.
class SVOp<string mnemonic, list<OpTrait> traits = []> :
    Op<SVDialect, mnemonic, traits>;
class SVType<string name> : TypeDef<SVDialect, name> { }

// FIXME: Shouldn't have to duplicate this from the RTL dialect tblgen files.
def InOutType : Type<CPred<"$_self.isa<rtl::InOutType>()">, "InOutType">;

include "SVExpressions.td"
include "SVStatements.td"
include "SVTypeDecl.td"

#endif // SV_TD
