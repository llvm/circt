//===- SVDialect.td - SystemVerilog dialect definition -----*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This contains the SV dialect definition to be included in other files.
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_DIALECT_SV_SVDIALECT
#define CIRCT_DIALECT_SV_SVDIALECT

include "mlir/IR/OpBase.td"
include "mlir/IR/OpAsmInterface.td"

def SVDialect : Dialect {
  let name = "sv";
  let cppNamespace = "::circt::sv";

  let summary = "Types and operations for SV dialect";
  let description = [{
    This dialect defines the `sv` dialect, which represents various
    SystemVerilog-specific constructs in an AST-like representation.
  }];
  let dependentDialects = ["circt::comb::CombDialect", "circt::hw::HWDialect", 
    "circt::ltl::LTLDialect"];

  let useDefaultTypePrinterParser = 1;
  let useDefaultAttributePrinterParser = 1;

  let extraClassDeclaration = [{
    /// Register all SV types.
    void registerTypes();
    /// Register all attributes.
    void registerAttributes();
  }];
}

// Base class for the operation in this dialect.
class SVOp<string mnemonic, list<Trait> traits = []> :
    Op<SVDialect, mnemonic, traits>;

/// Mark an operation as being a vendor extension.
def VendorExtension : NativeOpTrait<"VendorExtension"> {
  let cppNamespace = "::circt::sv";
}

/// Shorthand for ops that provide a custom SSA name via OpAsmOpInterface.
def HasCustomSSAName : DeclareOpInterfaceMethods<OpAsmOpInterface,
                           ["getAsmResultNames"]>;

#endif // CIRCT_DIALECT_SV_SVDIALECT
