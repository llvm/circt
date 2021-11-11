//===- FIRRTLDialect.td - FIRRTL dialect definition --------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This is the FIRRTL dialect.
//
//===----------------------------------------------------------------------===//

def FIRRTLDialect : Dialect {
  let name = "firrtl";

  let summary = "Types and operations for firrtl dialect";
  let description = [{
    This dialect defines the `firrtl` dialect, which is used to lower from
    Chisel code to Verilog.  For more information, see the
    [FIRRTL GitHub page](https://github.com/freechipsproject/firrtl).
  }];

  let hasConstantMaterializer = 1;
  let cppNamespace = "::circt::firrtl";
  let extraClassDeclaration = [{
    /// Register all FIRRTL types.
    void registerTypes();
    /// Register all attributes.
    void registerAttributes();
  }];
}

// Base class for the operation in this dialect.
class FIRRTLOp<string mnemonic, list<OpTrait> traits = []> :
    Op<FIRRTLDialect, mnemonic, traits>;
