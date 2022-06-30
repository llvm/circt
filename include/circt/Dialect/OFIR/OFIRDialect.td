//===- OFIRDialect.td - OFIR dialect definition ------------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This is the OFIR dialect.
//
//===----------------------------------------------------------------------===//

def OFIRDialect : Dialect {
  let name = "ofir";
  let cppNamespace = "::circt::ofir";

  let summary = "Types and operations for ofir dialect";
  let description = [{
    This dialect defines the `ofir` dialect, which is used to lower from
    Chisel-TNG code to Verilog.  For more information, see the
    [FIRRTL GitHub page](https://github.com/freechipsproject/firrtl).
  }];

  let hasConstantMaterializer = 1;
  let useDefaultTypePrinterParser = 1;
  let useDefaultAttributePrinterParser = 1;

  let dependentDialects = [
    "circt::hw::HWDialect",
    "circt::firrtl::FIRRTLDialect"
  ];

  let extraClassDeclaration = [{
    /// Register all FIRRTL types.
    void registerTypes();
    /// Register all attributes.
    void registerAttributes();
  }];
}

// Base class for the operation in this dialect.
class OFIROp<string mnemonic, list<Trait> traits = []> :
    Op<OFIRDialect, mnemonic, traits>;
