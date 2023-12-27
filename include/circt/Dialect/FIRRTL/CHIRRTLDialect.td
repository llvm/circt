//===- CHIRRTLDialect.td - CHIRRTL dialect definition ------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This is the CHIRRTL dialect.
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_DIALECT_FIRRTL_CHIRRTLDIALECT_TD
#define CIRCT_DIALECT_FIRRTL_CHIRRTLDIALECT_TD

include "mlir/IR/OpAsmInterface.td"
include "mlir/IR/OpBase.td"


def CHIRRTLDialect : Dialect {
  let name = "chirrtl";
  let cppNamespace = "::circt::chirrtl";

  let summary = "Types and operations for the chirrtl dialect";
  let description = [{
    This dialect defines the `chirrtl` dialect, which contains high-level
    memory defintions which can be lowered to FIRRTL.
  }];
  let dependentDialects = ["circt::firrtl::FIRRTLDialect"];

  let useDefaultTypePrinterParser = 1;

  // Opt-out of properties for now, must migrate by LLVM 19.  #5273.
  let usePropertiesForAttributes = 0;
  
  let extraClassDeclaration = [{

    /// Register all FIRRTL types.
    void registerTypes();
  }];
}

#endif // CIRCT_DIALECT_FIRRTL_FIRRTLDIALECT_TD
