//===- CHIRRTL.td - CHIRRTL dialect definition -------------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This is the top level file for the CHIRRTL dialect.
//
//===----------------------------------------------------------------------===//

#ifndef CHIRRTL_TD
#define CHIRRTL_TD

include "mlir/IR/OpBase.td"

def CHIRRTLDialect : Dialect {
  let name = "chirrtl";

  let summary = "Types and operations for the chirrtl dialect";
  let description = [{
    This dialect defines the `chirrtl` dialect, which contains high-level
    memory defintions which can be lowered to FIRRTL.
  }];
  let dependentDialects = ["circt::firrtl::FIRRTLDialect"];
  let cppNamespace = "::circt::chirrtl";
}

#endif // CHIRRTL_TDf