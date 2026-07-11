//===- ProbeDialect.td - Probe dialect definition -----------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_DIALECT_PROBE_PROBEDIALECT_TD
#define CIRCT_DIALECT_PROBE_PROBEDIALECT_TD

include "mlir/IR/DialectBase.td"
include "mlir/IR/OpBase.td"

def ProbeDialect : Dialect {
  let name = "probe";
  let summary = "SSA-based hardware probe handles";
  let cppNamespace = "circt::probe";

  let useDefaultTypePrinterParser = 1;

  let extraClassDeclaration = [{
    void registerOps();
    void registerTypes();
  }];
}

#endif // CIRCT_DIALECT_PROBE_PROBEDIALECT_TD
