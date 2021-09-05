//===- CalyxAttributes.td - Calyx Attributes -------------*- tablegen -*---===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This describes the attributes in the Calyx dialect.
//
//===----------------------------------------------------------------------===//

include "mlir/IR/OpBase.td"
include "circt/Dialect/Calyx/Calyx.td"

def Go    : StrEnumAttrCase<"go", 0>;
def Clk   : StrEnumAttrCase<"clk", 1>;
def Done  : StrEnumAttrCase<"done", 2>;
def Reset : StrEnumAttrCase<"reset", 3>;

let cppNamespace = "circt::calyx" in
def RequiredPortAttr : StrEnumAttr<
  "RequiredPort",
  "calyx.component required ports",
  [Go, Clk, Done, Reset]
>;

def CalyxAttr : DialectAttr<
  CalyxDialect, CPred<"$_self.isa<IntegerAttr>">, "Custom attributes for the native dialect."> {
    let storageType = "I64Attr";
    let returnType = "uint64_t";
    let convertFromStorage = "$_self.getValue().getZExtValue();";
}