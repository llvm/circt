//===- SeqAttributes.td - Attributes for Seq dialect -------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_DIALECT_SEQ_SEQATTRIBUTES_TD
#define CIRCT_DIALECT_SEQ_SEQATTRIBUTES_TD

include "circt/Dialect/Seq/SeqDialect.td"
include "mlir/IR/EnumAttr.td"
include "mlir/IR/BuiltinAttributeInterfaces.td"

let cppNamespace = "circt::seq" in {

// Read-under-write behavior
def RUW_Undefined : I32EnumAttrCase<"Undefined", 0, "undefined">;
def RUW_Old : I32EnumAttrCase<"Old", 1, "old">;
def RUW_New : I32EnumAttrCase<"New", 2, "new">;
def RUWAttr : I32EnumAttr<"RUW", "Read-Under-Write Behavior",
                          [RUW_Undefined, RUW_Old, RUW_New]>;

// Write-under-write behavior
def WUW_Undefined : I32EnumAttrCase<"Undefined", 0, "undefined">;
def WUW_PortOrder : I32EnumAttrCase<"PortOrder", 1, "port_order">;
def WUWAttr : I32EnumAttr<"WUW", "Write-Under-Write Behavior",
                          [WUW_Undefined, WUW_PortOrder]>;

}

#endif // CIRCT_DIALECT_SEQ_SEQATTRIBUTES_TD
