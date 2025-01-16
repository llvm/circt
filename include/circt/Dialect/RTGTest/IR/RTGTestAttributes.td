//===- RTGTestAttributes.td - RTGTest attributes -----------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This describes the RTGTest attributes.
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_DIALECT_RTGTEST_IR_RTGTESTATTRIBUTES_TD
#define CIRCT_DIALECT_RTGTEST_IR_RTGTESTATTRIBUTES_TD

include "circt/Dialect/RTGTest/IR/RTGTestDialect.td"
include "circt/Dialect/RTG/IR/RTGInterfaces.td"
include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/EnumAttr.td"

// Flat allocation of unique IDs to all registers. The actual ID value does not
// matter.
def RegisterAttr : I32EnumAttr<
  "Registers", "Unique IDs for all RTGTest registers", [
    I32EnumAttrCase<"zero",  0>,
    I32EnumAttrCase<"ra",  1>,
    I32EnumAttrCase<"sp",  2>,
    I32EnumAttrCase<"gp",  3>,
    I32EnumAttrCase<"tp",  4>,
    I32EnumAttrCase<"t0",  5>,
    I32EnumAttrCase<"t1",  6>,
    I32EnumAttrCase<"t2",  7>,
    I32EnumAttrCase<"s0",  8>,
    I32EnumAttrCase<"s1",  9>,
    I32EnumAttrCase<"a0",  10>,
    I32EnumAttrCase<"a1",  11>,
    I32EnumAttrCase<"a2",  12>,
    I32EnumAttrCase<"a3",  13>,
    I32EnumAttrCase<"a4",  14>,
    I32EnumAttrCase<"a5",  15>,
    I32EnumAttrCase<"a6",  16>,
    I32EnumAttrCase<"a7",  17>,
    I32EnumAttrCase<"s2",  18>,
    I32EnumAttrCase<"s3",  19>,
    I32EnumAttrCase<"s4",  20>,
    I32EnumAttrCase<"s5",  21>,
    I32EnumAttrCase<"s6",  22>,
    I32EnumAttrCase<"s7",  23>,
    I32EnumAttrCase<"s8",  24>,
    I32EnumAttrCase<"s9",  25>,
    I32EnumAttrCase<"s10",  26>,
    I32EnumAttrCase<"s11",  27>,
    I32EnumAttrCase<"t3",  28>,
    I32EnumAttrCase<"t4",  29>,
    I32EnumAttrCase<"t5",  30>,
    I32EnumAttrCase<"t6",  31>,
    I32EnumAttrCase<"Virtual",  32>
  ]> {
  let cppNamespace = "::circt::rtgtest";
}

class RTGTestAttrDef<string name, list<Trait> traits = []> 
  : AttrDef<RTGTestDialect, name, traits>;

def CPUAttr : RTGTestAttrDef<"CPU", [ContextResourceAttrInterface]> {
  let summary = "this attribute represents a CPU referred to by the core ID";

  let parameters = (ins "size_t":$id);

  let mnemonic = "cpu";
  let assemblyFormat = "`<` $id `>`";

  let extraClassDeclaration = [{
    // TypedAttrInterface
    Type getType() const;
  }];
}

#endif // CIRCT_DIALECT_RTGTEST_IR_RTGTESTATTRIBUTES_TD
