//===- CalyxPrimitives.td - Calyx Primitives -------------*- tablegen -*---===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This describes the primitives in the Calyx dialect.
//
//===----------------------------------------------------------------------===//

def RegisterOp : CalyxOp<"register", [
    DeclareOpInterfaceMethods<OpAsmOpInterface>,
    HasParent<"ComponentOp">,
    SameTypeConstraint<"in", "out">
  ]> {
  let summary = "Defines a register";
  let description = [{
    The "calyx.register" op defines a register.
    ```
      %r.in, %r.write_en, %r.clk, %r.reset, %r.out, %r.done = calyx.register "r" : i32
    ```
  }];
  let arguments = (ins StrAttr:$name);
  let results = (outs AnyType:$in, I1:$write_en, I1:$clk, I1:$reset, AnyType:$out, I1:$done);
  let assemblyFormat = "$name attr-dict `:` type($in)";

  let builders = [
    OpBuilder<(ins "StringAttr":$name, "size_t":$width), [{
      $_state.addAttribute("name", name);
      auto i1Type = $_builder.getI1Type();
      auto widthType = $_builder.getIntegerType(width);
      $_state.addTypes({widthType, i1Type, i1Type, i1Type, widthType, i1Type});
    }]>
  ];
}
