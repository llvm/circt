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
    The "calyx.register" op defines a register. The clock signal of
    the register is not exposed in the Calyx IR.
    ```
      %r.in, %r.out, %r.write_en, %r.done = calyx.register "r" : i32, i32, i1, i1
    ```
  }];
  let arguments = (ins StrAttr:$name);
  let results = (outs AnyType:$in, AnyType:$out, I1:$write_en, I1:$done);
  let assemblyFormat = "$name attr-dict `:` type(results)";

  let builders = [
    OpBuilder<(ins "StringAttr":$name, "size_t":$width), [{
      $_state.addAttribute("name", name);
      auto i1Type = $_builder.getI1Type();
      auto widthType = $_builder.getIntegerType(width);
      $_state.addTypes({widthType, widthType, i1Type, i1Type});
    }]>
  ];
}
