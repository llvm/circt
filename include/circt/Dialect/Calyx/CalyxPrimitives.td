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

def RegisterOp : CalyxPrimitive<"register", [
    SameTypeConstraint<"in", "out">
  ]> {
  let summary = "Defines a register";
  let description = [{
    The "calyx.register" op defines a register.
    ```mlir
      // A 32-bit register.
      %r.in, %r.write_en, %r.clk, %r.reset, %r.out, %r.done = calyx.register "r" : i32, i1, i1, i1, i32, i1
    ```
  }];
  let results = (outs AnyType:$in, I1:$write_en, I1:$clk, I1:$reset, AnyType:$out, I1:$done);
  let builders = [
    OpBuilder<(ins "StringAttr":$instanceName, "size_t":$width), [{
      $_state.addAttribute("instanceName", instanceName);
      auto i1Type = $_builder.getI1Type();
      auto widthType = $_builder.getIntegerType(width);
      $_state.addTypes({widthType, i1Type, i1Type, i1Type, widthType, i1Type});
    }]>
  ];
  let extraClassDeclaration = [{
    SmallVector<StringRef> portNames() { return {"in", "write_en", "clk", "reset", "out", "done"}; }
  }];

}
