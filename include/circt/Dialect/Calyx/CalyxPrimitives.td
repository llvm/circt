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

/// Base class for Calyx primitives.
class CalyxPrimitive<string mnemonic, list<OpTrait> traits = []> :
  CalyxCell<mnemonic, traits> {
    let assemblyFormat = "$instanceName attr-dict `:` type(results)";
    let arguments = (ins StrAttr:$instanceName);
    let skipDefaultBuilders = 1;
}

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
    }]>,
    OpBuilder<(ins "StringAttr":$name, "Type":$type), [{
      $_state.addAttribute("name", name);
      auto i1Type = $_builder.getI1Type();
      $_state.addTypes({type, i1Type, i1Type, i1Type, type, i1Type});
    }]>
  ];
  let extraClassDeclaration = [{
    SmallVector<StringRef> portNames() {
      return {"in", "write_en", "clk", "reset", "out", "done"};
    }
    SmallVector<Direction> portDirections() {
      return { Input, Input, Input, Input, Output, Output };
    }

    Value inPort()      { return getResult(0); }
    Value writeEnPort() { return getResult(1); }
    Value clkPort()     { return getResult(2); }
    Value resetPort()   { return getResult(3); }
    Value outPort()     { return getResult(4); }
    Value donePort()    { return getResult(5); }
  }];

}

def MemoryOp : CalyxPrimitive<"memory", []> {
  let summary = "Defines a memory";
  let description = [{
    The "calyx.memory" op defines a memory. Memories can have any number of
    dimensions, as specified by the length of the `$sizes` and `$addrSizes`
    arrays. The `$addrSizes` specify the bitwidth of each dimension's address,
    and should be wide enough to address the range of the corresponding
    dimension in `$sizes`. The `$width` attribute dictates the width of a single
    element.

    See https://capra.cs.cornell.edu/docs/calyx/libraries/core.html#memories for
    more information.

    ```mlir
      // A 1-dimensional, 32-bit memory with size dimension 1. Equivalent representation in the native compiler:
      // `m1 = std_mem_d1(32, 1, 1)`
      %m1.addr0, %m1.write_data, %m1.write_en, %m1.read_data, %m1.done = calyx.memory "m1"<[1] x 32> [1] : i1, i32, i1, i32, i1

      // A 2-dimensional, 8-bit memory with size dimensions 64 x 64. Equivalent representation in the native compiler:
      // `m2 = std_mem_d2(8, 64, 64, 6, 6)`
      %m2.addr0, %m2.addr1, %m2.write_data, %m2.write_en, %m2.read_data, %m2.done = calyx.memory "m2"<[64, 64] x 8> [6, 6] : i6, i6, i8, i1, i8, i1
    ```
  }];

  let arguments = (ins
    StrAttr:$instanceName,
    I64Attr:$width,
    ArrayAttr:$sizes,
    ArrayAttr:$addrSizes
  );

  let results = (outs
    Variadic<AnySignlessInteger>:$results
  );

  let assemblyFormat = [{
    $instanceName `<` $sizes `x` $width `>` $addrSizes attr-dict `:` type($results)
  }];

  let verifier = "return ::verify$cppClass(*this);";

  let builders = [
    OpBuilder<(ins
      "Twine":$instanceName,
      "int64_t":$width,
      "ArrayRef<int64_t>":$sizes,
      "ArrayRef<int64_t>":$addrSizes
    )>
  ];

  let extraClassDeclaration = [{
    size_t getNumDimensions() { return addrSizes().size(); }

    ValueRange addrPorts() { return getResults().take_front(getNumDimensions()); }
    Value addrPort(size_t i) { assert(i<getNumDimensions() && "Out of bounds of address ports for this memory."); return getResults()[i]; }
    Value writeData() { return getResult(getNumDimensions()); }
    Value writeEn()   { return getResult(getNumDimensions() + 1); }
    Value readData()  { return getResult(getNumDimensions() + 2); }
    Value done()      { return getResult(getNumDimensions() + 3); }
  }];
}
