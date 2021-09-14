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

/// Trait used to mark certain operations in the Calyx dialect as combinational.
def Combinational : NativeOpTrait<"Combinational"> {
  let cppNamespace = "::circt::calyx";
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
      Type i1Type = $_builder.getI1Type();
      Type widthType = $_builder.getIntegerType(width);
      $_state.addTypes({widthType, i1Type, i1Type, i1Type, widthType, i1Type});
    }]>,
    OpBuilder<(ins "StringAttr":$instanceName, "Type":$type), [{
      $_state.addAttribute("instanceName", instanceName);
      Type i1Type = $_builder.getI1Type();
      $_state.addTypes({type, i1Type, i1Type, i1Type, type, i1Type});
    }]>
  ];
  let extraClassDeclaration = [{
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
      %m1.addr0, %m1.write_data, %m1.write_en, %m1.clk, %m1.read_data, %m1.done = calyx.memory "m1"<[1] x 32> [1] : i1, i32, i1, i1, i32, i1

      // A 2-dimensional, 8-bit memory with size dimensions 64 x 64. Equivalent representation in the native compiler:
      // `m2 = std_mem_d2(8, 64, 64, 6, 6)`
      %m2.addr0, %m2.addr1, %m2.write_data, %m2.write_en, %m2.clk, %m2.read_data, %m2.done = calyx.memory "m2"<[64, 64] x 8> [6, 6] : i6, i6, i8, i1, i1, i8, i1
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
    Value addrPort(size_t i) {
      assert(i < getNumDimensions() && "index greater than number of memory address ports.");
      return getResults()[i];
    }
    Value writeData() { return getResult(getNumDimensions()); }
    Value writeEn()   { return getResult(getNumDimensions() + 1); }
    Value clk()       { return getResult(getNumDimensions() + 2); }
    Value readData()  { return getResult(getNumDimensions() + 3); }
    Value done()      { return getResult(getNumDimensions() + 4); }
  }];
}

class CalyxLibraryOp<string mnemonic, list<OpTrait> traits = [
    Combinational
  ]> : CalyxPrimitive<"std_" # mnemonic, traits> {

  let summary = "Defines an operation which maps to a Calyx library primitive";
  let description = [{
    This operation represents an instance of a Calyx library primitive.
    A library primitive maps to some hardware-implemented component within the
    native Calyx compiler.
  }];

  let arguments = (ins StrAttr:$instanceName);
  let builders = [
    OpBuilder<(ins "StringAttr":$instanceName, "::mlir::TypeRange":$resultTypes), [{
      $_state.addAttribute("instanceName", instanceName);
      $_state.addTypes(resultTypes);
    }]>
  ];
}

class BoolBinaryLibraryOp<string mnemonic> : CalyxLibraryOp<mnemonic, [
    SameTypeConstraint<"left", "right">
  ]> {
  let results = (outs AnyType:$left, AnyType:$right, I1:$out);
  let extraClassDeclaration = [{
    Value lhsPort() { return getResult(0); }
    Value rhsPort() { return getResult(1); }
    Value outPort() { return getResult(2); }
  }];
}

def LtLibOp  : BoolBinaryLibraryOp<"lt"> {}
def GtLibOp  : BoolBinaryLibraryOp<"gt"> {}
def EqLibOp  : BoolBinaryLibraryOp<"eq"> {}
def NeqLibOp : BoolBinaryLibraryOp<"neq"> {}
def GeLibOp  : BoolBinaryLibraryOp<"ge"> {}
def LeLibOp  : BoolBinaryLibraryOp<"le"> {}
def SltLibOp  : BoolBinaryLibraryOp<"slt"> {}
def SgtLibOp  : BoolBinaryLibraryOp<"sgt"> {}
def SeqLibOp  : BoolBinaryLibraryOp<"seq"> {}
def SneqLibOp : BoolBinaryLibraryOp<"sneq"> {}
def SgeLibOp  : BoolBinaryLibraryOp<"sge"> {}
def SleLibOp  : BoolBinaryLibraryOp<"sle"> {}

class ArithBinaryLibraryOp<string mnemonic> : CalyxLibraryOp<mnemonic, [
    SameTypeConstraint<"left", "right">,
    SameTypeConstraint<"left", "out">
  ]> {
  let results = (outs AnyType:$left, AnyType:$right, AnyType:$out);
  let extraClassDeclaration = [{
    Value lhsPort() { return getResult(0); }
    Value rhsPort() { return getResult(1); }
    Value outPort() { return getResult(2); }
  }];
}

def AddLibOp  : ArithBinaryLibraryOp<"add"> {}
def SubLibOp  : ArithBinaryLibraryOp<"sub"> {}
def ShruLibOp : ArithBinaryLibraryOp<"shru"> {}
def RshLibOp  : ArithBinaryLibraryOp<"rsh"> {}
def SrshLibOp : ArithBinaryLibraryOp<"srsh"> {}
def LshLibOp  : ArithBinaryLibraryOp<"lsh"> {}
def AndLibOp  : ArithBinaryLibraryOp<"and"> {}
def OrLibOp   : ArithBinaryLibraryOp<"or"> {}
def XorLibOp  : ArithBinaryLibraryOp<"xor"> {}

class UnaryLibraryOp<string mnemonic> : CalyxLibraryOp<mnemonic, [
  ]> {
  let results = (outs AnyInteger:$in, AnyInteger:$out);
  let extraClassDeclaration = [{
    Value inPort()  { return getResult(0); }
    Value outPort() { return getResult(1); }
  }];
}

def PadLibOp : UnaryLibraryOp<"pad"> {
  let verifier = [{
    unsigned inBits = getResult(0).getType().getIntOrFloatBitWidth();
    unsigned outBits = getResult(1).getType().getIntOrFloatBitWidth();
    return success(inBits < outBits);
  }];
}

def SliceLibOp : UnaryLibraryOp<"slice"> {
  let verifier = [{
    unsigned inBits = getResult(0).getType().getIntOrFloatBitWidth();
    unsigned outBits = getResult(1).getType().getIntOrFloatBitWidth();
    return success(inBits > outBits);
  }];
}
def NotLibOp  : UnaryLibraryOp<"not"> {}
