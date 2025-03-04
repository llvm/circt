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

include "mlir/IR/BuiltinAttributeInterfaces.td"

/// Base class for Calyx primitives.
class CalyxPrimitive<string mnemonic, list<Trait> traits = []> :
  CalyxCell<mnemonic, traits> {
    let assemblyFormat = "$sym_name attr-dict `:` qualified(type(results))";
    let arguments = (ins SymbolNameAttr:$sym_name);
    let skipDefaultBuilders = 1;
}

def ConstantOp: CalyxPrimitive<"constant",
    [ConstantLike, DeclareOpInterfaceMethods<OpAsmOpInterface, ["getAsmResultNames"]>
    ]> {
  let summary = "constant capable of representing an integer or floating point value";
  let description = [{
    The `constant` operation is a wrapper around bit vectors with fixed-size number of bits.
    Specific value and intended type should be specified via attribute only.

    Example:

    ```
    // Integer constant
    %1 = calyx.constant <42 : i32> : i32

    // Floating point constant
    %1 = calyx.constant <4.2 : f32> : i32
    ```
  }];
  let arguments = (ins SymbolNameAttr:$sym_name, TypedAttrInterface:$value);

  let results = (outs AnySignlessInteger:$out);

  let builders = [
    OpBuilder <(ins "StringRef":$sym_name, "Attribute":$attr, "Type":$type)>,
  ];

  let hasFolder = 1;
  let assemblyFormat = "$sym_name ` ` `<` $value `>` attr-dict `:` qualified(type($out))";
  let hasVerifier = 1;
}

/// The n-bit, undef op which only provides the out signal
def UndefLibOp: CalyxPrimitive<"undefined", []> {
  let summary = "An undefined signal";
  let description = [{
    The "calyx.undef" op defines an undefined signal. An undefined signal can be
    replaced with any value representable in n-bits.
    ```mlir
      // A 32-bit undefined value
      %undef.out = calyx.undef @undef : i32
    ```
  }];
  let arguments = (ins SymbolNameAttr:$sym_name);
  let builders = [
    OpBuilder<(ins "StringRef":$sym_name, "::mlir::TypeRange":$resultTypes), [{
      $_state.addAttribute(mlir::SymbolTable::getSymbolAttrName(), $_builder.getStringAttr(sym_name));
      $_state.addTypes(resultTypes);
    }]>
  ];
  let results = (outs AnyType:$out);
  let extraClassDefinition = [{
    bool $cppClass::isCombinational() { return false; }
    SmallVector<StringRef> $cppClass::portNames() { return {"out"}; }
    SmallVector<Direction> $cppClass::portDirections() { return {Output}; }
    void $cppClass::getAsmResultNames(OpAsmSetValueNameFn setNameFn) {
      getCellAsmResultNames(setNameFn, *this, this->portNames());
    }
    SmallVector<DictionaryAttr> $cppClass::portAttributes() {
      return {DictionaryAttr::get(getContext())};
    }
  }];
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
      %r.in, %r.write_en, %r.clk, %r.reset, %r.out, %r.done = calyx.register @r : i32, i1, i1, i1, i32, i1
    ```
  }];
  let results = (outs AnyType:$in, I1:$write_en, I1:$clk, I1:$reset, AnyType:$out, I1:$done);
  let builders = [
    OpBuilder<(ins "StringRef":$sym_name, "size_t":$width), [{
      $_state.addAttribute(mlir::SymbolTable::getSymbolAttrName(), $_builder.getStringAttr(sym_name));
      Type i1Type = $_builder.getI1Type();
      Type widthType = $_builder.getIntegerType(width);
      $_state.addTypes({widthType, i1Type, i1Type, i1Type, widthType, i1Type});
    }]>,
    OpBuilder<(ins "StringRef":$sym_name, "Type":$type), [{
      $_state.addAttribute(mlir::SymbolTable::getSymbolAttrName(), $_builder.getStringAttr(sym_name));
      Type i1Type = $_builder.getI1Type();
      $_state.addTypes({type, i1Type, i1Type, i1Type, type, i1Type});
    }]>
  ];
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

    See https://docs.calyxir.org/libraries/core.html#memories for
    more information.

    ```mlir
      // A 1-dimensional, 32-bit memory with size dimension 1. Equivalent representation in the native compiler:
      // `m1 = std_mem_d1(32, 1, 1)`
      %m1.addr0, %m1.write_data, %m1.write_en, %m1.clk, %m1.read_data, %m1.done = calyx.memory @m1 <[1] x 32> [1] : i1, i32, i1, i1, i32, i1

      // A 2-dimensional, 8-bit memory with size dimensions 64 x 64. Equivalent representation in the native compiler:
      // `m2 = std_mem_d2(8, 64, 64, 6, 6)`
      %m2.addr0, %m2.addr1, %m2.write_data, %m2.write_en, %m2.clk, %m2.read_data, %m2.done = calyx.memory @m2 <[64, 64] x 8> [6, 6] : i6, i6, i8, i1, i1, i8, i1
    ```
  }];

  let arguments = (ins
    SymbolNameAttr:$sym_name,
    I64Attr:$width,
    ArrayAttr:$sizes,
    ArrayAttr:$addrSizes
  );

  let results = (outs
    Variadic<AnySignlessInteger>:$results
  );

  let assemblyFormat = [{
    $sym_name ` ` `<` $sizes `x` $width `>` $addrSizes attr-dict `:` qualified(type($results))
  }];

  let hasVerifier = 1;

  let builders = [
    OpBuilder<(ins
      "StringRef":$sym_name,
      "int64_t":$width,
      "ArrayRef<int64_t>":$sizes,
      "ArrayRef<int64_t>":$addrSizes
    )>
  ];

  let extraClassDeclaration = [{
    size_t getNumDimensions() { return getAddrSizes().size(); }

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

  let extraClassDefinition = [{
    bool $cppClass::isCombinational() { return false; }
  }];
}

def SeqMemoryOp : CalyxPrimitive<"seq_mem", []> {
  let summary = "Defines a memory with sequential read";
  let description = [{
    The "calyx.seq_mem" op defines a memory with sequential reads. Memories can
    have any number of dimensions, as specified by the length of the `$sizes` and
    `$addrSizes` arrays. The `$addrSizes` specify the bitwidth of each dimension's
    address, and should be wide enough to address the range of the corresponding
    dimension in `$sizes`. The `$width` attribute dictates the width of a single
    element.

    See https://docs.calyxir.org/libraries/core.html#memories for
    more information.

    ```mlir
      // A 1-dimensional, 32-bit memory with size dimension 1. Equivalent representation in the native compiler:
      // `m1 = seq_mem_d1(32, 1, 1)`
      %m1.addr0, %m1.write_data, %m1.write_en, %m1.clk, %m1.read_data, %m1.read_en, %m1.done = calyx.memory @m1 <[1] x 32> [1] : i1, i32, i1, i1, i32, i1

      // A 2-dimensional, 8-bit memory with size dimensions 64 x 64. Equivalent representation in the native compiler:
      // `m2 = seq_mem_d2(8, 64, 64, 6, 6)`
      %m2.addr0, %m2.addr1, %m2.write_data, %m2.write_en, %m2.write_done, %m2.clk, %m2.read_data, %m2.read_en, %m2.read_done = calyx.memory @m2 <[64, 64] x 8> [6, 6] : i6, i6, i8, i1, i1, i8, i1
    ```
  }];

  let arguments = (ins
    SymbolNameAttr:$sym_name,
    I64Attr:$width,
    ArrayAttr:$sizes,
    ArrayAttr:$addrSizes
  );

  let results = (outs
    Variadic<AnySignlessInteger>:$results
  );

  let assemblyFormat = [{
    $sym_name ` ` `<` $sizes `x` $width `>` $addrSizes attr-dict `:` qualified(type($results))
  }];

  let hasVerifier = 1;

  let builders = [
    OpBuilder<(ins
      "StringRef":$sym_name,
      "int64_t":$width,
      "ArrayRef<int64_t>":$sizes,
      "ArrayRef<int64_t>":$addrSizes
    )>
  ];

  let extraClassDeclaration = [{
    size_t getNumDimensions() { return getAddrSizes().size(); }

    ValueRange addrPorts() { return getResults().take_front(getNumDimensions()); }
    Value addrPort(size_t i) {
      assert(i < getNumDimensions() && "index greater than number of memory address ports.");
      return getResults()[i];
    }
    Value clk() { return getResult(getNumDimensions()); }
    Value reset()   { return getResult(getNumDimensions() + 1); }
    Value contentEn()      { return getResult(getNumDimensions() + 2); }
    Value writeEn()       { return getResult(getNumDimensions() + 3); }
    Value writeData()  { return getResult(getNumDimensions() + 4); }
    Value readData()  { return getResult(getNumDimensions() + 5); }
    Value done()      { return getResult(getNumDimensions() + 6); }
  }];

  let extraClassDefinition = [{
    bool $cppClass::isCombinational() { return false; }
  }];
}

class CalyxLibraryOp<string mnemonic, string prefix = "std_", list<Trait> traits = []> :
  CalyxPrimitive<prefix # mnemonic, traits> {

  let summary = "Defines an operation which maps to a Calyx library primitive";
  let description = [{
    This operation represents an instance of a Calyx library primitive.
    A library primitive maps to some hardware-implemented component within the
    native Calyx compiler.

    ```mlir
      // A 32-bit adder. This falls under the binary library operations.
      %add.left, %add.right, %add.out = calyx.std_add @add : i32, i32, i32

      // An 8-bit comparison operator (with a 1-bit output). This falls under
      // the boolean binary library operations.
      %gt.left, %gt.right, %gt.out = calyx.std_gt @gt : i8, i8, i1

      // An 8-bit to 16-bit pad operator. This falls under the unary
      // library operations.
      %pad.in, %pad.out = calyx.std_pad @pad : i8, i16
    ```
  }];

  let arguments = (ins SymbolNameAttr:$sym_name);
  let builders = [
    OpBuilder<(ins "StringRef":$sym_name, "::mlir::TypeRange":$resultTypes), [{
      $_state.addAttribute(mlir::SymbolTable::getSymbolAttrName(), $_builder.getStringAttr(sym_name));
      $_state.addTypes(resultTypes);
    }]>
  ];
}

class BoolBinaryLibraryOp<string mnemonic> : CalyxLibraryOp<mnemonic, "std_", [
    Combinational,
    SameTypeConstraint<"left", "right">
  ]> {
  let results = (outs AnyType:$left, AnyType:$right, I1:$out);
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

class ArithBinaryLibraryOp<string mnemonic, string prefix, list<Trait> traits = []> :
  CalyxLibraryOp<mnemonic, prefix, !listconcat(traits, [
    SameTypeConstraint<"left", "right">
  ])> {}

class CombinationalArithBinaryLibraryOp<string mnemonic> :
  ArithBinaryLibraryOp<mnemonic, "std_", [
    Combinational,
    SameTypeConstraint<"left", "out">
  ]> {
  let results = (outs AnyType:$left, AnyType:$right, AnyType:$out);
}

def AddLibOp  : CombinationalArithBinaryLibraryOp<"add"> {}
def SubLibOp  : CombinationalArithBinaryLibraryOp<"sub"> {}
def ShruLibOp : CombinationalArithBinaryLibraryOp<"shru"> {}
def RshLibOp  : CombinationalArithBinaryLibraryOp<"rsh"> {}
def SrshLibOp : CombinationalArithBinaryLibraryOp<"srsh"> {}
def LshLibOp  : CombinationalArithBinaryLibraryOp<"lsh"> {}
def AndLibOp  : CombinationalArithBinaryLibraryOp<"and"> {}
def OrLibOp   : CombinationalArithBinaryLibraryOp<"or"> {}
def XorLibOp  : CombinationalArithBinaryLibraryOp<"xor"> {}

class ArithBinaryFloatingPointLibraryOp<string mnemonic, list<Trait> traits = []> :
  ArithBinaryLibraryOp<mnemonic, "", !listconcat(traits, [
    DeclareOpInterfaceMethods<FloatingPointOpInterface>,
    SameTypeConstraint<"left", "right">
  ])> {}

def AddFOpIEEE754 : ArithBinaryFloatingPointLibraryOp<"ieee754.add", [
  SameTypeConstraint<"left", "out">
  ]> {
  let results = (outs I1:$clk, I1:$reset, I1:$go, I1:$control, I1:$subOp,
    AnySignlessInteger:$left, AnySignlessInteger:$right, AnySignlessInteger:$roundingMode, AnySignlessInteger:$out,
    AnySignlessInteger:$exceptionalFlags, I1:$done);

  let extraClassDefinition = [{
    SmallVector<StringRef> $cppClass::portNames() {
      return {clkPort, resetPort, goPort, "control", "subOp",
              "left", "right", "roundingMode", "out", "exceptionalFlags", donePort
      };
    }

    SmallVector<Direction> $cppClass::portDirections() {
      return {Input, Input, Input, Input, Input, Input, Input, Input, Output, Output, Output};
    }

    void $cppClass::getAsmResultNames(OpAsmSetValueNameFn setNameFn) {
      getCellAsmResultNames(setNameFn, *this, this->portNames());
    }

    bool $cppClass::isCombinational() { return false; }

    SmallVector<DictionaryAttr> $cppClass::portAttributes() {
      IntegerAttr isSet = IntegerAttr::get(IntegerType::get(getContext(), 1), 1);
      NamedAttrList go, clk, reset, done;
      go.append(goPort, isSet);
      clk.append(clkPort, isSet);
      reset.append(resetPort, isSet);
      done.append(donePort, isSet);
      return {clk.getDictionary(getContext()), reset.getDictionary(getContext()),
              go.getDictionary(getContext()), DictionaryAttr::get(getContext()),
              DictionaryAttr::get(getContext()), DictionaryAttr::get(getContext()),
              DictionaryAttr::get(getContext()), DictionaryAttr::get(getContext()),
              DictionaryAttr::get(getContext()), done.getDictionary(getContext()),
              DictionaryAttr::get(getContext())
      };
    }
  }];
}

def MulFOpIEEE754 : ArithBinaryFloatingPointLibraryOp<"ieee754.mul", [
    SameTypeConstraint<"left", "out">
  ]> {
  let results = (outs I1:$clk, I1:$reset, I1:$go, I1:$control,
    AnySignlessInteger:$left, AnySignlessInteger:$right, AnySignlessInteger:$roundingMode, AnySignlessInteger:$out,
    AnySignlessInteger:$exceptionalFlags, I1:$done);
  let assemblyFormat = "$sym_name attr-dict `:` qualified(type(results))";
  let extraClassDefinition = [{
    SmallVector<StringRef> $cppClass::portNames() {
      return {clkPort, resetPort, goPort, "control", "left", "right",
              "roundingMode", "out", "exceptionalFlags", donePort
      };
    }
    SmallVector<Direction> $cppClass::portDirections() {
      return {Input, Input, Input, Input, Input, Input, Input, Output, Output, Output};
    }
    void $cppClass::getAsmResultNames(OpAsmSetValueNameFn setNameFn) {
      getCellAsmResultNames(setNameFn, *this, this->portNames());
    }
    bool $cppClass::isCombinational() { return false; }
    SmallVector<DictionaryAttr> $cppClass::portAttributes() {
      IntegerAttr isSet = IntegerAttr::get(IntegerType::get(getContext(), 1), 1);
      NamedAttrList go, clk, reset, done;
      go.append(goPort, isSet);
      clk.append(clkPort, isSet);
      reset.append(resetPort, isSet);
      done.append(donePort, isSet);
      return {clk.getDictionary(getContext()), reset.getDictionary(getContext()),
              go.getDictionary(getContext()), DictionaryAttr::get(getContext()),
              DictionaryAttr::get(getContext()), DictionaryAttr::get(getContext()),
              DictionaryAttr::get(getContext()), DictionaryAttr::get(getContext()),
              done.getDictionary(getContext()), DictionaryAttr::get(getContext())
      };
    }
  }];
}

// This models the compare operation interface in Berkeley HardFloat implementation.
def CompareFOpIEEE754 : ArithBinaryFloatingPointLibraryOp<"ieee754.compare", []> {
  let results = (outs I1:$clk, I1:$reset, I1:$go,
    AnySignlessInteger:$left, AnySignlessInteger:$right, I1:$signaling,
    I1:$lt, I1: $eq, I1: $gt, I1: $unordered, AnySignlessInteger: $exceptionalFlags, I1: $done);
  let assemblyFormat = "$sym_name attr-dict `:` qualified(type(results))";
  let extraClassDefinition = [{
    SmallVector<StringRef> $cppClass::portNames() {
      return {clkPort, resetPort, goPort, "left", "right", "signaling",
              "lt", "eq", "gt", "unordered", "exceptionalFlags", donePort
      };
    }
    SmallVector<Direction> $cppClass::portDirections() {
      return {Input, Input, Input, Input, Input, Input, Output, Output, Output, Output, Output, Output};
    }
    void $cppClass::getAsmResultNames(OpAsmSetValueNameFn setNameFn) {
      getCellAsmResultNames(setNameFn, *this, this->portNames());
    }
    bool $cppClass::isCombinational() { return false; }
    SmallVector<DictionaryAttr> $cppClass::portAttributes() {
      IntegerAttr isSet = IntegerAttr::get(IntegerType::get(getContext(), 1), 1);
      NamedAttrList go, clk, reset, done;
      go.append(goPort, isSet);
      clk.append(clkPort, isSet);
      reset.append(resetPort, isSet);
      done.append(donePort, isSet);
      return {clk.getDictionary(getContext()), reset.getDictionary(getContext()),
              go.getDictionary(getContext()), DictionaryAttr::get(getContext()),
              DictionaryAttr::get(getContext()), DictionaryAttr::get(getContext()),
              DictionaryAttr::get(getContext()), DictionaryAttr::get(getContext()),
              DictionaryAttr::get(getContext()), DictionaryAttr::get(getContext()),
              DictionaryAttr::get(getContext()), done.getDictionary(getContext())
      };
    }
  }];
}

def MuxLibOp : CalyxLibraryOp<"mux", "std_", [
  Combinational, SameTypeConstraint<"tru", "fal">, SameTypeConstraint<"tru", "out">
  ]> {
  let results = (outs I1:$cond, AnyType:$tru, AnyType:$fal, AnyType:$out);
  let extraClassDefinition = [{
    SmallVector<StringRef> $cppClass::portNames() {
      return {"cond", "tru", "fal", "out"};
    }
    SmallVector<Direction> $cppClass::portDirections() {
      return {Input, Input, Input, Output};
    }
    void $cppClass::getAsmResultNames(OpAsmSetValueNameFn setNameFn) {
      getCellAsmResultNames(setNameFn, *this, this->portNames());
    }
    bool $cppClass::isCombinational() { return true; }
    SmallVector<DictionaryAttr> $cppClass::portAttributes() {
      return {DictionaryAttr::get(getContext()), DictionaryAttr::get(getContext()),
              DictionaryAttr::get(getContext()), DictionaryAttr::get(getContext())};
    }
  }];
}

class ArithBinaryPipeLibraryOp<string mnemonic> : ArithBinaryLibraryOp<mnemonic # "_pipe", "std_", [
    SameTypeConstraint<"left", "out">
  ]> {
  let results = (outs I1:$clk, I1:$reset, I1:$go, AnyType:$left, AnyType:$right, AnyType:$out, I1:$done);
}

def MultPipeLibOp : ArithBinaryPipeLibraryOp<"mult"> {}
def DivSPipeLibOp : ArithBinaryPipeLibraryOp<"divs"> {}
def DivUPipeLibOp : ArithBinaryPipeLibraryOp<"divu"> {}
def RemUPipeLibOp : ArithBinaryPipeLibraryOp<"remu"> {}
def RemSPipeLibOp : ArithBinaryPipeLibraryOp<"rems"> {}

class UnaryLibraryOp<string mnemonic, string prefix, list<Trait> traits = []> :
    CalyxLibraryOp<mnemonic, prefix, !listconcat(traits, [Combinational])> {
  let results = (outs AnyInteger:$in, AnyInteger:$out);
}

def PadLibOp : UnaryLibraryOp<"pad", "std_"> {
  let hasVerifier = 1;
}

def SliceLibOp : UnaryLibraryOp<"slice", "std_"> {
  let hasVerifier = 1;
}

def NotLibOp  : UnaryLibraryOp<"not", "std_"> {}

def WireLibOp : UnaryLibraryOp<"wire", "std_", [SameTypeConstraint<"in", "out">]> {
  let builders = [
    OpBuilder<(ins "StringRef":$sym_name, "::mlir::Type":$wireType), [{
      llvm::SmallVector<mlir::Type> resultTypes = {wireType, wireType};
      $_state.addAttribute(mlir::SymbolTable::getSymbolAttrName(), $_builder.getStringAttr(sym_name));
      $_state.addTypes(resultTypes);
    }]>
  ];
}

def ExtSILibOp  : UnaryLibraryOp<"signext", "std_"> {}

class UnaryFloatingPointLibraryOp<string mnemonic, list<Trait> traits = []> :
  UnaryLibraryOp<mnemonic, "", !listconcat(traits, [
    DeclareOpInterfaceMethods<FloatingPointOpInterface>
  ])> {}

// This models the floating point to integer conversion interface in Berkeley HardFloat implementation. It rounds to the nearest (towards zero) integer, which is `round_minMag` in the Berkeley HardFloat library.
def FpToIntOpIEEE754 : UnaryFloatingPointLibraryOp<"ieee754.fptoint"> {
  let results = (outs I1:$clk, I1:$reset, I1:$go,
    AnySignlessInteger:$in, I1: $signedOut, AnySignlessInteger:$out, I1:$done);

  let extraClassDefinition = [{
    SmallVector<StringRef> $cppClass::portNames() {
      return {clkPort, resetPort, goPort, "in", "signedOut", "out", donePort};
    }

    SmallVector<Direction> $cppClass::portDirections() {
      return {Input, Input, Input, Input, Input, Output, Output};
    }

    void $cppClass::getAsmResultNames(OpAsmSetValueNameFn setNameFn) {
      getCellAsmResultNames(setNameFn, *this, this->portNames());
    }

    bool $cppClass::isCombinational() { return false; }

    SmallVector<DictionaryAttr> $cppClass::portAttributes() {
      IntegerAttr isSet = IntegerAttr::get(IntegerType::get(getContext(), 1), 1);
      NamedAttrList go, clk, reset, done;
      go.append(goPort, isSet);
      clk.append(clkPort, isSet);
      reset.append(resetPort, isSet);
      done.append(donePort, isSet);
      return {clk.getDictionary(getContext()), reset.getDictionary(getContext()),
              go.getDictionary(getContext()),
              DictionaryAttr::get(getContext()), DictionaryAttr::get(getContext()),
              DictionaryAttr::get(getContext()), done.getDictionary(getContext())
      };
    }
  }];
}
