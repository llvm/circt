//===- RTGTestRegisters.td - RTGTest Register Definitions --------------------===//
//
// This file defines the dummy test registers.
//
//===----------------------------------------------------------------------===//


include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/EnumAttr.td"

// Flat allocaetion of unique ids to all registers.  The actual id value doesn't
// matter.

def RTGTestRegisters : I32EnumAttr<
    "RTGTestRegisters",
    "Unique IDs for all RTGTest registers",
    [
      I32EnumAttrCase<"i_0",  0>,
      I32EnumAttrCase<"i_1",  1>,
      I32EnumAttrCase<"i_2",  2>,
      I32EnumAttrCase<"i_3",  3>,
      I32EnumAttrCase<"f_0",  4>,
      I32EnumAttrCase<"f_1",  5>,
      I32EnumAttrCase<"f_2",  6>,
      I32EnumAttrCase<"f_3",  7>,
      I32EnumAttrCase<"f_4",  8>,
      I32EnumAttrCase<"f_5",  9>,
      I32EnumAttrCase<"v_0",  10>,
      I32EnumAttrCase<"v_1",  11>,
      I32EnumAttrCase<"MAX",  12> // Must be last
    ]> {
      let cppNamespace = "circt::rtgtest";
    }


def IntegerRegisterType : TypeDef<RTGTestDialect, "IntegerRegister", [
  DeclareTypeInterfaceMethods<ResourceTypeInterface>,
]> {
  let mnemonic = "ireg";
  let assemblyFormat = "";
}

def IntegerRegisterOp : Op<RTGTestDialect, "ireg", [
  DeclareOpInterfaceMethods<RegisterOpInterface>,
]> {
  let arguments = (ins I32Attr:$number);
  let results = (outs IntegerRegisterType:$reg);
  let assemblyFormat = "$number attr-dict";
  let hasVerifier = true;
}

def FloatRegisterType : TypeDef<RTGTestDialect, "FloatRegister", [
  DeclareTypeInterfaceMethods<ResourceTypeInterface>,
]> {
  let mnemonic = "freg";
  let assemblyFormat = "";
}

def FloatRegisterOp : Op<RTGTestDialect, "freg", [
  DeclareOpInterfaceMethods<RegisterOpInterface>,
]> {
  let arguments = (ins I32Attr:$number);
  let results = (outs FloatRegisterType:$reg);
  let assemblyFormat = "$number attr-dict";
  let hasVerifier = true;
}

def VectorRegisterType : TypeDef<RTGTestDialect, "VectorRegister", [
  DeclareTypeInterfaceMethods<ResourceTypeInterface>,
]> {
  let mnemonic = "vreg";
  let assemblyFormat = "";
}

def VectorRegisterOp : Op<RTGTestDialect, "vreg", [
  DeclareOpInterfaceMethods<RegisterOpInterface>,
]> {
  let arguments = (ins I32Attr:$number);
  let results = (outs VectorRegisterType:$reg);
  let assemblyFormat = "$number attr-dict";
  let hasVerifier = true;
}

