// RUN: circt-tblgen -gen-rtg-instruction-methods -I %S/../../../include -I %S/../../../llvm/mlir/include %s | FileCheck %s

include "mlir/IR/OpBase.td"
include "mlir/IR/AttrTypeBase.td"
include "circt/Dialect/RTG/IR/RTGDialect.td"
include "circt/Dialect/RTG/IR/RTGTypes.td"
include "circt/Dialect/RTG/IR/RTGISAAssemblyInterfaces.td"

def TestDialect : Dialect {
  let name = "test";
  let cppNamespace = "::test";
}

class TestOp<string mnemonic, list<Trait> traits = []> :
  Op<TestDialect, mnemonic, traits>, ISAInstructionFormat;

def TestRegType : TypeDef<RTGDialect, "TestReg">, RegisterTypeBase {
  let mnemonic = "test_reg";
  let binaryEncodingWidth = 5;
  let pythonName = "TestReg";
  let pythonTypeName = "TestRegType";
}

def Test1_BasicRegisterOp : TestOp<"basic_register"> {
  let arguments = (ins Arg<TestRegType, "", [SourceReg]>:$rs);
  let isaAssemblyFormat = "$rs";
}
// CHECK-LABEL: void BasicRegisterOp::printInstructionAssembly(llvm::raw_ostream &os, FoldAdaptor adaptor) {
// CHECK-NEXT:    os << ::llvm::cast<::circt::rtg::RegisterAttrInterface>(adaptor.getRs()).getRegisterAssembly();
// CHECK-NEXT:  }
// CHECK-EMPTY:
// CHECK-NEXT:  void BasicRegisterOp::printInstructionBinary(llvm::raw_ostream &os, FoldAdaptor adaptor) {
// CHECK-NEXT:    assert(false && "binary not supported");
// CHECK-NEXT:  }

def Test2_MultipleRegistersOp : TestOp<"multiple_registers"> {
  let arguments = (ins Arg<TestRegType, "", [SourceReg]>:$rs1,
                       Arg<TestRegType, "", [SourceReg]>:$rs2);
  let isaAssemblyFormat = "$rs1 `, ` $rs2";
}
// CHECK-EMPTY:
// CHECK-NEXT:  void MultipleRegistersOp::printInstructionAssembly(llvm::raw_ostream &os, FoldAdaptor adaptor) {
// CHECK-NEXT:    os << ::llvm::cast<::circt::rtg::RegisterAttrInterface>(adaptor.getRs1()).getRegisterAssembly();
// CHECK-NEXT:    os << ", ";
// CHECK-NEXT:    os << ::llvm::cast<::circt::rtg::RegisterAttrInterface>(adaptor.getRs2()).getRegisterAssembly();
// CHECK-NEXT:  }
// CHECK-EMPTY:
// CHECK-NEXT:  void MultipleRegistersOp::printInstructionBinary(llvm::raw_ostream &os, FoldAdaptor adaptor) {
// CHECK-NEXT:    assert(false && "binary not supported");
// CHECK-NEXT:  }

def Test3_AnyTypeImmOrLabelOp : TestOp<"anytype_imm_or_label"> {
  let arguments = (ins AnyTypeOf<[ImmediateOfWidth<13>, LabelType]>:$offset);
  let isaAssemblyFormat = "unsigned($offset)";
}
// CHECK-EMPTY:
// CHECK-NEXT:  void AnyTypeImmOrLabelOp::printInstructionAssembly(llvm::raw_ostream &os, FoldAdaptor adaptor) {
// CHECK-NEXT:    if (auto label = ::llvm::dyn_cast<::circt::rtg::LabelAttr>(adaptor.getOffset())) {
// CHECK-NEXT:      os << label.getName();
// CHECK-NEXT:    } else if (auto imm = ::llvm::dyn_cast<::circt::rtg::ImmediateAttr>(adaptor.getOffset())) {
// CHECK-NEXT:      {
// CHECK-NEXT:        ::llvm::SmallVector<char> strBuf;
// CHECK-NEXT:        imm.getValue().toStringUnsigned(strBuf);
// CHECK-NEXT:        os << strBuf;
// CHECK-NEXT:      }
// CHECK-NEXT:    }
// CHECK-NEXT:  }
// CHECK-EMPTY:
// CHECK-NEXT:  void AnyTypeImmOrLabelOp::printInstructionBinary(llvm::raw_ostream &os, FoldAdaptor adaptor) {
// CHECK-NEXT:    assert(false && "binary not supported");
// CHECK-NEXT:  }

def Test4_ImmSlicingOp : TestOp<"imm_slicing"> {
  let arguments = (ins ImmediateOfWidth<20>:$imm,
                       Arg<TestRegType, "", [DestReg]>:$rd);
  let isaBinaryFormat = "$imm[19:12] $imm[11:11] $imm[10:1] $imm[0:0] $rd 0b0110111";
  let isaAssemblyFormat = "$rd `, ` unsigned($imm)";
}
// CHECK-EMPTY:
// CHECK-NEXT:  void ImmSlicingOp::printInstructionAssembly(llvm::raw_ostream &os, FoldAdaptor adaptor) {
// CHECK-NEXT:    os << ::llvm::cast<::circt::rtg::RegisterAttrInterface>(adaptor.getRd()).getRegisterAssembly();
// CHECK-NEXT:    os << ", ";
// CHECK-NEXT:    {
// CHECK-NEXT:      ::llvm::SmallVector<char> strBuf;
// CHECK-NEXT:      ::llvm::cast<::circt::rtg::ImmediateAttr>(adaptor.getImm()).getValue().toStringUnsigned(strBuf);
// CHECK-NEXT:      os << strBuf;
// CHECK-NEXT:    }
// CHECK-NEXT:  }
// CHECK-EMPTY:
// CHECK-NEXT:  void ImmSlicingOp::printInstructionBinary(llvm::raw_ostream &os, FoldAdaptor adaptor) {
// CHECK-NEXT:    ::llvm::APInt imm = ::llvm::cast<::circt::rtg::ImmediateAttr>(adaptor.getImm()).getValue();
// CHECK-NEXT:    ::llvm::APInt rd = ::llvm::APInt(5, ::llvm::cast<::circt::rtg::RegisterAttrInterface>(adaptor.getRd()).getClassIndex());
// CHECK-EMPTY:
// CHECK-NEXT:    ::llvm::APInt binary = llvm::APInt::getZero(0);
// CHECK-NEXT:    binary = binary.concat(imm.extractBits(8, 12));
// CHECK-NEXT:    binary = binary.concat(imm.extractBits(1, 11));
// CHECK-NEXT:    binary = binary.concat(imm.extractBits(10, 1));
// CHECK-NEXT:    binary = binary.concat(imm.extractBits(1, 0));
// CHECK-NEXT:    binary = binary.concat(rd);
// CHECK-NEXT:    binary = binary.concat(llvm::APInt(7, 55));
// CHECK-EMPTY:
// CHECK-NEXT:    ::llvm::SmallVector<char> str;
// CHECK-NEXT:    binary.toStringUnsigned(str, 16);
// CHECK-NEXT:    os << str;
// CHECK-NEXT:  }

def Test5_LabelOnlyOp : TestOp<"label_only"> {
  let arguments = (ins LabelType:$target);
  let isaBinaryFormat = "$target";
  let isaAssemblyFormat = "$target";
}
// CHECK-EMPTY:
// CHECK-NEXT:  void LabelOnlyOp::printInstructionAssembly(llvm::raw_ostream &os, FoldAdaptor adaptor) {
// CHECK-NEXT:    os << ::llvm::cast<::circt::rtg::LabelAttr>(adaptor.getTarget()).getName();
// CHECK-NEXT:  }
// CHECK-EMPTY:
// CHECK-NEXT:  void LabelOnlyOp::printInstructionBinary(llvm::raw_ostream &os, FoldAdaptor adaptor) {
// CHECK-NEXT:    assert(false && "binary not supported");
// CHECK-NEXT:  }

def Test6_MixedRegSlicingOp : TestOp<"mixed_reg_slicing"> {
  let arguments = (ins Arg<TestRegType, "", [SourceReg]>:$rs1,
                       Arg<TestRegType, "", [SourceReg]>:$rs2,
                       Arg<TestRegType, "", [DestReg]>:$rd);
  let isaBinaryFormat = "$rs1[3:0] $rs2 $rd[4:2] 0b111";
  let isaAssemblyFormat = "$rd `, ` $rs1 `, ` $rs2";
}
// CHECK-EMPTY:
// CHECK-NEXT:  void MixedRegSlicingOp::printInstructionAssembly(llvm::raw_ostream &os, FoldAdaptor adaptor) {
// CHECK-NEXT:    os << ::llvm::cast<::circt::rtg::RegisterAttrInterface>(adaptor.getRd()).getRegisterAssembly();
// CHECK-NEXT:    os << ", ";
// CHECK-NEXT:    os << ::llvm::cast<::circt::rtg::RegisterAttrInterface>(adaptor.getRs1()).getRegisterAssembly();
// CHECK-NEXT:    os << ", ";
// CHECK-NEXT:    os << ::llvm::cast<::circt::rtg::RegisterAttrInterface>(adaptor.getRs2()).getRegisterAssembly();
// CHECK-NEXT:  }
// CHECK-EMPTY:
// CHECK-NEXT:  void MixedRegSlicingOp::printInstructionBinary(llvm::raw_ostream &os, FoldAdaptor adaptor) {
// CHECK-NEXT:    ::llvm::APInt rs1 = ::llvm::APInt(5, ::llvm::cast<::circt::rtg::RegisterAttrInterface>(adaptor.getRs1()).getClassIndex());
// CHECK-NEXT:    ::llvm::APInt rs2 = ::llvm::APInt(5, ::llvm::cast<::circt::rtg::RegisterAttrInterface>(adaptor.getRs2()).getClassIndex());
// CHECK-NEXT:    ::llvm::APInt rd = ::llvm::APInt(5, ::llvm::cast<::circt::rtg::RegisterAttrInterface>(adaptor.getRd()).getClassIndex());
// CHECK-EMPTY:
// CHECK-NEXT:    ::llvm::APInt binary = llvm::APInt::getZero(0);
// CHECK-NEXT:    binary = binary.concat(rs1.extractBits(4, 0));
// CHECK-NEXT:    binary = binary.concat(rs2);
// CHECK-NEXT:    binary = binary.concat(rd.extractBits(3, 2));
// CHECK-NEXT:    binary = binary.concat(llvm::APInt(3, 7));
// CHECK-EMPTY:
// CHECK-NEXT:    ::llvm::SmallVector<char> str;
// CHECK-NEXT:    binary.toStringUnsigned(str, 16);
// CHECK-NEXT:    os << str;
// CHECK-NEXT:  }

def Test7_SignedImmediateOp : TestOp<"signed_immediate"> {
  let arguments = (ins ImmediateOfWidth<12>:$imm);
  let isaAssemblyFormat = "signed($imm)";
}
// CHECK-EMPTY:
// CHECK-NEXT:  void SignedImmediateOp::printInstructionAssembly(llvm::raw_ostream &os, FoldAdaptor adaptor) {
// CHECK-NEXT:    {
// CHECK-NEXT:      ::llvm::SmallVector<char> strBuf;
// CHECK-NEXT:      ::llvm::cast<::circt::rtg::ImmediateAttr>(adaptor.getImm()).getValue().toStringSigned(strBuf);
// CHECK-NEXT:      os << strBuf;
// CHECK-NEXT:    }
// CHECK-NEXT:  }
// CHECK-EMPTY:
// CHECK-NEXT:  void SignedImmediateOp::printInstructionBinary(llvm::raw_ostream &os, FoldAdaptor adaptor) {
// CHECK-NEXT:    assert(false && "binary not supported");
// CHECK-NEXT:  }

// Test: String literal in assembly format
def Test8_StringLiteralOp : TestOp<"string_literal"> {
  let isaAssemblyFormat = "` ` `0x`";
}
// CHECK-EMPTY:
// CHECK-NEXT:  void StringLiteralOp::printInstructionAssembly(llvm::raw_ostream &os, FoldAdaptor adaptor) {
// CHECK-NEXT:    os << " ";
// CHECK-NEXT:    os << "0x";
// CHECK-NEXT:  }
// CHECK-EMPTY:
// CHECK-NEXT:  void StringLiteralOp::printInstructionBinary(llvm::raw_ostream &os, FoldAdaptor adaptor) {
// CHECK-NEXT:    assert(false && "binary not supported");
// CHECK-NEXT:  }

def Test9_AnyTypeRegOrImmOp : TestOp<"anytype_reg_or_imm"> {
  let arguments = (ins Arg<TestRegType, "", [DestReg]>:$rd,
                       AnyTypeOf<[TestRegType, ImmediateOfWidth<5>]>:$src);
  let isaAssemblyFormat = "unsigned($src)";
}
// CHECK-EMPTY:
// CHECK-NEXT:  void AnyTypeRegOrImmOp::printInstructionAssembly(llvm::raw_ostream &os, FoldAdaptor adaptor) {
// CHECK-NEXT:    if (auto reg = ::llvm::dyn_cast<::circt::rtg::RegisterAttrInterface>(adaptor.getSrc())) {
// CHECK-NEXT:      os << reg.getRegisterAssembly();
// CHECK-NEXT:    } else if (auto imm = ::llvm::dyn_cast<::circt::rtg::ImmediateAttr>(adaptor.getSrc())) {
// CHECK-NEXT:      {
// CHECK-NEXT:        ::llvm::SmallVector<char> strBuf;
// CHECK-NEXT:        imm.getValue().toStringUnsigned(strBuf);
// CHECK-NEXT:        os << strBuf;
// CHECK-NEXT:      }
// CHECK-NEXT:    }
// CHECK-NEXT:  }
// CHECK-EMPTY:
// CHECK-NEXT:  void AnyTypeRegOrImmOp::printInstructionBinary(llvm::raw_ostream &os, FoldAdaptor adaptor) {
// CHECK-NEXT:    assert(false && "binary not supported");
// CHECK-NEXT:  }

// Test: Single-bit slicing in binary format
def Test10_SingleBitSliceOp : TestOp<"single_bit_slice"> {
  let arguments = (ins ImmediateOfWidth<8>:$imm);
  let isaBinaryFormat = "$imm[7] $imm[0] 0b11";
  let isaAssemblyFormat = "unsigned($imm)";
}
// CHECK-LABEL:  void SingleBitSliceOp::printInstructionBinary
// CHECK-NEXT:    ::llvm::APInt imm = ::llvm::cast<::circt::rtg::ImmediateAttr>(adaptor.getImm()).getValue();
// CHECK-EMPTY:
// CHECK-NEXT:    ::llvm::APInt binary = llvm::APInt::getZero(0);
// CHECK-NEXT:    binary = binary.concat(imm.extractBits(1, 7));
// CHECK-NEXT:    binary = binary.concat(imm.extractBits(1, 0));
// CHECK-NEXT:    binary = binary.concat(llvm::APInt(2, 3));
// CHECK-EMPTY:
// CHECK-NEXT:    ::llvm::SmallVector<char> str;
// CHECK-NEXT:    binary.toStringUnsigned(str, 16);
// CHECK-NEXT:    os << str;
// CHECK-NEXT:  }

def Test11_DecimalLiteralOp : TestOp<"decimal_literal"> {
  let isaBinaryFormat = "3d1 7d55";
  let isaAssemblyFormat = "`decimal-literal`";
}
// CHECK-LABEL:  void DecimalLiteralOp::printInstructionBinary
// CHECK:    ::llvm::APInt binary = llvm::APInt::getZero(0);
// CHECK-NEXT:    binary = binary.concat(llvm::APInt(3, 1));
// CHECK-NEXT:    binary = binary.concat(llvm::APInt(7, 55));

def Test12_IfThenElseBothBranchesOp : TestOp<"if_then_else_both"> {
  let arguments = (ins ImmediateOfWidth<1>:$cond,
                       Arg<TestRegType, "", [DestReg]>:$rd,
                       Arg<TestRegType, "", [DestReg]>:$rs);
  let isaBinaryFormat = "if $cond then ($rd 0b11) else ($rd $rs 0b00)";
  let isaAssemblyFormat = "if $cond then (`+` $rd) else (`-` $rd $rs)";
}
// CHECK-LABEL:  void IfThenElseBothBranchesOp::printInstructionAssembly
// CHECK-NEXT:    ::llvm::APInt cond = ::llvm::cast<::circt::rtg::ImmediateAttr>(adaptor.getCond()).getValue();
// CHECK-NEXT:    if (cond.isOne()) {
// CHECK-NEXT:      os << "+";
// CHECK-NEXT:      os << ::llvm::cast<::circt::rtg::RegisterAttrInterface>(adaptor.getRd()).getRegisterAssembly();
// CHECK-NEXT:    } else {
// CHECK-NEXT:      os << "-";
// CHECK-NEXT:      os << ::llvm::cast<::circt::rtg::RegisterAttrInterface>(adaptor.getRd()).getRegisterAssembly();
// CHECK-NEXT:      os << ::llvm::cast<::circt::rtg::RegisterAttrInterface>(adaptor.getRs()).getRegisterAssembly();
// CHECK-NEXT:    }
// CHECK-NEXT:  }

// CHECK-LABEL:  void IfThenElseBothBranchesOp::printInstructionBinary
// CHECK-NEXT:    ::llvm::APInt cond = ::llvm::cast<::circt::rtg::ImmediateAttr>(adaptor.getCond()).getValue();
// CHECK-NEXT:    ::llvm::APInt rd = ::llvm::APInt(5, ::llvm::cast<::circt::rtg::RegisterAttrInterface>(adaptor.getRd()).getClassIndex());
// CHECK-NEXT:    ::llvm::APInt rs = ::llvm::APInt(5, ::llvm::cast<::circt::rtg::RegisterAttrInterface>(adaptor.getRs()).getClassIndex());
// CHECK-EMPTY:
// CHECK-NEXT:    ::llvm::APInt binary = llvm::APInt::getZero(0);
// CHECK-NEXT:    if (cond.isOne()) {
// CHECK-NEXT:      binary = binary.concat(rd);
// CHECK-NEXT:      binary = binary.concat(llvm::APInt(2, 3));
// CHECK-NEXT:    } else {
// CHECK-NEXT:      binary = binary.concat(rd);
// CHECK-NEXT:      binary = binary.concat(rs);
// CHECK-NEXT:      binary = binary.concat(llvm::APInt(2, 0));
// CHECK-NEXT:    }

def Test13_IfThenOnlyOp : TestOp<"if_then_only"> {
  let arguments = (ins ImmediateOfWidth<1>:$cond,
                       Arg<TestRegType, "", [DestReg]>:$rd);
  let isaAssemblyFormat = "if $cond then (`opt:` $rd) else ()";
}
// CHECK-LABEL:  void IfThenOnlyOp::printInstructionAssembly
// CHECK-NEXT:    ::llvm::APInt cond = ::llvm::cast<::circt::rtg::ImmediateAttr>(adaptor.getCond()).getValue();
// CHECK-NEXT:    if (cond.isOne()) {
// CHECK-NEXT:      os << "opt:";
// CHECK-NEXT:      os << ::llvm::cast<::circt::rtg::RegisterAttrInterface>(adaptor.getRd()).getRegisterAssembly();
// CHECK-NEXT:    }
// CHECK-NEXT:  }

def Test14_IfElseOnlyOp : TestOp<"if_else_only"> {
  let arguments = (ins ImmediateOfWidth<1>:$cond,
                       Arg<TestRegType, "", [DestReg]>:$rd);
  let isaAssemblyFormat = "if $cond else (`alt:` $rd)";
}
// CHECK-LABEL:  void IfElseOnlyOp::printInstructionAssembly
// CHECK-NEXT:    ::llvm::APInt cond = ::llvm::cast<::circt::rtg::ImmediateAttr>(adaptor.getCond()).getValue();
// CHECK-NEXT:    if (cond.isZero()) {
// CHECK-NEXT:      os << "alt:";
// CHECK-NEXT:      os << ::llvm::cast<::circt::rtg::RegisterAttrInterface>(adaptor.getRd()).getRegisterAssembly();
// CHECK-NEXT:    }
// CHECK-NEXT:  }

def Test17_CustomDirectiveOp : TestOp<"custom_directive"> {
  let arguments = (ins Arg<TestRegType, "", [SourceReg]>:$rs1,
                       Arg<TestRegType, "", [SourceReg]>:$rd,
                       ImmediateOfWidth<12>:$imm);
  let isaBinaryFormat = "custom<myFunc1>($rs1 $rd)";
  let isaAssemblyFormat = "custom<myFunc2>($rs1 $imm)";
}
// CHECK-LABEL:  void CustomDirectiveOp::printInstructionAssembly
// CHECK-NEXT:    myFunc2(adaptor.getRs1(), adaptor.getImm(), os);

// CHECK-LABEL:  void CustomDirectiveOp::printInstructionBinary
// CHECK:    binary = binary.concat(myFunc1(adaptor.getRs1(), adaptor.getRd()));

def Test18_CustomDirectiveNoArgumentOp : TestOp<"custom_directive_no_argument"> {
  let isaAssemblyFormat = "custom<myFunc3>()";
}
// CHECK-LABEL:  void CustomDirectiveNoArgumentOp::printInstructionAssembly
// CHECK: myFunc3(os);

def Test19_FallibleCustomDirectiveOp : TestOp<"fallible_custom_directive"> {
  let arguments = (ins Arg<TestRegType, "", [SourceReg]>:$rs1,
                       ImmediateOfWidth<12>:$imm);
  let isaAssemblyFormat = "custom<myFallibleFunc>?($rs1 $imm)";
}
// CHECK-LABEL:  void FallibleCustomDirectiveOp::printInstructionAssembly
// CHECK-NEXT:    if (!myFallibleFuncPossible(adaptor.getRs1(), adaptor.getImm())) {
// CHECK-NEXT:      os << ".word 0x";
// CHECK-NEXT:      printInstructionBinary(os, adaptor);
// CHECK-NEXT:      return;
// CHECK-NEXT:    }
// CHECK-NEXT:    myFallibleFunc(adaptor.getRs1(), adaptor.getImm(), os);

def Test20_FallibleCustomDirectiveNoArgumentOp : TestOp<"fallible_custom_directive_no_argument"> {
  let isaAssemblyFormat = "custom<myFallibleFunc2>?() ` ` custom<myFallibleFunc3>?()";
}
// CHECK-LABEL:  void FallibleCustomDirectiveNoArgumentOp::printInstructionAssembly
// CHECK-NEXT:    if (!myFallibleFunc2Possible() || !myFallibleFunc3Possible()) {
// CHECK-NEXT:      os << ".word 0x";
// CHECK-NEXT:      printInstructionBinary(os, adaptor);
// CHECK-NEXT:      return;
// CHECK-NEXT:    }
// CHECK-NEXT:    myFallibleFunc2(os);
// CHECK-NEXT:    os << " ";
// CHECK-NEXT:    myFallibleFunc3(os);

// Test: Operand only used in custom directive should NOT get a declaration
def Test21_CustomDirectiveOnlyOp : TestOp<"custom_directive_only"> {
  let arguments = (ins Arg<TestRegType, "", [SourceReg]>:$rs1);
  let isaBinaryFormat = "custom<encodeReg>($rs1)";
  let isaAssemblyFormat = "custom<printReg>($rs1)";
}
// CHECK-LABEL:  void CustomDirectiveOnlyOp::printInstructionBinary
// CHECK-NOT:     ::llvm::APInt rs1
// CHECK:         binary = binary.concat(encodeReg(adaptor.getRs1()));

def Test22_AnyImmediateOp : TestOp<"any_immediate"> {
  let arguments = (ins ImmediateType:$imm);
  let isaAssemblyFormat = "signed($imm)";
}
// CHECK-LABEL: void AnyImmediateOp::printInstructionAssembly
//      CHECK:   ::llvm::SmallVector<char> strBuf;
// CHECK-NEXT:   ::llvm::cast<::circt::rtg::ImmediateAttr>(adaptor.getImm()).getValue().toStringSigned(strBuf);
// CHECK-NEXT:   os << strBuf;

def Test23_MultipleSpacesOp : TestOp<"multiple_spaces"> {
  let isaAssemblyFormat = "`  ` \t  `,  `";
}
// CHECK-LABEL:  void MultipleSpacesOp::printInstructionAssembly
// CHECK-NEXT:    os << "  ";
// CHECK-NEXT:    os << ",  ";

def Test24_MultiLineFormatOp : TestOp<"multi_line_format"> {
  let isaBinaryFormat = [{
    0b001 
    0b010
    0b011
  }];
  let isaAssemblyFormat = "`a`\n`b`\n";
}
// CHECK-LABEL:  void MultiLineFormatOp::printInstructionAssembly
// CHECK-NEXT:    os << "a";
// CHECK-NEXT:    os << "b";

// CHECK-LABEL:  void MultiLineFormatOp::printInstructionBinary
//      CHECK:    binary = binary.concat(llvm::APInt(3, 1));
// CHECK-NEXT:    binary = binary.concat(llvm::APInt(3, 2));
// CHECK-NEXT:    binary = binary.concat(llvm::APInt(3, 3));

// Test: Binary literal followed by tab
def Test25_BinaryLiteralTabOp : TestOp<"binary_literal_tab"> {
  let isaBinaryFormat = "0b111  0b000";
  let isaAssemblyFormat = "` `";
}
// CHECK-LABEL:  void BinaryLiteralTabOp::printInstructionBinary
// CHECK:         binary = binary.concat(llvm::APInt(3, 7));
// CHECK-NEXT:    binary = binary.concat(llvm::APInt(3, 0));

def Test26_IfThenElseNewlinesOp : TestOp<"if_then_else_newlines"> {
  let arguments = (ins ImmediateOfWidth<1>:$cond);
  let isaAssemblyFormat = "if\n$cond\nthen\n(`a`\n`b`)\nelse\n( `c` )";
}
// CHECK-LABEL:  void IfThenElseNewlinesOp::printInstructionAssembly
// CHECK:         ::llvm::APInt cond = ::llvm::cast<::circt::rtg::ImmediateAttr>(adaptor.getCond()).getValue();
// CHECK:         if (cond.isOne()) {
// CHECK-NEXT:      os << "a";
// CHECK-NEXT:      os << "b";
// CHECK-NEXT:    } else {
// CHECK-NEXT:      os << "c";

def Test27_CustomDirectiveNewlinesOp : TestOp<"custom_directive_newlines"> {
  let arguments = (ins Arg<TestRegType, "", [SourceReg]>:$rs1,
                       Arg<TestRegType, "", [SourceReg]>:$rs2);
  let isaAssemblyFormat = "custom<myFunc>($rs1\n$rs2)";
}
// CHECK-LABEL:  void CustomDirectiveNewlinesOp::printInstructionAssembly
// CHECK:         myFunc(adaptor.getRs1(), adaptor.getRs2(), os);

def Test28_SignednessNewlineOp : TestOp<"signedness_newline"> {
  let arguments = (ins ImmediateOfWidth<12>:$imm);
  let isaAssemblyFormat = "` `\nsigned($imm)";
}
// CHECK-LABEL:  void SignednessNewlineOp::printInstructionAssembly
// CHECK-NEXT:    os << " ";
// CHECK:         ::llvm::SmallVector<char> strBuf;
// CHECK-NEXT:    ::llvm::cast<::circt::rtg::ImmediateAttr>(adaptor.getImm()).getValue().toStringSigned(strBuf);
