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
}

def Test1_BasicRegisterOp : TestOp<"basic_register"> {
  let arguments = (ins Arg<TestRegType, "", [SourceReg]>:$rs);
  let isaAssemblyFormat = "mnemonic ` ` $rs";
}
// CHECK-LABEL: void BasicRegisterOp::printInstructionAssembly(llvm::raw_ostream &os, FoldAdaptor adaptor) {
// CHECK-NEXT:    os << getOperationName().rsplit('.').second;
// CHECK-NEXT:    os << " ";
// CHECK-NEXT:    os << cast<rtg::RegisterAttrInterface>(adaptor.getRs()).getRegisterAssembly();
// CHECK-NEXT:  }
// CHECK-EMPTY:
// CHECK-NEXT:  void BasicRegisterOp::printInstructionBinary(llvm::raw_ostream &os, FoldAdaptor adaptor) {
// CHECK-NEXT:    assert(false && "binary not supported");
// CHECK-NEXT:  }

def Test2_MultipleRegistersOp : TestOp<"multiple_registers"> {
  let arguments = (ins Arg<TestRegType, "", [SourceReg]>:$rs1,
                       Arg<TestRegType, "", [SourceReg]>:$rs2);
  let isaAssemblyFormat = "mnemonic ` ` $rs1 `, ` $rs2";
}
// CHECK-EMPTY:
// CHECK-NEXT:  void MultipleRegistersOp::printInstructionAssembly(llvm::raw_ostream &os, FoldAdaptor adaptor) {
// CHECK-NEXT:    os << getOperationName().rsplit('.').second;
// CHECK-NEXT:    os << " ";
// CHECK-NEXT:    os << cast<rtg::RegisterAttrInterface>(adaptor.getRs1()).getRegisterAssembly();
// CHECK-NEXT:    os << ", ";
// CHECK-NEXT:    os << cast<rtg::RegisterAttrInterface>(adaptor.getRs2()).getRegisterAssembly();
// CHECK-NEXT:  }
// CHECK-EMPTY:
// CHECK-NEXT:  void MultipleRegistersOp::printInstructionBinary(llvm::raw_ostream &os, FoldAdaptor adaptor) {
// CHECK-NEXT:    assert(false && "binary not supported");
// CHECK-NEXT:  }

def Test3_AnyTypeImmOrLabelOp : TestOp<"anytype_imm_or_label"> {
  let arguments = (ins AnyTypeOf<[ImmediateOfWidth<13>, LabelType]>:$offset);
  let isaAssemblyFormat = "mnemonic ` ` unsigned($offset)";
}
// CHECK-EMPTY:
// CHECK-NEXT:  void AnyTypeImmOrLabelOp::printInstructionAssembly(llvm::raw_ostream &os, FoldAdaptor adaptor) {
// CHECK-NEXT:    os << getOperationName().rsplit('.').second;
// CHECK-NEXT:    os << " ";
// CHECK-NEXT:    if (auto label = dyn_cast<rtg::LabelAttr>(adaptor.getOffset())) {
// CHECK-NEXT:      os << label.getName();
// CHECK-NEXT:    } else if (auto imm = dyn_cast<rtg::ImmediateAttr>(adaptor.getOffset())) {
// CHECK-NEXT:      {
// CHECK-NEXT:        SmallVector<char> strBuf;
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
  let isaAssemblyFormat = "mnemonic ` ` $rd `, ` unsigned($imm)";
}
// CHECK-EMPTY:
// CHECK-NEXT:  void ImmSlicingOp::printInstructionAssembly(llvm::raw_ostream &os, FoldAdaptor adaptor) {
// CHECK-NEXT:    os << getOperationName().rsplit('.').second;
// CHECK-NEXT:    os << " ";
// CHECK-NEXT:    os << cast<rtg::RegisterAttrInterface>(adaptor.getRd()).getRegisterAssembly();
// CHECK-NEXT:    os << ", ";
// CHECK-NEXT:    {
// CHECK-NEXT:      SmallVector<char> strBuf;
// CHECK-NEXT:      cast<rtg::ImmediateAttr>(adaptor.getImm()).getValue().toStringUnsigned(strBuf);
// CHECK-NEXT:      os << strBuf;
// CHECK-NEXT:    }
// CHECK-NEXT:  }
// CHECK-EMPTY:
// CHECK-NEXT:  void ImmSlicingOp::printInstructionBinary(llvm::raw_ostream &os, FoldAdaptor adaptor) {
// CHECK-NEXT:    llvm::APInt imm = cast<rtg::ImmediateAttr>(adaptor.getImm()).getValue();
// CHECK-NEXT:    llvm::APInt rd = llvm::APInt(5, cast<rtg::RegisterAttrInterface>(adaptor.getRd()).getClassIndex());
// CHECK-EMPTY:  
// CHECK-NEXT:    llvm::APInt binary = llvm::APInt::getZero(0);
// CHECK-NEXT:    binary = binary.concat(imm.extractBits(8, 12));
// CHECK-NEXT:    binary = binary.concat(imm.extractBits(1, 11));
// CHECK-NEXT:    binary = binary.concat(imm.extractBits(10, 1));
// CHECK-NEXT:    binary = binary.concat(imm.extractBits(1, 0));
// CHECK-NEXT:    binary = binary.concat(rd);
// CHECK-NEXT:    binary = binary.concat(llvm::APInt(7, 55));
// CHECK-EMPTY:  
// CHECK-NEXT:    llvm::SmallVector<char> str;
// CHECK-NEXT:    binary.toStringUnsigned(str, 16);
// CHECK-NEXT:    os << str;
// CHECK-NEXT:  }

def Test5_LabelOnlyOp : TestOp<"label_only"> {
  let arguments = (ins LabelType:$target);
  let isaAssemblyFormat = "mnemonic ` ` $target";
}
// CHECK-EMPTY:
// CHECK-NEXT:  void LabelOnlyOp::printInstructionAssembly(llvm::raw_ostream &os, FoldAdaptor adaptor) {
// CHECK-NEXT:    os << getOperationName().rsplit('.').second;
// CHECK-NEXT:    os << " ";
// CHECK-NEXT:    os << cast<rtg::LabelAttr>(adaptor.getTarget()).getName();
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
  let isaAssemblyFormat = "mnemonic ` ` $rd `, ` $rs1 `, ` $rs2";
}
// CHECK-EMPTY:
// CHECK-NEXT:  void MixedRegSlicingOp::printInstructionAssembly(llvm::raw_ostream &os, FoldAdaptor adaptor) {
// CHECK-NEXT:    os << getOperationName().rsplit('.').second;
// CHECK-NEXT:    os << " ";
// CHECK-NEXT:    os << cast<rtg::RegisterAttrInterface>(adaptor.getRd()).getRegisterAssembly();
// CHECK-NEXT:    os << ", ";
// CHECK-NEXT:    os << cast<rtg::RegisterAttrInterface>(adaptor.getRs1()).getRegisterAssembly();
// CHECK-NEXT:    os << ", ";
// CHECK-NEXT:    os << cast<rtg::RegisterAttrInterface>(adaptor.getRs2()).getRegisterAssembly();
// CHECK-NEXT:  }
// CHECK-EMPTY:
// CHECK-NEXT:  void MixedRegSlicingOp::printInstructionBinary(llvm::raw_ostream &os, FoldAdaptor adaptor) {
// CHECK-NEXT:    llvm::APInt rs1 = llvm::APInt(5, cast<rtg::RegisterAttrInterface>(adaptor.getRs1()).getClassIndex());
// CHECK-NEXT:    llvm::APInt rs2 = llvm::APInt(5, cast<rtg::RegisterAttrInterface>(adaptor.getRs2()).getClassIndex());
// CHECK-NEXT:    llvm::APInt rd = llvm::APInt(5, cast<rtg::RegisterAttrInterface>(adaptor.getRd()).getClassIndex());
// CHECK-EMPTY:
// CHECK-NEXT:    llvm::APInt binary = llvm::APInt::getZero(0);
// CHECK-NEXT:    binary = binary.concat(rs1.extractBits(4, 0));
// CHECK-NEXT:    binary = binary.concat(rs2);
// CHECK-NEXT:    binary = binary.concat(rd.extractBits(3, 2));
// CHECK-NEXT:    binary = binary.concat(llvm::APInt(3, 7));
// CHECK-EMPTY:
// CHECK-NEXT:    llvm::SmallVector<char> str;
// CHECK-NEXT:    binary.toStringUnsigned(str, 16);
// CHECK-NEXT:    os << str;
// CHECK-NEXT:  }

def Test7_SignedImmediateOp : TestOp<"signed_immediate"> {
  let arguments = (ins ImmediateOfWidth<12>:$imm,
                       Arg<TestRegType, "", [DestReg]>:$rd);
  let isaAssemblyFormat = "mnemonic ` ` $rd `, ` signed($imm)";
}
// CHECK-EMPTY:
// CHECK-NEXT:  void SignedImmediateOp::printInstructionAssembly(llvm::raw_ostream &os, FoldAdaptor adaptor) {
// CHECK-NEXT:    os << getOperationName().rsplit('.').second;
// CHECK-NEXT:    os << " ";
// CHECK-NEXT:    os << cast<rtg::RegisterAttrInterface>(adaptor.getRd()).getRegisterAssembly();
// CHECK-NEXT:    os << ", ";
// CHECK-NEXT:    { 
// CHECK-NEXT:      SmallVector<char> strBuf; 
// CHECK-NEXT:      cast<rtg::ImmediateAttr>(adaptor.getImm()).getValue().toStringSigned(strBuf); 
// CHECK-NEXT:      os << strBuf; 
// CHECK-NEXT:    }
// CHECK-NEXT:  }
// CHECK-EMPTY:
// CHECK-NEXT:  void SignedImmediateOp::printInstructionBinary(llvm::raw_ostream &os, FoldAdaptor adaptor) {
// CHECK-NEXT:    assert(false && "binary not supported");
// CHECK-NEXT:  }

// Test: String literal in assembly format
def Test8_StringLiteralOp : TestOp<"string_literal"> {
  let arguments = (ins Arg<TestRegType, "", [DestReg]>:$rd);
  let isaAssemblyFormat = "mnemonic ` ` `0x` $rd";
}
// CHECK-EMPTY:
// CHECK-NEXT:  void StringLiteralOp::printInstructionAssembly(llvm::raw_ostream &os, FoldAdaptor adaptor) {
// CHECK-NEXT:    os << getOperationName().rsplit('.').second;
// CHECK-NEXT:    os << " ";
// CHECK-NEXT:    os << "0x";
// CHECK-NEXT:    os << cast<rtg::RegisterAttrInterface>(adaptor.getRd()).getRegisterAssembly();
// CHECK-NEXT:  }
// CHECK-EMPTY:
// CHECK-NEXT:  void StringLiteralOp::printInstructionBinary(llvm::raw_ostream &os, FoldAdaptor adaptor) {
// CHECK-NEXT:    assert(false && "binary not supported");
// CHECK-NEXT:  }

def Test9_AnyTypeRegOrImmOp : TestOp<"anytype_reg_or_imm"> {
  let arguments = (ins Arg<TestRegType, "", [DestReg]>:$rd,
                       AnyTypeOf<[TestRegType, ImmediateOfWidth<5>]>:$src);
  let isaAssemblyFormat = "mnemonic ` ` $rd `, ` unsigned($src)";
}
// CHECK-EMPTY:
// CHECK-NEXT:  void AnyTypeRegOrImmOp::printInstructionAssembly(llvm::raw_ostream &os, FoldAdaptor adaptor) {
// CHECK-NEXT:    os << getOperationName().rsplit('.').second;
// CHECK-NEXT:    os << " ";
// CHECK-NEXT:    os << cast<rtg::RegisterAttrInterface>(adaptor.getRd()).getRegisterAssembly();
// CHECK-NEXT:    os << ", ";
// CHECK-NEXT:    if (auto reg = dyn_cast<rtg::RegisterAttrInterface>(adaptor.getSrc())) {
// CHECK-NEXT:      os << reg.getRegisterAssembly();
// CHECK-NEXT:    } else if (auto imm = dyn_cast<rtg::ImmediateAttr>(adaptor.getSrc())) {
// CHECK-NEXT:      { 
// CHECK-NEXT:        SmallVector<char> strBuf; 
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
  let isaAssemblyFormat = "mnemonic ` ` unsigned($imm)";
}
// CHECK-EMPTY:
// CHECK-NEXT:  void SingleBitSliceOp::printInstructionAssembly(llvm::raw_ostream &os, FoldAdaptor adaptor) {
// CHECK-NEXT:    os << getOperationName().rsplit('.').second;
// CHECK-NEXT:    os << " ";
// CHECK-NEXT:    {
// CHECK-NEXT:    SmallVector<char> strBuf;
// CHECK-NEXT:    cast<rtg::ImmediateAttr>(adaptor.getImm()).getValue().toStringUnsigned(strBuf);
// CHECK-NEXT:    os << strBuf;
// CHECK-NEXT:    }
// CHECK-NEXT:  }
// CHECK-EMPTY:
// CHECK-NEXT:  void SingleBitSliceOp::printInstructionBinary(llvm::raw_ostream &os, FoldAdaptor adaptor) {
// CHECK-NEXT:    llvm::APInt imm = cast<rtg::ImmediateAttr>(adaptor.getImm()).getValue();
// CHECK-EMPTY:  
// CHECK-NEXT:    llvm::APInt binary = llvm::APInt::getZero(0);
// CHECK-NEXT:    binary = binary.concat(imm.extractBits(1, 7));
// CHECK-NEXT:    binary = binary.concat(imm.extractBits(1, 0));
// CHECK-NEXT:    binary = binary.concat(llvm::APInt(2, 3));
// CHECK-EMPTY:
// CHECK-NEXT:    llvm::SmallVector<char> str;
// CHECK-NEXT:    binary.toStringUnsigned(str, 16);
// CHECK-NEXT:    os << str;
// CHECK-NEXT:  }
