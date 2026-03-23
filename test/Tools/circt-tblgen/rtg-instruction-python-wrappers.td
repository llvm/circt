// RUN: circt-tblgen -gen-rtg-instruction-python-wrappers --return-val-func=mock_return_val_func -I %S/../../../include -I %S/../../../llvm/mlir/include %s | FileCheck %s
// RUN: circt-tblgen -gen-rtg-instruction-python-wrappers --return-val-func=mock_return_val_func --header-file=%S/rtg-instruction-python-wrappers-header.py -I %S/../../../include -I %S/../../../llvm/mlir/include %s | FileCheck %s --check-prefix=HEADER

include "mlir/IR/OpBase.td"
include "mlir/IR/AttrTypeBase.td"
include "circt/Dialect/RTG/IR/RTGDialect.td"
include "circt/Dialect/RTG/IR/RTGTypes.td"
include "circt/Dialect/RTG/IR/RTGISAAssemblyInterfaces.td"

def TestDialect : Dialect {
  let name = "test";
  let cppNamespace = "::test";
}

class TestOp<string mnemonic> :
  Op<TestDialect, mnemonic, [InstructionOpAdaptor]>;

def TestRegType : TypeDef<RTGDialect, "TestReg", [
  DeclareTypeInterfaceMethods<RegisterTypeInterface>
]>, RegisterTypeBase {
  let mnemonic = "test_reg";
  let binaryEncodingWidth = 5;
  let pythonName = "TestReg";
  let pythonTypeName = "TestRegType";
}

def TestFloatRegType : TypeDef<RTGDialect, "TestFloatReg", [
  DeclareTypeInterfaceMethods<RegisterTypeInterface>
]>, RegisterTypeBase {
  let mnemonic = "test_freg";
  let binaryEncodingWidth = 5;
  let pythonName = "TestFloatReg";
  let pythonTypeName = "TestFloatRegType";
}

// HEADER: from my_isa import TestRegType
// HEADER-EMPTY:
// HEADER-EMPTY:
// HEADER-NEXT: def mock_return_val_func():
// HEADER-NEXT:   pass
// HEADER-EMPTY:
// HEADER-NEXT: @instruction(return_val_func=mock_return_val_func,
// HEADER-NEXT:              args=[(TestRegType(), SideEffect.READ)])
// HEADER-NEXT: def source_reg(rs: TestReg):
// HEADER-NEXT:   test.SourceRegOp(rs)

def Test1_SourceRegOp : TestOp<"source_reg"> {
  let arguments = (ins Arg<TestRegType, "", [SourceReg]>:$rs);
}
// CHECK: @instruction(return_val_func=mock_return_val_func,
// CHECK-NEXT:              args=[(TestRegType(), SideEffect.READ)])
// CHECK-NEXT: def source_reg(rs: TestReg):
// CHECK-NEXT:   test.SourceRegOp(rs)
// CHECK-EMPTY:

def Test2_DestRegOp : TestOp<"dest_reg"> {
  let arguments = (ins Arg<TestRegType, "", [DestReg]>:$rd);
}
// CHECK-NEXT: @instruction(return_val_func=mock_return_val_func,
// CHECK-NEXT:              args=[(TestRegType(), SideEffect.WRITE)])
// CHECK-NEXT: def dest_reg(rd: TestReg):
// CHECK-NEXT:   test.DestRegOp(rd)
// CHECK-EMPTY:

def Test3_ReadWriteRegOp : TestOp<"read_write_reg"> {
  let arguments = (ins Arg<TestRegType, "", [SourceReg, DestReg]>:$rd_rs);
}
// CHECK-NEXT: @instruction(return_val_func=mock_return_val_func,
// CHECK-NEXT:              args=[(TestRegType(), SideEffect.READ_WRITE)])
// CHECK-NEXT: def read_write_reg(rd_rs: TestReg):
// CHECK-NEXT:   test.ReadWriteRegOp(rd_rs)
// CHECK-EMPTY:

def Test4_ImmediateOp : TestOp<"immediate"> {
  let arguments = (ins ImmediateOfWidth<12>:$imm);
}
// CHECK-NEXT: @instruction(return_val_func=mock_return_val_func,
// CHECK-NEXT:              args=[(ImmediateType(12), SideEffect.READ)])
// CHECK-NEXT: def immediate(imm: Immediate):
// CHECK-NEXT:   test.ImmediateOp(imm)
// CHECK-EMPTY:

def Test5_LabelOp : TestOp<"label"> {
  let arguments = (ins LabelType:$label);
}
// CHECK-NEXT: @instruction(return_val_func=mock_return_val_func,
// CHECK-NEXT:              args=[(LabelType(), SideEffect.READ)])
// CHECK-NEXT: def label(label: Label):
// CHECK-NEXT:   test.LabelOp(label)
// CHECK-EMPTY:

def Test6_DestSourceOp : TestOp<"dest_source"> {
  let arguments = (ins Arg<TestRegType, "", [DestReg]>:$rd,
                       Arg<TestRegType, "", [SourceReg]>:$rs);
}
// CHECK-NEXT: @instruction(return_val_func=mock_return_val_func,
// CHECK-NEXT:              args=[(TestRegType(), SideEffect.WRITE), (TestRegType(), SideEffect.READ)])
// CHECK-NEXT: def dest_source(rd: TestReg, rs: TestReg):
// CHECK-NEXT:   test.DestSourceOp(rd, rs)
// CHECK-EMPTY:

def Test7_NoOperandsOp : TestOp<"no_operands"> {}
// CHECK-NEXT: @instruction(return_val_func=mock_return_val_func,
// CHECK-NEXT:              args=[])
// CHECK-NEXT: def no_operands():
// CHECK-NEXT:   test.NoOperandsOp()
// CHECK-EMPTY:

def Test8_NoDecoratorOp : TestOp<"no_decorator"> {
  let arguments = (ins TestRegType:$rd);
}
// CHECK-NEXT: @instruction(return_val_func=mock_return_val_func,
// CHECK-NEXT:              args=[(TestRegType(), SideEffect.READ_WRITE)])
// CHECK-NEXT: def no_decorator(rd: TestReg):
// CHECK-NEXT:   test.NoDecoratorOp(rd)
// CHECK-EMPTY:

def Test9_UnionWithRegisterOp : TestOp<"union_with_register"> {
  let arguments = (ins Arg<TestRegType, "", [DestReg]>:$rd,
                       AnyTypeOf<[TestRegType, ImmediateOfWidth<12>]>:$src);
}
// CHECK-NEXT: @instruction(return_val_func=mock_return_val_func,
// CHECK-NEXT:              args=[(TestRegType(), SideEffect.WRITE), (TestRegType(), SideEffect.READ_WRITE)])
// CHECK-NEXT: def union_with_register_testreg(rd: TestReg, src: TestReg):
// CHECK-NEXT:   test.UnionWithRegisterOp(rd, src)
// CHECK-EMPTY:
// CHECK-NEXT: @instruction(return_val_func=mock_return_val_func,
// CHECK-NEXT:              args=[(TestRegType(), SideEffect.WRITE), (ImmediateType(12), SideEffect.READ_WRITE)])
// CHECK-NEXT: def union_with_register_imm12(rd: TestReg, src: Immediate):
// CHECK-NEXT:   test.UnionWithRegisterOp(rd, src)
// CHECK-EMPTY:
// CHECK-NEXT: def union_with_register(rd: TestReg, src: Union[TestReg, Immediate]):
// CHECK-NEXT:   if isinstance(src, TestReg):
// CHECK-NEXT:     return union_with_register_testreg(rd, src)
// CHECK-NEXT:   else:
// CHECK-NEXT:     return union_with_register_imm12(rd, src)
// CHECK-EMPTY:

def Test10_MultiRegisterUnionOp : TestOp<"multi_register_union"> {
  let arguments = (ins AnyTypeOf<[TestRegType, TestFloatRegType]>:$reg);
}
// CHECK-NEXT: @instruction(return_val_func=mock_return_val_func,
// CHECK-NEXT:              args=[(TestFloatRegType(), SideEffect.READ_WRITE)])
// CHECK-NEXT: def multi_register_union_testfloatreg(reg: TestFloatReg):
// CHECK-NEXT:   test.MultiRegisterUnionOp(reg)
// CHECK-EMPTY:
// CHECK-NEXT: @instruction(return_val_func=mock_return_val_func,
// CHECK-NEXT:              args=[(TestRegType(), SideEffect.READ_WRITE)])
// CHECK-NEXT: def multi_register_union_testreg(reg: TestReg):
// CHECK-NEXT:   test.MultiRegisterUnionOp(reg)
// CHECK-EMPTY:
// CHECK-NEXT: def multi_register_union(reg: Union[TestFloatReg, TestReg]):
// CHECK-NEXT:   if isinstance(reg, TestFloatReg):
// CHECK-NEXT:     return multi_register_union_testfloatreg(reg)
// CHECK-NEXT:   else:
// CHECK-NEXT:     return multi_register_union_testreg(reg)
// CHECK-EMPTY:

def Test11_TwoUnionArgsOp : TestOp<"two_union_args"> {
  let arguments = (ins Arg<AnyTypeOf<[TestRegType, ImmediateOfWidth<12>]>, "", [DestReg]>:$dst,
                       AnyTypeOf<[TestRegType, TestFloatRegType]>:$src);
}
// CHECK-NEXT: @instruction(return_val_func=mock_return_val_func,
// CHECK-NEXT:              args=[(TestRegType(), SideEffect.WRITE), (TestFloatRegType(), SideEffect.READ_WRITE)])
// CHECK-NEXT: def two_union_args_testreg_testfloatreg(dst: TestReg, src: TestFloatReg):
// CHECK-NEXT:   test.TwoUnionArgsOp(dst, src)
// CHECK-EMPTY:
// CHECK-NEXT: @instruction(return_val_func=mock_return_val_func,
// CHECK-NEXT:              args=[(TestRegType(), SideEffect.WRITE), (TestRegType(), SideEffect.READ_WRITE)])
// CHECK-NEXT: def two_union_args_testreg_testreg(dst: TestReg, src: TestReg):
// CHECK-NEXT:   test.TwoUnionArgsOp(dst, src)
// CHECK-EMPTY:
// CHECK-NEXT: @instruction(return_val_func=mock_return_val_func,
// CHECK-NEXT:              args=[(ImmediateType(12), SideEffect.WRITE), (TestFloatRegType(), SideEffect.READ_WRITE)])
// CHECK-NEXT: def two_union_args_imm12_testfloatreg(dst: Immediate, src: TestFloatReg):
// CHECK-NEXT:   test.TwoUnionArgsOp(dst, src)
// CHECK-EMPTY:
// CHECK-NEXT: @instruction(return_val_func=mock_return_val_func,
// CHECK-NEXT:              args=[(ImmediateType(12), SideEffect.WRITE), (TestRegType(), SideEffect.READ_WRITE)])
// CHECK-NEXT: def two_union_args_imm12_testreg(dst: Immediate, src: TestReg):
// CHECK-NEXT:   test.TwoUnionArgsOp(dst, src)
// CHECK-EMPTY:
// CHECK-NEXT: def two_union_args(dst: Union[TestReg, Immediate], src: Union[TestFloatReg, TestReg]):
// CHECK-NEXT:   if isinstance(dst, TestReg) and isinstance(src, TestFloatReg):
// CHECK-NEXT:     return two_union_args_testreg_testfloatreg(dst, src)
// CHECK-NEXT:   elif isinstance(dst, TestReg) and isinstance(src, TestReg):
// CHECK-NEXT:     return two_union_args_testreg_testreg(dst, src)
// CHECK-NEXT:   elif isinstance(dst, Immediate) and isinstance(src, TestFloatReg):
// CHECK-NEXT:     return two_union_args_imm12_testfloatreg(dst, src)
// CHECK-NEXT:   else:
// CHECK-NEXT:     return two_union_args_imm12_testreg(dst, src)
// CHECK-EMPTY:

def Test12_AndKeywordOp : TestOp<"and"> {
  let arguments = (ins Arg<TestRegType, "", [SourceReg]>:$rs);
}
// CHECK-NEXT: @instruction(return_val_func=mock_return_val_func,
// CHECK-NEXT:              args=[(TestRegType(), SideEffect.READ)])
// CHECK-NEXT: def and_(rs: TestReg):
// CHECK-NEXT:   test.AndKeywordOp(rs)
// CHECK-EMPTY:

def Test13_LeadingDigitOp : TestOp<"1add"> {
  let arguments = (ins Arg<TestRegType, "", [SourceReg]>:$rs);
}
// CHECK-NEXT: @instruction(return_val_func=mock_return_val_func,
// CHECK-NEXT:              args=[(TestRegType(), SideEffect.READ)])
// CHECK-NEXT: def _1add(rs: TestReg):
// CHECK-NEXT:   test.LeadingDigitOp(rs)
// CHECK-EMPTY:

def Test14_SpecialCharsOp : TestOp<"load#word"> {
  let arguments = (ins Arg<TestRegType, "", [SourceReg]>:$rs);
}
// CHECK-NEXT: @instruction(return_val_func=mock_return_val_func,
// CHECK-NEXT:              args=[(TestRegType(), SideEffect.READ)])
// CHECK-NEXT: def load_word(rs: TestReg):
// CHECK-NEXT:   test.SpecialCharsOp(rs)
// CHECK-EMPTY:

def Test15_MetadataOp : TestOp<"metadata">, ISAInstructionFormat {
  let isaMnemonic = "some_mnemonic";
  let extension = "some_ext";
}
// CHECK-NEXT: @instruction(return_val_func=mock_return_val_func,
// CHECK-NEXT:              args=[],
// CHECK-NEXT:              mnemonic="some_mnemonic",
// CHECK-NEXT:              extension="some_ext")
// CHECK-NEXT: def metadata():
// CHECK-NEXT:   test.MetadataOp()
// CHECK-EMPTY:

#ifdef TEST_EXTENSION_FILTER
// RUN: circt-tblgen -gen-rtg-instruction-python-wrappers --return-val-func=test_func --extension=nonexistent -I %S/../../../include -I %S/../../../llvm/mlir/include %s -DTEST_EXTENSION_FILTER 2>&1 | FileCheck %s --check-prefix=NO-MATCH --allow-empty
// NO-MATCH-NOT: @instruction

// RUN: circt-tblgen -gen-rtg-instruction-python-wrappers --return-val-func=test_func --extension=extension_name -I %S/../../../include -I %S/../../../llvm/mlir/include %s -DTEST_EXTENSION_FILTER 2>&1 | FileCheck %s --check-prefix=EXTENSION-MATCH
// EXTENSION-MATCH: @instruction

def FilterTestOp : TestOp<"filter_test">, ISAInstructionFormat {
  let extension = "extension_name";
}
#endif

#ifdef TEST_XLEN_FILTER
// RUN: circt-tblgen -gen-rtg-instruction-python-wrappers --return-val-func=test_func --xlen=128 -I %S/../../../include -I %S/../../../llvm/mlir/include %s -DTEST_XLEN_FILTER 2>&1 | FileCheck %s --check-prefix=XLEN-NO-MATCH --allow-empty
// XLEN-NO-MATCH-NOT: xlen_filter_test_both
// XLEN-NO-MATCH-NOT: xlen_filter_test_32_only

// RUN: circt-tblgen -gen-rtg-instruction-python-wrappers --return-val-func=test_func --xlen=32 -I %S/../../../include -I %S/../../../llvm/mlir/include %s -DTEST_XLEN_FILTER 2>&1 | FileCheck %s --check-prefix=XLEN-MATCH
// XLEN-MATCH-DAG: xlen_filter_test_both
// XLEN-MATCH-DAG: xlen_filter_test_32_only

// RUN: circt-tblgen -gen-rtg-instruction-python-wrappers --return-val-func=test_func --xlen=32,64 -I %S/../../../include -I %S/../../../llvm/mlir/include %s -DTEST_XLEN_FILTER 2>&1 | FileCheck %s --check-prefix=XLEN-BOTH-MATCH
// XLEN-BOTH-MATCH: xlen_filter_test_both
// XLEN-BOTH-MATCH-NOT: xlen_filter_test_32_only

def XlenFilterTestBothOp : TestOp<"xlen_filter_test_both">, ISAInstructionFormat {
  let xlen = [32, 64];
}

def XlenFilterTest32OnlyOp : TestOp<"xlen_filter_test_32_only">, ISAInstructionFormat {
  let xlen = [32];
}
#endif
