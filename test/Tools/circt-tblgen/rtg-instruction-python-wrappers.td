// RUN: circt-tblgen -gen-rtg-instruction-python-wrappers -I %S/../../../include -I %S/../../../llvm/mlir/include %s | FileCheck %s

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

def Test1_SourceRegOp : TestOp<"source_reg"> {
  let arguments = (ins Arg<TestRegType, "", [SourceReg]>:$rs);
}
// CHECK: @instruction(args=[(TestRegType(), SideEffect.READ)])
// CHECK-NEXT: def source_reg(rs: TestReg):
// CHECK-NEXT:   SourceRegOp(rs)
// CHECK-EMPTY:

def Test2_DestRegOp : TestOp<"dest_reg"> {
  let arguments = (ins Arg<TestRegType, "", [DestReg]>:$rd);
}
// CHECK-NEXT: @instruction(args=[(TestRegType(), SideEffect.WRITE)])
// CHECK-NEXT: def dest_reg(rd: TestReg):
// CHECK-NEXT:   DestRegOp(rd)
// CHECK-EMPTY:

def Test3_ReadWriteRegOp : TestOp<"read_write_reg"> {
  let arguments = (ins Arg<TestRegType, "", [SourceReg, DestReg]>:$rd_rs);
}
// CHECK-NEXT: @instruction(args=[(TestRegType(), SideEffect.READ_WRITE)])
// CHECK-NEXT: def read_write_reg(rd_rs: TestReg):
// CHECK-NEXT:   ReadWriteRegOp(rd_rs)
// CHECK-EMPTY:

def Test4_ImmediateOp : TestOp<"immediate"> {
  let arguments = (ins ImmediateOfWidth<12>:$imm);
}
// CHECK-NEXT: @instruction(args=[(ImmediateType(12), SideEffect.READ)])
// CHECK-NEXT: def immediate(imm: Immediate):
// CHECK-NEXT:   ImmediateOp(imm)
// CHECK-EMPTY:

def Test5_LabelOp : TestOp<"label"> {
  let arguments = (ins LabelType:$label);
}
// CHECK-NEXT: @instruction(args=[(LabelType(), SideEffect.READ)])
// CHECK-NEXT: def label(label: Label):
// CHECK-NEXT:   LabelOp(label)
// CHECK-EMPTY:

def Test6_DestSourceOp : TestOp<"dest_source"> {
  let arguments = (ins Arg<TestRegType, "", [DestReg]>:$rd,
                       Arg<TestRegType, "", [SourceReg]>:$rs);
}
// CHECK-NEXT: @instruction(args=[(TestRegType(), SideEffect.WRITE), (TestRegType(), SideEffect.READ)])
// CHECK-NEXT: def dest_source(rd: TestReg, rs: TestReg):
// CHECK-NEXT:   DestSourceOp(rd, rs)
// CHECK-EMPTY:

def Test7_NoOperandsOp : TestOp<"no_operands"> {}
// CHECK-NEXT: @instruction(args=[])
// CHECK-NEXT: def no_operands():
// CHECK-NEXT:   NoOperandsOp()
// CHECK-EMPTY:

def Test8_NoDecoratorOp : TestOp<"no_decorator"> {
  let arguments = (ins TestRegType:$rd);
}
// CHECK-NEXT: @instruction(args=[(TestRegType(), SideEffect.READ_WRITE)])
// CHECK-NEXT: def no_decorator(rd: TestReg):
// CHECK-NEXT:   NoDecoratorOp(rd)
// CHECK-EMPTY:

def Test9_UnionWithRegisterOp : TestOp<"union_with_register"> {
  let arguments = (ins Arg<TestRegType, "", [DestReg]>:$rd,
                       AnyTypeOf<[TestRegType, ImmediateOfWidth<12>]>:$src);
}
// CHECK-NEXT: @instruction(args=[(TestRegType(), SideEffect.WRITE), (TestRegType(), SideEffect.READ_WRITE)])
// CHECK-NEXT: def union_with_register_testreg(rd: TestReg, src: TestReg):
// CHECK-NEXT:   UnionWithRegisterOp(rd, src)
// CHECK-EMPTY:
// CHECK-NEXT: @instruction(args=[(TestRegType(), SideEffect.WRITE), (ImmediateType(12), SideEffect.READ_WRITE)])
// CHECK-NEXT: def union_with_register_imm12(rd: TestReg, src: Immediate):
// CHECK-NEXT:   UnionWithRegisterOp(rd, src)
// CHECK-EMPTY:

def Test10_MultiRegisterUnionOp : TestOp<"multi_register_union"> {
  let arguments = (ins AnyTypeOf<[TestRegType, TestFloatRegType]>:$reg);
}
// CHECK-NEXT: @instruction(args=[(TestFloatRegType(), SideEffect.READ_WRITE)])
// CHECK-NEXT: def multi_register_union_testfloatreg(reg: TestFloatReg):
// CHECK-NEXT:   MultiRegisterUnionOp(reg)
// CHECK-EMPTY:
// CHECK-NEXT: @instruction(args=[(TestRegType(), SideEffect.READ_WRITE)])
// CHECK-NEXT: def multi_register_union_testreg(reg: TestReg):
// CHECK-NEXT:   MultiRegisterUnionOp(reg)
// CHECK-EMPTY:

def Test11_TwoUnionArgsOp : TestOp<"two_union_args"> {
  let arguments = (ins Arg<AnyTypeOf<[TestRegType, ImmediateOfWidth<12>]>, "", [DestReg]>:$dst,
                       AnyTypeOf<[TestRegType, TestFloatRegType]>:$src);
}
// CHECK-NEXT: @instruction(args=[(TestRegType(), SideEffect.WRITE), (TestFloatRegType(), SideEffect.READ_WRITE)])
// CHECK-NEXT: def two_union_args_testreg_testfloatreg(dst: TestReg, src: TestFloatReg):
// CHECK-NEXT:   TwoUnionArgsOp(dst, src)
// CHECK-EMPTY:
// CHECK-NEXT: @instruction(args=[(TestRegType(), SideEffect.WRITE), (TestRegType(), SideEffect.READ_WRITE)])
// CHECK-NEXT: def two_union_args_testreg_testreg(dst: TestReg, src: TestReg):
// CHECK-NEXT:   TwoUnionArgsOp(dst, src)
// CHECK-EMPTY:
// CHECK-NEXT: @instruction(args=[(ImmediateType(12), SideEffect.WRITE), (TestFloatRegType(), SideEffect.READ_WRITE)])
// CHECK-NEXT: def two_union_args_imm12_testfloatreg(dst: Immediate, src: TestFloatReg):
// CHECK-NEXT:   TwoUnionArgsOp(dst, src)
// CHECK-EMPTY:
// CHECK-NEXT: @instruction(args=[(ImmediateType(12), SideEffect.WRITE), (TestRegType(), SideEffect.READ_WRITE)])
// CHECK-NEXT: def two_union_args_imm12_testreg(dst: Immediate, src: TestReg):
// CHECK-NEXT:   TwoUnionArgsOp(dst, src)
// CHECK-EMPTY:
