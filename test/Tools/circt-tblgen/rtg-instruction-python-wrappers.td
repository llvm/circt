// RUN: circt-tblgen -gen-rtg-instruction-python-wrappers --return-val-func=mock_return_val_func --op-prefix=rtgtest -I %S/../../../include -I %S/../../../llvm/mlir/include %s | FileCheck %s
// RUN: circt-tblgen -gen-rtg-instruction-python-wrappers --return-val-func=mock_return_val_func --op-prefix=rtgtest --header-file=%S/rtg-instruction-python-wrappers-header.py -I %S/../../../include -I %S/../../../llvm/mlir/include %s | FileCheck %s --check-prefix=WITH-HEADER

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

// Test: Single SourceReg operand
def Test1_SourceRegOp : TestOp<"source_reg"> {
  let arguments = (ins Arg<TestRegType, "", [SourceReg]>:$rs);
}
// CHECK-LABEL: @instruction(return_val_func=mock_return_val_func,
// CHECK-NEXT:              args=[(TestRegType(), SideEffect.READ)])
// CHECK-NEXT: def source_reg(rs: TestReg):
// CHECK-NEXT:   rtgtest.SourceRegOp(rs)

// Test: Single DestReg operand
def Test2_DestRegOp : TestOp<"dest_reg"> {
  let arguments = (ins Arg<TestRegType, "", [DestReg]>:$rd);
}
// CHECK-LABEL: @instruction(return_val_func=mock_return_val_func,
// CHECK-NEXT:              args=[(TestRegType(), SideEffect.WRITE)])
// CHECK-NEXT: def dest_reg(rd: TestReg):
// CHECK-NEXT:   rtgtest.DestRegOp(rd)

// Test: Register with both SourceReg and DestReg (READ_WRITE)
def Test3_ReadWriteRegOp : TestOp<"read_write_reg"> {
  let arguments = (ins Arg<TestRegType, "", [SourceReg, DestReg]>:$rd_rs);
}
// CHECK-LABEL: @instruction(return_val_func=mock_return_val_func,
// CHECK-NEXT:              args=[(TestRegType(), SideEffect.READ_WRITE)])
// CHECK-NEXT: def read_write_reg(rd_rs: TestReg):
// CHECK-NEXT:   rtgtest.ReadWriteRegOp(rd_rs)

// Test: Single immediate operand
def Test4_ImmediateOp : TestOp<"immediate"> {
  let arguments = (ins ImmediateOfWidth<12>:$imm);
}
// CHECK-LABEL: @instruction(return_val_func=mock_return_val_func,
// CHECK-NEXT:              args=[(ImmediateType(12), SideEffect.READ)])
// CHECK-NEXT: def immediate(imm: Immediate):
// CHECK-NEXT:   rtgtest.ImmediateOp(imm)

// Test: Single label operand
def Test5_LabelOp : TestOp<"label"> {
  let arguments = (ins LabelType:$label);
}
// CHECK-LABEL: @instruction(return_val_func=mock_return_val_func,
// CHECK-NEXT:              args=[(LabelType(), SideEffect.READ)])
// CHECK-NEXT: def label(label: Label):
// CHECK-NEXT:   rtgtest.LabelOp(label)

// Test: DestReg + SourceReg
def Test6_DestSourceOp : TestOp<"dest_source"> {
  let arguments = (ins Arg<TestRegType, "", [DestReg]>:$rd,
                       Arg<TestRegType, "", [SourceReg]>:$rs);
}
// CHECK-LABEL: @instruction(return_val_func=mock_return_val_func,
// CHECK-NEXT:              args=[(TestRegType(), SideEffect.WRITE), (TestRegType(), SideEffect.READ)])
// CHECK-NEXT: def dest_source(rd: TestReg, rs: TestReg):
// CHECK-NEXT:   rtgtest.DestSourceOp(rd, rs)

// Test: DestReg + SourceReg + Immediate
def Test7_ITypeOp : TestOp<"i_type"> {
  let arguments = (ins Arg<TestRegType, "", [DestReg]>:$rd,
                       Arg<TestRegType, "", [SourceReg]>:$rs,
                       ImmediateOfWidth<12>:$imm);
}
// CHECK-LABEL: @instruction(return_val_func=mock_return_val_func,
// CHECK-NEXT:              args=[(TestRegType(), SideEffect.WRITE), (TestRegType(), SideEffect.READ), (ImmediateType(12), SideEffect.READ)])
// CHECK-NEXT: def i_type(rd: TestReg, rs: TestReg, imm: Immediate):
// CHECK-NEXT:   rtgtest.ITypeOp(rd, rs, imm)

// Test: DestReg + two SourceRegs
def Test8_RTypeOp : TestOp<"r_type"> {
  let arguments = (ins Arg<TestRegType, "", [DestReg]>:$rd,
                       Arg<TestRegType, "", [SourceReg]>:$rs1,
                       Arg<TestRegType, "", [SourceReg]>:$rs2);
}
// CHECK-LABEL: @instruction(return_val_func=mock_return_val_func,
// CHECK-NEXT:              args=[(TestRegType(), SideEffect.WRITE), (TestRegType(), SideEffect.READ), (TestRegType(), SideEffect.READ)])
// CHECK-NEXT: def r_type(rd: TestReg, rs1: TestReg, rs2: TestReg):
// CHECK-NEXT:   rtgtest.RTypeOp(rd, rs1, rs2)

// Test: Two SourceRegs + Immediate
def Test9_STypeOp : TestOp<"s_type"> {
  let arguments = (ins Arg<TestRegType, "", [SourceReg]>:$rs1,
                       Arg<TestRegType, "", [SourceReg]>:$rs2,
                       ImmediateOfWidth<12>:$imm);
}
// CHECK-LABEL: @instruction(return_val_func=mock_return_val_func,
// CHECK-NEXT:              args=[(TestRegType(), SideEffect.READ), (TestRegType(), SideEffect.READ), (ImmediateType(12), SideEffect.READ)])
// CHECK-NEXT: def s_type(rs1: TestReg, rs2: TestReg, imm: Immediate):
// CHECK-NEXT:   rtgtest.STypeOp(rs1, rs2, imm)

// Test: Two SourceRegs + Label
def Test10_BTypeOp : TestOp<"b_type"> {
  let arguments = (ins Arg<TestRegType, "", [SourceReg]>:$rs1,
                       Arg<TestRegType, "", [SourceReg]>:$rs2,
                       LabelType:$label);
}
// CHECK-LABEL: @instruction(return_val_func=mock_return_val_func,
// CHECK-NEXT:              args=[(TestRegType(), SideEffect.READ), (TestRegType(), SideEffect.READ), (LabelType(), SideEffect.READ)])
// CHECK-NEXT: def b_type(rs1: TestReg, rs2: TestReg, label: Label):
// CHECK-NEXT:   rtgtest.BTypeOp(rs1, rs2, label)

// Test: DestReg + Immediate
def Test11_UTypeOp : TestOp<"u_type"> {
  let arguments = (ins Arg<TestRegType, "", [DestReg]>:$rd,
                       ImmediateOfWidth<20>:$imm);
}
// CHECK-LABEL: @instruction(return_val_func=mock_return_val_func,
// CHECK-NEXT:              args=[(TestRegType(), SideEffect.WRITE), (ImmediateType(20), SideEffect.READ)])
// CHECK-NEXT: def u_type(rd: TestReg, imm: Immediate):
// CHECK-NEXT:   rtgtest.UTypeOp(rd, imm)

// Test: DestReg + Label
def Test12_JTypeOp : TestOp<"j_type"> {
  let arguments = (ins Arg<TestRegType, "", [DestReg]>:$rd,
                       LabelType:$label);
}
// CHECK-LABEL: @instruction(return_val_func=mock_return_val_func,
// CHECK-NEXT:              args=[(TestRegType(), SideEffect.WRITE), (LabelType(), SideEffect.READ)])
// CHECK-NEXT: def j_type(rd: TestReg, label: Label):
// CHECK-NEXT:   rtgtest.JTypeOp(rd, label)

// Test: READ_WRITE register + SourceReg
def Test13_ReadWriteSourceOp : TestOp<"read_write_source"> {
  let arguments = (ins Arg<TestRegType, "", [SourceReg, DestReg]>:$rd_rs1,
                       Arg<TestRegType, "", [SourceReg]>:$rs2);
}
// CHECK-LABEL: @instruction(return_val_func=mock_return_val_func,
// CHECK-NEXT:              args=[(TestRegType(), SideEffect.READ_WRITE), (TestRegType(), SideEffect.READ)])
// CHECK-NEXT: def read_write_source(rd_rs1: TestReg, rs2: TestReg):
// CHECK-NEXT:   rtgtest.ReadWriteSourceOp(rd_rs1, rs2)

// Test: Multiple DestRegs (multiple outputs)
def Test14_MultiDestOp : TestOp<"multi_dest"> {
  let arguments = (ins Arg<TestRegType, "", [DestReg]>:$rd1,
                       Arg<TestRegType, "", [DestReg]>:$rd2,
                       Arg<TestRegType, "", [SourceReg]>:$rs);
}
// CHECK-LABEL: @instruction(return_val_func=mock_return_val_func,
// CHECK-NEXT:              args=[(TestRegType(), SideEffect.WRITE), (TestRegType(), SideEffect.WRITE), (TestRegType(), SideEffect.READ)])
// CHECK-NEXT: def multi_dest(rd1: TestReg, rd2: TestReg, rs: TestReg):
// CHECK-NEXT:   rtgtest.MultiDestOp(rd1, rd2, rs)

// Test: No operands
def Test15_NoOperandsOp : TestOp<"no_operands"> {
}
// CHECK-LABEL: @instruction(return_val_func=mock_return_val_func,
// CHECK-NEXT:              args=[])
// CHECK-NEXT: def no_operands():
// CHECK-NEXT:   rtgtest.NoOperandsOp()

// Test: Different immediate widths
def Test16_Imm5Op : TestOp<"imm5"> {
  let arguments = (ins Arg<TestRegType, "", [DestReg]>:$rd,
                       Arg<TestRegType, "", [SourceReg]>:$rs,
                       ImmediateOfWidth<5>:$imm);
}
// CHECK-LABEL: @instruction(return_val_func=mock_return_val_func,
// CHECK-NEXT:              args=[(TestRegType(), SideEffect.WRITE), (TestRegType(), SideEffect.READ), (ImmediateType(5), SideEffect.READ)])
// CHECK-NEXT: def imm5(rd: TestReg, rs: TestReg, imm: Immediate):
// CHECK-NEXT:   rtgtest.Imm5Op(rd, rs, imm)

// Test: Register without decorator (should default to READ_WRITE)
def Test17_NoDecoratorOp : TestOp<"no_decorator"> {
  let arguments = (ins TestRegType:$rd);
}
// CHECK-LABEL: @instruction(return_val_func=mock_return_val_func,
// CHECK-NEXT:              args=[(TestRegType(), SideEffect.READ_WRITE)])
// CHECK-NEXT: def no_decorator(rd: TestReg):
// CHECK-NEXT:   rtgtest.NoDecoratorOp(rd)

// Test: Union type with RegisterTypeInterface
def Test18_UnionWithRegisterOp : TestOp<"union_with_register"> {
  let arguments = (ins Arg<TestRegType, "", [DestReg]>:$rd,
                       AnyTypeOf<[TestRegType, ImmediateOfWidth<12>]>:$src);
}
// CHECK: @instruction(return_val_func=mock_return_val_func,
// CHECK-NEXT:              args=[(TestRegType(), SideEffect.WRITE), (TestRegType(), SideEffect.READ_WRITE)])
// CHECK-NEXT: def union_with_register_reg(rd: TestReg, src: TestReg):
// CHECK-NEXT:   rtgtest.UnionWithRegisterOp(rd, src)
// CHECK-EMPTY:
// CHECK-NEXT: @instruction(return_val_func=mock_return_val_func,
// CHECK-NEXT:              args=[(TestRegType(), SideEffect.WRITE), (ImmediateType(12), SideEffect.READ_WRITE)])
// CHECK-NEXT: def union_with_register_imm(rd: TestReg, src: Immediate):
// CHECK-NEXT:   rtgtest.UnionWithRegisterOp(rd, src)
// CHECK-EMPTY:
// CHECK-NEXT: def union_with_register(rd: TestReg, src: Union[TestReg, Immediate]):
// CHECK-NEXT:   if isinstance(src, TestReg):
// CHECK-NEXT:     return union_with_register_reg(rd, src)
// CHECK-NEXT:   else:
// CHECK-NEXT:     return union_with_register_imm(rd, src)

// Test: Union type with multiple register types - verifies interface detection
def Test19_MultiRegisterUnionOp : TestOp<"multi_register_union"> {
  let arguments = (ins AnyTypeOf<[TestRegType, TestFloatRegType]>:$reg);
}
// CHECK: @instruction(return_val_func=mock_return_val_func,
// CHECK-NEXT:              args=[(TestFloatRegType(), SideEffect.READ_WRITE)])
// CHECK-NEXT: def multi_register_union_reg(reg: TestFloatReg):
// CHECK-NEXT:   rtgtest.MultiRegisterUnionOp(reg)
// CHECK-EMPTY:
// CHECK-NEXT: @instruction(return_val_func=mock_return_val_func,
// CHECK-NEXT:              args=[(TestRegType(), SideEffect.READ_WRITE)])
// CHECK-NEXT: def multi_register_union_reg(reg: TestReg):
// CHECK-NEXT:   rtgtest.MultiRegisterUnionOp(reg)
// CHECK-EMPTY:
// CHECK-NEXT: def multi_register_union(reg: Union[TestFloatReg, TestReg]):
// CHECK-NEXT:   if isinstance(reg, TestFloatReg):
// CHECK-NEXT:     return multi_register_union_reg(reg)
// CHECK-NEXT:   else:
// CHECK-NEXT:     return multi_register_union_reg(reg)

// Test: with header file
// WITH-HEADER: from rtg import instruction, SideEffect, ImmediateType, LabelType
// WITH-HEADER: from my_isa import TestRegType
// WITH-HEADER-EMPTY:
// WITH-HEADER: def mock_return_val_func():
// WITH-HEADER:     pass
// WITH-HEADER: @instruction(return_val_func=mock_return_val_func,
