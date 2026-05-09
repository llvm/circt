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

class TestOp<string mnemonic> : Op<TestDialect, mnemonic, [
  DeclareOpInterfaceMethods<RegisterAllocationOpInterface>
]>;

def TestRegType : TypeDef<RTGDialect, "TestReg">, RegisterTypeBase {
  let mnemonic = "test_reg";
  let binaryEncodingWidth = 5;
  let pythonName = "TestReg";
  let pythonTypeName = "TestRegType";
}

def Test1_NoRegistersOp : TestOp<"no_registers"> {
  let arguments = (ins ImmediateOfWidth<8>:$imm);
}
// CHECK-LABEL: bool NoRegistersOp::isSourceRegister(unsigned index) {
// CHECK-NEXT:    return false;
// CHECK-NEXT:  }
// CHECK-EMPTY:
// CHECK-NEXT:  bool NoRegistersOp::isDestinationRegister(unsigned index) {
// CHECK-NEXT:    return false;
// CHECK-NEXT:  }

def Test2_MixedRegsOp : TestOp<"mixed_regs"> {
  let arguments = (ins Arg<TestRegType, "", [DestReg]>:$rd,
                       Arg<TestRegType, "", [SourceReg]>:$rs1,
                       Arg<TestRegType, "", [SourceReg]>:$rs2);
}
// CHECK-LABEL: bool MixedRegsOp::isSourceRegister(unsigned index) {
// CHECK-NEXT:    return index == 1 || index == 2;
// CHECK-NEXT:  }
// CHECK-EMPTY:
// CHECK-NEXT:  bool MixedRegsOp::isDestinationRegister(unsigned index) {
// CHECK-NEXT:    return index == 0;
// CHECK-NEXT:  }
