// RUN: circt-tblgen -gen-rtg-instruction-list -I %S/../../../include -I %S/../../../llvm/mlir/include %s | FileCheck %s

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

def Test1Op : TestOp<"test1"> {
  let isaAssemblyFormat = "`test1`";
}

def Test2Op : TestOp<"test2"> {
  let isaAssemblyFormat = "`test2`";
}

def NonInstructionOp : Op<TestDialect, "non_instruction"> {}

// CHECK-NOT: test.non_instruction
// CHECK: test.test1,
// CHECK-NEXT: test.test2,
