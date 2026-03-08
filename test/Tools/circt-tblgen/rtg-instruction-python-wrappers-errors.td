// RUN: not circt-tblgen -gen-rtg-instruction-python-wrappers -I %S/../../../include -I %S/../../../llvm/mlir/include %s 2>&1 | FileCheck %s --check-prefix=UNSUPPORTED-TYPE

include "mlir/IR/OpBase.td"
include "circt/Dialect/RTG/IR/RTGISAAssemblyInterfaces.td"

def TestDialect : Dialect {
  let name = "test";
  let cppNamespace = "::test";
}

class TestOp<string mnemonic> :
    Op<TestDialect, mnemonic, [InstructionOpInterface]>;

def UnsupportedI32Op : TestOp<"unsupported_i32"> {
  let arguments = (ins I32:$value);
}
// UNSUPPORTED-TYPE: error: failed to classify operand type for 'value'
