// RUN: not circt-tblgen -gen-rtg-instruction-python-wrappers -I %S/../../../include -I %S/../../../llvm/mlir/include %s 2>&1 | FileCheck %s --check-prefix=NO-RETURN-VAL-FUNC
// RUN: not circt-tblgen -gen-rtg-instruction-python-wrappers --return-val-func=test_func -I %S/../../../include -I %S/../../../llvm/mlir/include %s 2>&1 | FileCheck %s --check-prefix=NO-OP-PREFIX
// RUN: not circt-tblgen -gen-rtg-instruction-python-wrappers --return-val-func=test_func --op-prefix=test -I %S/../../../include -I %S/../../../llvm/mlir/include %s 2>&1 | FileCheck %s --check-prefix=UNSUPPORTED-TYPE

include "mlir/IR/OpBase.td"
include "circt/Dialect/RTG/IR/RTGISAAssemblyInterfaces.td"

def TestDialect : Dialect {
  let name = "test";
  let cppNamespace = "::test";
}

class TestOp<string mnemonic> : Op<TestDialect, mnemonic, [InstructionOpInterface]>;

// Test: Missing --return-val-func CLI argument
// NO-RETURN-VAL-FUNC: error: --return-val-func is required for -gen-rtg-instruction-python-wrappers

// Test: Missing --op-prefix CLI argument
// NO-OP-PREFIX: error: --op-prefix is required for -gen-rtg-instruction-python-wrappers

// Test: Unsupported operand type
def UnsupportedI32Op : TestOp<"unsupported_i32"> {
  let arguments = (ins I32:$value);
}
// UNSUPPORTED-TYPE: error: Unsupported operand type for instruction operand type
