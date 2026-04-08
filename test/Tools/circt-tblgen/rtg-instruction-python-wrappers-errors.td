include "mlir/IR/OpBase.td"
include "circt/Dialect/RTG/IR/RTGISAAssemblyInterfaces.td"

def TestDialect : Dialect {
  let name = "test";
  let cppNamespace = "::test";
}

class TestOp<string mnemonic> :
    Op<TestDialect, mnemonic, [InstructionOpInterface]>;

#ifdef TEST_UNSUPPORTED_TYPE
// RUN: not circt-tblgen -gen-rtg-instruction-python-wrappers --return-val-func=test_func -I %circt_src_include_dir -I %mlir_src_include_dir %s -DTEST_UNSUPPORTED_TYPE 2>&1 | FileCheck %s --check-prefix=UNSUPPORTED-TYPE
def UnsupportedI32Op : TestOp<"unsupported_i32"> {
  let arguments = (ins I32:$value);
}
// UNSUPPORTED-TYPE: error: failed to classify operand type for 'value'
#endif

#ifdef TEST_EMPTY_NAME
// RUN: not circt-tblgen -gen-rtg-instruction-python-wrappers --return-val-func=test_func -I %circt_src_include_dir -I %mlir_src_include_dir %s -DTEST_EMPTY_NAME 2>&1 | FileCheck %s --check-prefix=EMPTY-NAME
def EmptyNameOp : TestOp<""> {
}
// EMPTY-NAME: error: Empty string is not a valid Python identifier
#endif

#ifdef TEST_ALL_UNDERSCORES
// RUN: not circt-tblgen -gen-rtg-instruction-python-wrappers --return-val-func=test_func -I %circt_src_include_dir -I %mlir_src_include_dir %s -DTEST_ALL_UNDERSCORES 2>&1 | FileCheck %s --check-prefix=ALL-UNDERSCORES
def AllUnderscoresOp : TestOp<"___"> {
}
// ALL-UNDERSCORES: error: String contains no valid characters for Python identifier
#endif

// Test: Missing --return-val-func CLI argument
// RUN: not circt-tblgen -gen-rtg-instruction-python-wrappers -I %circt_src_include_dir -I %mlir_src_include_dir %s 2>&1 | FileCheck %s --check-prefix=NO-RETURN-VAL-FUNC
// NO-RETURN-VAL-FUNC: error: --return-val-func is required for -gen-rtg-instruction-python-wrappers
