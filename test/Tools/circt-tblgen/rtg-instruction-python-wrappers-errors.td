include "mlir/IR/OpBase.td"
include "circt/Dialect/RTG/IR/RTGISAAssemblyInterfaces.td"

def TestDialect : Dialect {
  let name = "test";
  let cppNamespace = "::test";
}

class TestOp<string mnemonic> :
    Op<TestDialect, mnemonic, [InstructionOpInterface]>;

#ifdef TEST_EMPTY_NAME
// RUN: not circt-tblgen -gen-rtg-instruction-python-wrappers --return-val-func=test_func %TABLEGEN_INC% %s -DTEST_EMPTY_NAME 2>&1 | FileCheck %s --check-prefix=EMPTY-NAME
def EmptyNameOp : TestOp<"">, ISAInstructionFormat {
  let isaMnemonic = "";
}
// EMPTY-NAME: error: isaMnemonic field must be set for instruction operations
#endif

#ifdef TEST_ALL_UNDERSCORES
// RUN: not circt-tblgen -gen-rtg-instruction-python-wrappers --return-val-func=test_func %TABLEGEN_INC% %s -DTEST_ALL_UNDERSCORES 2>&1 | FileCheck %s --check-prefix=ALL-UNDERSCORES
def AllUnderscoresOp : TestOp<"___">, ISAInstructionFormat {
  let isaMnemonic = "___";
}
// ALL-UNDERSCORES: error: String contains no valid characters for Python identifier
#endif

#ifdef TEST_MISSING_ISA_MNEMONIC
// RUN: not circt-tblgen -gen-rtg-instruction-python-wrappers --return-val-func=test_func %TABLEGEN_INC% %s -DTEST_MISSING_ISA_MNEMONIC 2>&1 | FileCheck %s --check-prefix=MISSING-ISA-MNEMONIC
def MissingIsaMnemonicOp : TestOp<"missing_isa_mnemonic"> {
}
// MISSING-ISA-MNEMONIC: error: isaMnemonic field must be set for instruction operations
#endif

// Test: Missing --return-val-func CLI argument
// RUN: not circt-tblgen -gen-rtg-instruction-python-wrappers %TABLEGEN_INC% %s 2>&1 | FileCheck %s --check-prefix=NO-RETURN-VAL-FUNC
// NO-RETURN-VAL-FUNC: error: --return-val-func is required for -gen-rtg-instruction-python-wrappers
