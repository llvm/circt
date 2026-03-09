include "circt/Dialect/RTG/IR/RTGISAAssemblyInterfaces.td"
include "circt/Dialect/RTG/IR/RTGTypes.td"
include "mlir/IR/OpBase.td"

def TestDialect : Dialect {
  let name = "test";
  let cppNamespace = "::test";
}

class TestOp<string mnemonic> : Op<TestDialect, mnemonic, [InstructionOpInterface]>, ISAInstructionFormat;

def TestRegType : Type<CPred<"true">, "test register">, RegisterTypeBase {
  let binaryEncodingWidth = 5;
  let pythonName = "TestReg";
  let pythonTypeName = "TestRegType";
}

// RUN: not circt-tblgen -gen-rtg-instruction-methods -I %S/../../../include -I %S/../../../llvm/mlir/include %s -DTEST_IMM_NO_WRAPPER 2>&1 | FileCheck %s --check-prefix=IMM_NO_WRAPPER
#ifdef TEST_IMM_NO_WRAPPER
def ErrorImmNoWrapperOp : TestOp<"error_imm_no_wrapper"> {
  let arguments = (ins ImmediateOfWidth<12>:$imm);
  let isaAssemblyFormat = "mnemonic ` ` $imm";
}
// IMM_NO_WRAPPER: [[@LINE-2]]:41: error: immediate operand '$imm' must be wrapped in signed() or unsigned()
#endif

// RUN: not circt-tblgen -gen-rtg-instruction-methods -I %S/../../../include -I %S/../../../llvm/mlir/include %s -DTEST_LABEL_SIGNED 2>&1 | FileCheck %s --check-prefix=LABEL_SIGNED
#ifdef TEST_LABEL_SIGNED
def ErrorLabelSignedOp : TestOp<"error_label_signed"> {
  let arguments = (ins LabelType:$label);
  let isaAssemblyFormat = "mnemonic ` ` signed($label)";
}
// LABEL_SIGNED: [[@LINE-2]]:48: error: label operand '$label' cannot be wrapped in signed() or unsigned()
#endif

// RUN: not circt-tblgen -gen-rtg-instruction-methods -I %S/../../../include -I %S/../../../llvm/mlir/include %s -DTEST_LABEL_UNSIGNED 2>&1 | FileCheck %s --check-prefix=LABEL_UNSIGNED
#ifdef TEST_LABEL_UNSIGNED
def ErrorLabelUnsignedOp : TestOp<"error_label_unsigned"> {
  let arguments = (ins LabelType:$label);
  let isaAssemblyFormat = "mnemonic ` ` unsigned($label)";
}
// LABEL_UNSIGNED: [[@LINE-2]]:50: error: label operand '$label' cannot be wrapped in signed() or unsigned()
#endif

// RUN: not circt-tblgen -gen-rtg-instruction-methods -I %S/../../../include -I %S/../../../llvm/mlir/include %s -DTEST_REG_SIGNED 2>&1 | FileCheck %s --check-prefix=REG_SIGNED
#ifdef TEST_REG_SIGNED
def ErrorRegSignedOp : TestOp<"error_reg_signed"> {
  let arguments = (ins TestRegType:$reg);
  let isaAssemblyFormat = "mnemonic ` ` signed($reg)";
}
// REG_SIGNED: [[@LINE-2]]:48: error: register operand '$reg' cannot be wrapped in signed() or unsigned()
#endif

// RUN: not circt-tblgen -gen-rtg-instruction-methods -I %S/../../../include -I %S/../../../llvm/mlir/include %s -DTEST_REG_UNSIGNED 2>&1 | FileCheck %s --check-prefix=REG_UNSIGNED
#ifdef TEST_REG_UNSIGNED
def ErrorRegUnsignedOp : TestOp<"error_reg_unsigned"> {
  let arguments = (ins TestRegType:$reg);
  let isaAssemblyFormat = "mnemonic ` ` unsigned($reg)";
}
// REG_UNSIGNED: [[@LINE-2]]:50: error: register operand '$reg' cannot be wrapped in signed() or unsigned()
#endif

// RUN: not circt-tblgen -gen-rtg-instruction-methods -I %S/../../../include -I %S/../../../llvm/mlir/include %s -DTEST_UNKNOWN_OPERAND_ASM 2>&1 | FileCheck %s --check-prefix=UNKNOWN_OPERAND_ASM
#ifdef TEST_UNKNOWN_OPERAND_ASM
def ErrorUnknownOperandAsmOp : TestOp<"error_unknown_operand_asm"> {
  let arguments = (ins TestRegType:$reg);
  let isaAssemblyFormat = "mnemonic ` ` $unknown";
}
// UNKNOWN_OPERAND_ASM: [[@LINE-2]]:41: error: unknown operand '$unknown'
#endif

// RUN: not circt-tblgen -gen-rtg-instruction-methods -I %S/../../../include -I %S/../../../llvm/mlir/include %s -DTEST_UNKNOWN_OPERAND_BIN 2>&1 | FileCheck %s --check-prefix=UNKNOWN_OPERAND_BIN
#ifdef TEST_UNKNOWN_OPERAND_BIN
def ErrorUnknownOperandBinOp : TestOp<"error_unknown_operand_bin"> {
  let arguments = (ins TestRegType:$reg);
  let isaBinaryFormat = "$unknown 0b1111";
  let isaAssemblyFormat = "mnemonic ` ` $reg";
}
// UNKNOWN_OPERAND_BIN: [[@LINE-3]]:26: error: unknown operand '$unknown'
#endif

// RUN: not circt-tblgen -gen-rtg-instruction-methods -I %S/../../../include -I %S/../../../llvm/mlir/include %s -DTEST_INVALID_BIT_SLICE 2>&1 | FileCheck %s --check-prefix=INVALID_BIT_SLICE
#ifdef TEST_INVALID_BIT_SLICE
def ErrorInvalidBitSliceOp : TestOp<"error_invalid_bit_slice"> {
  let arguments = (ins ImmediateOfWidth<12>:$imm);
  let isaBinaryFormat = "$imm[5:10] 0b1111";
  let isaAssemblyFormat = "mnemonic ` ` signed($imm)";
}
// INVALID_BIT_SLICE: [[@LINE-3]]:30: error: high bit (5) must be >= low bit (10)
#endif

// RUN: not circt-tblgen -gen-rtg-instruction-methods -I %S/../../../include -I %S/../../../llvm/mlir/include %s -DTEST_INVALID_BINARY_LITERAL 2>&1 | FileCheck %s --check-prefix=INVALID_BINARY_LITERAL
#ifdef TEST_INVALID_BINARY_LITERAL
def ErrorInvalidBinaryLiteralOp : TestOp<"error_invalid_binary_literal"> {
  let arguments = (ins TestRegType:$reg);
  let isaBinaryFormat = "$reg 0b10102";
  let isaAssemblyFormat = "mnemonic ` ` $reg";
}
// INVALID_BINARY_LITERAL: [[@LINE-3]]:37: error: invalid character ('2') after binary literal
#endif

// RUN: not circt-tblgen -gen-rtg-instruction-methods -I %S/../../../include -I %S/../../../llvm/mlir/include %s -DTEST_MALFORMED_BIT_SLICE 2>&1 | FileCheck %s --check-prefix=MALFORMED_BIT_SLICE
#ifdef TEST_MALFORMED_BIT_SLICE
def ErrorMalformedBitSliceOp : TestOp<"error_malformed_bit_slice"> {
  let arguments = (ins ImmediateOfWidth<12>:$imm);
  let isaBinaryFormat = "$imm[:5] 0b1111";
  let isaAssemblyFormat = "mnemonic ` ` signed($imm)";
}
// MALFORMED_BIT_SLICE: [[@LINE-3]]:31: error: expected unsigned integer literal
#endif

// RUN: not circt-tblgen -gen-rtg-instruction-methods -I %S/../../../include -I %S/../../../llvm/mlir/include %s -DTEST_UNTERMINATED_BACKTICK 2>&1 | FileCheck %s --check-prefix=UNTERMINATED_BACKTICK
#ifdef TEST_UNTERMINATED_BACKTICK
def ErrorUnterminatedBacktickOp : TestOp<"error_unterminated_backtick"> {
  let arguments = (ins TestRegType:$reg);
  let isaAssemblyFormat = "mnemonic ` unterminated";
}
// UNTERMINATED_BACKTICK: [[@LINE-2]]:37: error: unterminated backtick literal
#endif

// RUN: not circt-tblgen -gen-rtg-instruction-methods -I %S/../../../include -I %S/../../../llvm/mlir/include %s -DTEST_MISSING_CLOSING_PAREN 2>&1 | FileCheck %s --check-prefix=MISSING_CLOSING_PAREN
#ifdef TEST_MISSING_CLOSING_PAREN
def ErrorMissingClosingParenOp : TestOp<"error_missing_closing_paren"> {
  let arguments = (ins ImmediateOfWidth<12>:$imm);
  let isaAssemblyFormat = "mnemonic ` ` signed($imm";
}
// MISSING_CLOSING_PAREN: [[@LINE-2]]:52: error: expected ')'
#endif

// RUN: not circt-tblgen -gen-rtg-instruction-methods -I %S/../../../include -I %S/../../../llvm/mlir/include %s -DTEST_UNEXPECTED_CHAR 2>&1 | FileCheck %s --check-prefix=UNEXPECTED_CHAR
#ifdef TEST_UNEXPECTED_CHAR
def ErrorUnexpectedCharOp : TestOp<"error_unexpected_char"> {
  let arguments = (ins TestRegType:$reg);
  let isaAssemblyFormat = "mnemonic @ $reg";
}
// UNEXPECTED_CHAR: [[@LINE-2]]:37: error: unexpected character '@'
#endif

// RUN: not circt-tblgen -gen-rtg-instruction-methods -I %S/../../../include -I %S/../../../llvm/mlir/include %s -DTEST_MISSING_OPERAND_AFTER_SIGNED 2>&1 | FileCheck %s --check-prefix=MISSING_OPERAND_AFTER_SIGNED
#ifdef TEST_MISSING_OPERAND_AFTER_SIGNED
def ErrorMissingOperandAfterSignedOp : TestOp<"error_missing_operand_after_signed"> {
  let arguments = (ins ImmediateOfWidth<12>:$imm);
  let isaAssemblyFormat = "mnemonic ` ` signed()";
}
// MISSING_OPERAND_AFTER_SIGNED: [[@LINE-2]]:48: error: expected operand after 'signed' specifier
#endif

// RUN: not circt-tblgen -gen-rtg-instruction-methods -I %S/../../../include -I %S/../../../llvm/mlir/include %s -DTEST_MISSING_OPERAND_AFTER_UNSIGNED 2>&1 | FileCheck %s --check-prefix=MISSING_OPERAND_AFTER_UNSIGNED
#ifdef TEST_MISSING_OPERAND_AFTER_UNSIGNED
def ErrorMissingOperandAfterUnsignedOp : TestOp<"error_missing_operand_after_unsigned"> {
  let arguments = (ins ImmediateOfWidth<12>:$imm);
  let isaAssemblyFormat = "mnemonic ` ` unsigned()";
}
// MISSING_OPERAND_AFTER_UNSIGNED: [[@LINE-2]]:50: error: expected operand after 'unsigned' specifier
#endif

// RUN: not circt-tblgen -gen-rtg-instruction-methods -I %S/../../../include -I %S/../../../llvm/mlir/include %s -DTEST_NEGATIVE_BIT_SLICE 2>&1 | FileCheck %s --check-prefix=NEGATIVE_BIT_SLICE
#ifdef TEST_NEGATIVE_BIT_SLICE
def ErrorNegativeBitSliceOp : TestOp<"error_negative_bit_slice"> {
  let arguments = (ins ImmediateOfWidth<12>:$imm);
  let isaBinaryFormat = "$imm[-1:5] 0b1111";
  let isaAssemblyFormat = "mnemonic ` ` signed($imm)";
}
// NEGATIVE_BIT_SLICE: [[@LINE-3]]:31: error: expected unsigned integer literal
#endif

// RUN: not circt-tblgen -gen-rtg-instruction-methods -I %S/../../../include -I %S/../../../llvm/mlir/include %s -DTEST_MNEMONIC_IN_BINARY 2>&1 | FileCheck %s --check-prefix=MNEMONIC_IN_BINARY
#ifdef TEST_MNEMONIC_IN_BINARY
def ErrorMnemonicInBinaryOp : TestOp<"error_mnemonic_in_binary"> {
  let arguments = (ins TestRegType:$reg);
  let isaBinaryFormat = "mnemonic $reg";
  let isaAssemblyFormat = "mnemonic ` ` $reg";
}
// MNEMONIC_IN_BINARY: [[@LINE-3]]:26: error: binary format cannot contain 'mnemonic' keyword
#endif

// RUN: not circt-tblgen -gen-rtg-instruction-methods -I %S/../../../include -I %S/../../../llvm/mlir/include %s -DTEST_STRING_LITERAL_IN_BINARY 2>&1 | FileCheck %s --check-prefix=STRING_LITERAL_IN_BINARY
#ifdef TEST_STRING_LITERAL_IN_BINARY
def ErrorStringLiteralInBinaryOp : TestOp<"error_string_literal_in_binary"> {
  let arguments = (ins TestRegType:$reg);
  let isaBinaryFormat = "`hello` $reg";
  let isaAssemblyFormat = "mnemonic ` ` $reg";
}
// STRING_LITERAL_IN_BINARY: [[@LINE-3]]:26: error: binary format cannot contain string literal
#endif

// RUN: not circt-tblgen -gen-rtg-instruction-methods -I %S/../../../include -I %S/../../../llvm/mlir/include %s -DTEST_SIGNEDNESS_IN_BINARY 2>&1 | FileCheck %s --check-prefix=SIGNEDNESS_IN_BINARY
#ifdef TEST_SIGNEDNESS_IN_BINARY
def ErrorSignednessInBinaryOp : TestOp<"error_signedness_in_binary"> {
  let arguments = (ins ImmediateOfWidth<12>:$imm);
  let isaBinaryFormat = "signed($imm) 0b1111";
  let isaAssemblyFormat = "mnemonic ` ` signed($imm)";
}
// SIGNEDNESS_IN_BINARY: [[@LINE-3]]:26: error: binary format cannot contain signed()/unsigned()
#endif

// RUN: not circt-tblgen -gen-rtg-instruction-methods -I %S/../../../include -I %S/../../../llvm/mlir/include %s -DTEST_BINARY_LITERAL_IN_ASSEMBLY 2>&1 | FileCheck %s --check-prefix=BINARY_LITERAL_IN_ASSEMBLY
#ifdef TEST_BINARY_LITERAL_IN_ASSEMBLY
def ErrorBinaryLiteralInAssemblyOp : TestOp<"error_binary_literal_in_assembly"> {
  let arguments = (ins TestRegType:$reg);
  let isaAssemblyFormat = "mnemonic ` ` 0b1010 $reg";
}
// BINARY_LITERAL_IN_ASSEMBLY: [[@LINE-2]]:41: error: assembly format cannot contain binary literals
#endif

// RUN: not circt-tblgen -gen-rtg-instruction-methods -I %S/../../../include -I %S/../../../llvm/mlir/include %s -DTEST_FAILED_TO_CLASSIFY 2>&1 | FileCheck %s --check-prefix=FAILED_TO_CLASSIFY
#ifdef TEST_FAILED_TO_CLASSIFY
def FailedToClassifyOp : TestOp<"error_failed_to_classify"> {
  let arguments = (ins I1:$imm);
  let isaBinaryFormat = "$imm";
  let isaAssemblyFormat = "$imm";
}
// FAILED_TO_CLASSIFY: [[@LINE-2]]:28: error: failed to classify ISA type for operand 'imm' with type 'I1'
#endif

// RUN: not circt-tblgen -gen-rtg-instruction-methods -I %S/../../../include -I %S/../../../llvm/mlir/include %s -DTEST_MISSING_SPACE 2>&1 | FileCheck %s --check-prefix=MISSING_SPACE
#ifdef TEST_MISSING_SPACE
def ErrorMissingSpaceOp : TestOp<"error_missing_space"> {
  let arguments = (ins TestRegType:$reg1, TestRegType:$reg2);
  let isaAssemblyFormat = "mnemonic ` ` $reg1$reg2";
}
// MISSING_SPACE: [[@LINE-2]]:46: error: expected ' '
#endif

// RUN: not circt-tblgen -gen-rtg-instruction-methods -I %S/../../../include -I %S/../../../llvm/mlir/include %s -DTEST_EMPTY_OPERAND_NAME 2>&1 | FileCheck %s --check-prefix=EMPTY_OPERAND_NAME
#ifdef TEST_EMPTY_OPERAND_NAME
def ErrorEmptyOperandNameOp : TestOp<"error_empty_operand_name"> {
  let arguments = (ins TestRegType:$reg);
  let isaAssemblyFormat = "mnemonic ` ` $ ` ` $reg";
}
// EMPTY_OPERAND_NAME: [[@LINE-2]]:41: error: unknown operand '$'
#endif

// RUN: not circt-tblgen -gen-rtg-instruction-methods -I %S/../../../include -I %S/../../../llvm/mlir/include %s -DTEST_MISSING_CLOSING_BRACKET 2>&1 | FileCheck %s --check-prefix=MISSING_CLOSING_BRACKET
#ifdef TEST_MISSING_CLOSING_BRACKET
def ErrorMissingClosingBracketOp : TestOp<"error_missing_closing_bracket"> {
  let arguments = (ins ImmediateOfWidth<12>:$imm);
  let isaBinaryFormat = "$imm[11:5 0b1111";
  let isaAssemblyFormat = "mnemonic ` ` signed($imm)";
}
// MISSING_CLOSING_BRACKET: [[@LINE-3]]:35: error: expected ']'
#endif

// RUN: not circt-tblgen -gen-rtg-instruction-methods -I %S/../../../include -I %S/../../../llvm/mlir/include %s -DTEST_ANYTYPE_MUST_CONTAIN_LABEL 2>&1 | FileCheck %s --check-prefix=ANYTYPE_MUST_CONTAIN_LABEL
#ifdef TEST_ANYTYPE_MUST_CONTAIN_LABEL
def ErrorAnyTypeMustContainLabelOp : TestOp<"error_anytype_must_contain_label"> {
  let arguments = (ins AnyTypeOf<[TestRegType, ImmediateOfWidth<5>]>:$src);
  let isaBinaryFormat = "$src 0b1010";
  let isaAssemblyFormat = "mnemonic ` ` unsigned($src)";
}
// ANYTYPE_MUST_CONTAIN_LABEL: [[@LINE-3]]:26: error: one of the operand types must be a label in binary format if there are 2 operand types possible
#endif

// RUN: not circt-tblgen -gen-rtg-instruction-methods -I %S/../../../include -I %S/../../../llvm/mlir/include %s -DTEST_DECIMAL_LITERAL_OVERFLOW 2>&1 | FileCheck %s --check-prefix=DECIMAL_LITERAL_OVERFLOW
#ifdef TEST_DECIMAL_LITERAL_OVERFLOW
def ErrorDecimalLiteralOverflowOp : TestOp<"error_decimal_literal_overflow"> {
  let arguments = (ins TestRegType:$reg);
  let isaBinaryFormat = "$reg 3d15";
  let isaAssemblyFormat = "mnemonic ` ` $reg";
}
// DECIMAL_LITERAL_OVERFLOW: [[@LINE-3]]:31: error: decimal literal value 15 does not fit in 3 bits
#endif

// RUN: not circt-tblgen -gen-rtg-instruction-methods -I %S/../../../include -I %S/../../../llvm/mlir/include %s -DTEST_IF_BOTH_BRANCHES_EMPTY 2>&1 | FileCheck %s --check-prefix=IF_BOTH_BRANCHES_EMPTY
#ifdef TEST_IF_BOTH_BRANCHES_EMPTY
def ErrorIfBothBranchesEmptyOp : TestOp<"error_if_both_branches_empty"> {
  let arguments = (ins ImmediateOfWidth<1>:$cond, TestRegType:$reg);
  let isaAssemblyFormat = "mnemonic ` ` if $cond then () else ()";
}
// IF_BOTH_BRANCHES_EMPTY: [[@LINE-2]]:41: error: if-then-else must have at least one of 'then' or 'else' branch
#endif

// RUN: not circt-tblgen -gen-rtg-instruction-methods -I %S/../../../include -I %S/../../../llvm/mlir/include %s -DTEST_IF_MISSING_THEN_PAREN 2>&1 | FileCheck %s --check-prefix=IF_MISSING_THEN_PAREN
#ifdef TEST_IF_MISSING_THEN_PAREN
def ErrorIfMissingThenParenOp : TestOp<"error_if_missing_then_paren"> {
  let arguments = (ins ImmediateOfWidth<1>:$cond, TestRegType:$reg);
  let isaAssemblyFormat = "mnemonic ` ` if $cond then $reg else ()";
}
// IF_MISSING_THEN_PAREN: [[@LINE-2]]:54: error: expected '('
#endif

// RUN: not circt-tblgen -gen-rtg-instruction-methods -I %S/../../../include -I %S/../../../llvm/mlir/include %s -DTEST_IF_MISSING_ELSE_PAREN 2>&1 | FileCheck %s --check-prefix=IF_MISSING_ELSE_PAREN
#ifdef TEST_IF_MISSING_ELSE_PAREN
def ErrorIfMissingElseParenOp : TestOp<"error_if_missing_else_paren"> {
  let arguments = (ins ImmediateOfWidth<1>:$cond, TestRegType:$reg);
  let isaAssemblyFormat = "mnemonic ` ` if $cond then () else $reg";
}
// IF_MISSING_ELSE_PAREN: [[@LINE-2]]:62: error: expected '('
#endif

// RUN: not circt-tblgen -gen-rtg-instruction-methods -I %S/../../../include -I %S/../../../llvm/mlir/include %s -DTEST_IF_MISSING_OPERAND 2>&1 | FileCheck %s --check-prefix=IF_MISSING_OPERAND
#ifdef TEST_IF_MISSING_OPERAND
def ErrorIfMissingOperandOp : TestOp<"error_if_missing_operand"> {
  let arguments = (ins ImmediateOfWidth<1>:$cond, TestRegType:$reg);
  let isaAssemblyFormat = "mnemonic ` ` if then ($reg) else ()";
}
// IF_MISSING_OPERAND: [[@LINE-2]]:44: error: expected operand after 'if' keyword
#endif

// RUN: not circt-tblgen -gen-rtg-instruction-methods -I %S/../../../include -I %S/../../../llvm/mlir/include %s -DTEST_IF_BOTH_BRANCHES_EMPTY2 2>&1 | FileCheck %s --check-prefix=IF_BOTH_BRANCHES_EMPTY2
#ifdef TEST_IF_BOTH_BRANCHES_EMPTY2
def ErrorIfBothBranchesEmptyOp : TestOp<"error_if_both_branches_empty"> {
  let arguments = (ins ImmediateOfWidth<1>:$cond, TestRegType:$reg);
  let isaAssemblyFormat = "mnemonic ` ` if $cond `end`";
}
// IF_BOTH_BRANCHES_EMPTY2: [[@LINE-2]]:50: error: expected 'then' or 'else' after 'if' condition
#endif
