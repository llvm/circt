// RUN: not circt-tblgen -gen-rtg-instruction-methods -I %S/../../../include -I %S/../../../llvm/mlir/include %s -DTEST_IMM_NO_WRAPPER 2>&1 | FileCheck %s --check-prefix=IMM_NO_WRAPPER
// RUN: not circt-tblgen -gen-rtg-instruction-methods -I %S/../../../include -I %S/../../../llvm/mlir/include %s -DTEST_LABEL_SIGNED 2>&1 | FileCheck %s --check-prefix=LABEL_SIGNED
// RUN: not circt-tblgen -gen-rtg-instruction-methods -I %S/../../../include -I %S/../../../llvm/mlir/include %s -DTEST_LABEL_UNSIGNED 2>&1 | FileCheck %s --check-prefix=LABEL_UNSIGNED
// RUN: not circt-tblgen -gen-rtg-instruction-methods -I %S/../../../include -I %S/../../../llvm/mlir/include %s -DTEST_REG_SIGNED 2>&1 | FileCheck %s --check-prefix=REG_SIGNED
// RUN: not circt-tblgen -gen-rtg-instruction-methods -I %S/../../../include -I %S/../../../llvm/mlir/include %s -DTEST_REG_UNSIGNED 2>&1 | FileCheck %s --check-prefix=REG_UNSIGNED
// RUN: not circt-tblgen -gen-rtg-instruction-methods -I %S/../../../include -I %S/../../../llvm/mlir/include %s -DTEST_UNKNOWN_OPERAND_ASM 2>&1 | FileCheck %s --check-prefix=UNKNOWN_OPERAND_ASM
// RUN: not circt-tblgen -gen-rtg-instruction-methods -I %S/../../../include -I %S/../../../llvm/mlir/include %s -DTEST_UNKNOWN_OPERAND_BIN 2>&1 | FileCheck %s --check-prefix=UNKNOWN_OPERAND_BIN
// RUN: not circt-tblgen -gen-rtg-instruction-methods -I %S/../../../include -I %S/../../../llvm/mlir/include %s -DTEST_INVALID_BIT_SLICE 2>&1 | FileCheck %s --check-prefix=INVALID_BIT_SLICE
// RUN: not circt-tblgen -gen-rtg-instruction-methods -I %S/../../../include -I %S/../../../llvm/mlir/include %s -DTEST_INVALID_BINARY_LITERAL 2>&1 | FileCheck %s --check-prefix=INVALID_BINARY_LITERAL
// RUN: not circt-tblgen -gen-rtg-instruction-methods -I %S/../../../include -I %S/../../../llvm/mlir/include %s -DTEST_MALFORMED_BIT_SLICE 2>&1 | FileCheck %s --check-prefix=MALFORMED_BIT_SLICE
// RUN: not circt-tblgen -gen-rtg-instruction-methods -I %S/../../../include -I %S/../../../llvm/mlir/include %s -DTEST_UNTERMINATED_BACKTICK 2>&1 | FileCheck %s --check-prefix=UNTERMINATED_BACKTICK
// RUN: not circt-tblgen -gen-rtg-instruction-methods -I %S/../../../include -I %S/../../../llvm/mlir/include %s -DTEST_MISSING_CLOSING_PAREN 2>&1 | FileCheck %s --check-prefix=MISSING_CLOSING_PAREN
// RUN: not circt-tblgen -gen-rtg-instruction-methods -I %S/../../../include -I %S/../../../llvm/mlir/include %s -DTEST_UNEXPECTED_CHAR 2>&1 | FileCheck %s --check-prefix=UNEXPECTED_CHAR
// RUN: not circt-tblgen -gen-rtg-instruction-methods -I %S/../../../include -I %S/../../../llvm/mlir/include %s -DTEST_MISSING_OPERAND_AFTER_SIGNED 2>&1 | FileCheck %s --check-prefix=MISSING_OPERAND_AFTER_SIGNED
// RUN: not circt-tblgen -gen-rtg-instruction-methods -I %S/../../../include -I %S/../../../llvm/mlir/include %s -DTEST_MISSING_OPERAND_AFTER_UNSIGNED 2>&1 | FileCheck %s --check-prefix=MISSING_OPERAND_AFTER_UNSIGNED
// RUN: not circt-tblgen -gen-rtg-instruction-methods -I %S/../../../include -I %S/../../../llvm/mlir/include %s -DTEST_NEGATIVE_BIT_SLICE 2>&1 | FileCheck %s --check-prefix=NEGATIVE_BIT_SLICE
// RUN: not circt-tblgen -gen-rtg-instruction-methods -I %S/../../../include -I %S/../../../llvm/mlir/include %s -DTEST_MNEMONIC_IN_BINARY 2>&1 | FileCheck %s --check-prefix=MNEMONIC_IN_BINARY
// RUN: not circt-tblgen -gen-rtg-instruction-methods -I %S/../../../include -I %S/../../../llvm/mlir/include %s -DTEST_STRING_LITERAL_IN_BINARY 2>&1 | FileCheck %s --check-prefix=STRING_LITERAL_IN_BINARY
// RUN: not circt-tblgen -gen-rtg-instruction-methods -I %S/../../../include -I %S/../../../llvm/mlir/include %s -DTEST_SIGNEDNESS_IN_BINARY 2>&1 | FileCheck %s --check-prefix=SIGNEDNESS_IN_BINARY
// RUN: not circt-tblgen -gen-rtg-instruction-methods -I %S/../../../include -I %S/../../../llvm/mlir/include %s -DTEST_BINARY_LITERAL_IN_ASSEMBLY 2>&1 | FileCheck %s --check-prefix=BINARY_LITERAL_IN_ASSEMBLY
// RUN: not circt-tblgen -gen-rtg-instruction-methods -I %S/../../../include -I %S/../../../llvm/mlir/include %s -DTEST_FAILED_TO_CLASSIFY 2>&1 | FileCheck %s --check-prefix=FAILED_TO_CLASSIFY
// RUN: not circt-tblgen -gen-rtg-instruction-methods -I %S/../../../include -I %S/../../../llvm/mlir/include %s -DTEST_MISSING_SPACE 2>&1 | FileCheck %s --check-prefix=MISSING_SPACE
// RUN: not circt-tblgen -gen-rtg-instruction-methods -I %S/../../../include -I %S/../../../llvm/mlir/include %s -DTEST_EMPTY_OPERAND_NAME 2>&1 | FileCheck %s --check-prefix=EMPTY_OPERAND_NAME
// RUN: not circt-tblgen -gen-rtg-instruction-methods -I %S/../../../include -I %S/../../../llvm/mlir/include %s -DTEST_MISSING_CLOSING_BRACKET 2>&1 | FileCheck %s --check-prefix=MISSING_CLOSING_BRACKET
// RUN: not circt-tblgen -gen-rtg-instruction-methods -I %S/../../../include -I %S/../../../llvm/mlir/include %s -DTEST_ANYTYPE_MUST_CONTAIN_LABEL 2>&1 | FileCheck %s --check-prefix=ANYTYPE_MUST_CONTAIN_LABEL

// NOTE: extra space to add run commands without having to change the source locations in each test




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
}

#ifdef TEST_IMM_NO_WRAPPER
def ErrorImmNoWrapperOp : TestOp<"error_imm_no_wrapper"> {
  let arguments = (ins ImmediateOfWidth<12>:$imm);
  let isaAssemblyFormat = "mnemonic ` ` $imm";
}
// IMM_NO_WRAPPER: rtg-instruction-methods-errors.td:50:41: error: immediate operand '$imm' must be wrapped in signed() or unsigned()
#endif

#ifdef TEST_LABEL_SIGNED
def ErrorLabelSignedOp : TestOp<"error_label_signed"> {
  let arguments = (ins LabelType:$label);
  let isaAssemblyFormat = "mnemonic ` ` signed($label)";
}
// LABEL_SIGNED: rtg-instruction-methods-errors.td:58:48: error: label operand '$label' cannot be wrapped in signed() or unsigned()
#endif

#ifdef TEST_LABEL_UNSIGNED
def ErrorLabelUnsignedOp : TestOp<"error_label_unsigned"> {
  let arguments = (ins LabelType:$label);
  let isaAssemblyFormat = "mnemonic ` ` unsigned($label)";
}
// LABEL_UNSIGNED: rtg-instruction-methods-errors.td:66:50: error: label operand '$label' cannot be wrapped in signed() or unsigned()
#endif

#ifdef TEST_REG_SIGNED
def ErrorRegSignedOp : TestOp<"error_reg_signed"> {
  let arguments = (ins TestRegType:$reg);
  let isaAssemblyFormat = "mnemonic ` ` signed($reg)";
}
// REG_SIGNED: rtg-instruction-methods-errors.td:74:48: error: register operand '$reg' cannot be wrapped in signed() or unsigned()
#endif

#ifdef TEST_REG_UNSIGNED
def ErrorRegUnsignedOp : TestOp<"error_reg_unsigned"> {
  let arguments = (ins TestRegType:$reg);
  let isaAssemblyFormat = "mnemonic ` ` unsigned($reg)";
}
// REG_UNSIGNED: rtg-instruction-methods-errors.td:82:50: error: register operand '$reg' cannot be wrapped in signed() or unsigned()
#endif

#ifdef TEST_UNKNOWN_OPERAND_ASM
def ErrorUnknownOperandAsmOp : TestOp<"error_unknown_operand_asm"> {
  let arguments = (ins TestRegType:$reg);
  let isaAssemblyFormat = "mnemonic ` ` $unknown";
}
// UNKNOWN_OPERAND_ASM: rtg-instruction-methods-errors.td:90:41: error: unknown operand '$unknown'
#endif

#ifdef TEST_UNKNOWN_OPERAND_BIN
def ErrorUnknownOperandBinOp : TestOp<"error_unknown_operand_bin"> {
  let arguments = (ins TestRegType:$reg);
  let isaBinaryFormat = "$unknown 0b1111";
  let isaAssemblyFormat = "mnemonic ` ` $reg";
}
// UNKNOWN_OPERAND_BIN: rtg-instruction-methods-errors.td:98:26: error: unknown operand '$unknown'
#endif

#ifdef TEST_INVALID_BIT_SLICE
def ErrorInvalidBitSliceOp : TestOp<"error_invalid_bit_slice"> {
  let arguments = (ins ImmediateOfWidth<12>:$imm);
  let isaBinaryFormat = "$imm[5:10] 0b1111";
  let isaAssemblyFormat = "mnemonic ` ` signed($imm)";
}
// INVALID_BIT_SLICE: rtg-instruction-methods-errors.td:107:30: error: high bit (5) must be >= low bit (10)
#endif

#ifdef TEST_INVALID_BINARY_LITERAL
def ErrorInvalidBinaryLiteralOp : TestOp<"error_invalid_binary_literal"> {
  let arguments = (ins TestRegType:$reg);
  let isaBinaryFormat = "$reg 0b10102";
  let isaAssemblyFormat = "mnemonic ` ` $reg";
}
// INVALID_BINARY_LITERAL: rtg-instruction-methods-errors.td:116:37: error: invalid character ('2') after binary literal
#endif

#ifdef TEST_MALFORMED_BIT_SLICE
def ErrorMalformedBitSliceOp : TestOp<"error_malformed_bit_slice"> {
  let arguments = (ins ImmediateOfWidth<12>:$imm);
  let isaBinaryFormat = "$imm[:5] 0b1111";
  let isaAssemblyFormat = "mnemonic ` ` signed($imm)";
}
// MALFORMED_BIT_SLICE: rtg-instruction-methods-errors.td:125:31: error: expected integer literal
#endif

#ifdef TEST_UNTERMINATED_BACKTICK
def ErrorUnterminatedBacktickOp : TestOp<"error_unterminated_backtick"> {
  let arguments = (ins TestRegType:$reg);
  let isaAssemblyFormat = "mnemonic ` unterminated";
}
// UNTERMINATED_BACKTICK: rtg-instruction-methods-errors.td:134:37: error: unterminated backtick literal
#endif

#ifdef TEST_MISSING_CLOSING_PAREN
def ErrorMissingClosingParenOp : TestOp<"error_missing_closing_paren"> {
  let arguments = (ins ImmediateOfWidth<12>:$imm);
  let isaAssemblyFormat = "mnemonic ` ` signed($imm";
}
// MISSING_CLOSING_PAREN: rtg-instruction-methods-errors.td:142:52: error: expected ')'
#endif

#ifdef TEST_UNEXPECTED_CHAR
def ErrorUnexpectedCharOp : TestOp<"error_unexpected_char"> {
  let arguments = (ins TestRegType:$reg);
  let isaAssemblyFormat = "mnemonic @ $reg";
}
// UNEXPECTED_CHAR: rtg-instruction-methods-errors.td:150:37: error: unexpected character '@'
#endif

#ifdef TEST_MISSING_OPERAND_AFTER_SIGNED
def ErrorMissingOperandAfterSignedOp : TestOp<"error_missing_operand_after_signed"> {
  let arguments = (ins ImmediateOfWidth<12>:$imm);
  let isaAssemblyFormat = "mnemonic ` ` signed()";
}
// MISSING_OPERAND_AFTER_SIGNED: rtg-instruction-methods-errors.td:158:48: error: expected operand after 'signed' specifier
#endif

#ifdef TEST_MISSING_OPERAND_AFTER_UNSIGNED
def ErrorMissingOperandAfterUnsignedOp : TestOp<"error_missing_operand_after_unsigned"> {
  let arguments = (ins ImmediateOfWidth<12>:$imm);
  let isaAssemblyFormat = "mnemonic ` ` unsigned()";
}
// MISSING_OPERAND_AFTER_UNSIGNED: rtg-instruction-methods-errors.td:166:50: error: expected operand after 'unsigned' specifier
#endif

#ifdef TEST_NEGATIVE_BIT_SLICE
def ErrorNegativeBitSliceOp : TestOp<"error_negative_bit_slice"> {
  let arguments = (ins ImmediateOfWidth<12>:$imm);
  let isaBinaryFormat = "$imm[-1:5] 0b1111";
  let isaAssemblyFormat = "mnemonic ` ` signed($imm)";
}
// NEGATIVE_BIT_SLICE: rtg-instruction-methods-errors.td:174:30: error: slice bits must be non-negative
#endif

#ifdef TEST_MNEMONIC_IN_BINARY
def ErrorMnemonicInBinaryOp : TestOp<"error_mnemonic_in_binary"> {
  let arguments = (ins TestRegType:$reg);
  let isaBinaryFormat = "mnemonic $reg";
  let isaAssemblyFormat = "mnemonic ` ` $reg";
}
// MNEMONIC_IN_BINARY: rtg-instruction-methods-errors.td:183:26: error: binary format cannot contain 'mnemonic' keyword
#endif

#ifdef TEST_STRING_LITERAL_IN_BINARY
def ErrorStringLiteralInBinaryOp : TestOp<"error_string_literal_in_binary"> {
  let arguments = (ins TestRegType:$reg);
  let isaBinaryFormat = "`hello` $reg";
  let isaAssemblyFormat = "mnemonic ` ` $reg";
}
// STRING_LITERAL_IN_BINARY: rtg-instruction-methods-errors.td:192:26: error: binary format cannot contain string literal
#endif

#ifdef TEST_SIGNEDNESS_IN_BINARY
def ErrorSignednessInBinaryOp : TestOp<"error_signedness_in_binary"> {
  let arguments = (ins ImmediateOfWidth<12>:$imm);
  let isaBinaryFormat = "signed($imm) 0b1111";
  let isaAssemblyFormat = "mnemonic ` ` signed($imm)";
}
// SIGNEDNESS_IN_BINARY: rtg-instruction-methods-errors.td:201:26: error: binary format cannot contain signed()/unsigned()
#endif

#ifdef TEST_BINARY_LITERAL_IN_ASSEMBLY
def ErrorBinaryLiteralInAssemblyOp : TestOp<"error_binary_literal_in_assembly"> {
  let arguments = (ins TestRegType:$reg);
  let isaBinaryFormat = "$reg 0b1111";
  let isaAssemblyFormat = "mnemonic ` ` 0b1010 $reg";
}
// BINARY_LITERAL_IN_ASSEMBLY: rtg-instruction-methods-errors.td:211:41: error: assembly format cannot contain binary literals
#endif

#ifdef TEST_FAILED_TO_CLASSIFY
def FailedToClassifyOp : TestOp<"error_failed_to_classify"> {
  let arguments = (ins I1:$imm);
  let isaBinaryFormat = "$imm";
  let isaAssemblyFormat = "$imm";
}
// FAILED_TO_CLASSIFY: rtg-instruction-methods-errors.td:220:28: error: failed to classify ISA type for operand 'imm' with type 'I1'
#endif

#ifdef TEST_MISSING_SPACE
def ErrorMissingSpaceOp : TestOp<"error_missing_space"> {
  let arguments = (ins TestRegType:$reg1, TestRegType:$reg2);
  let isaAssemblyFormat = "mnemonic ` ` $reg1$reg2";
}
// MISSING_SPACE: rtg-instruction-methods-errors.td:228:46: error: expected ' '
#endif

#ifdef TEST_EMPTY_OPERAND_NAME
def ErrorEmptyOperandNameOp : TestOp<"error_empty_operand_name"> {
  let arguments = (ins TestRegType:$reg);
  let isaAssemblyFormat = "mnemonic ` ` $ ` ` $reg";
}
// EMPTY_OPERAND_NAME: rtg-instruction-methods-errors.td:236:41: error: unknown operand '$'
#endif

#ifdef TEST_MISSING_CLOSING_BRACKET
def ErrorMissingClosingBracketOp : TestOp<"error_missing_closing_bracket"> {
  let arguments = (ins ImmediateOfWidth<12>:$imm);
  let isaBinaryFormat = "$imm[11:5 0b1111";
  let isaAssemblyFormat = "mnemonic ` ` signed($imm)";
}
// MISSING_CLOSING_BRACKET: rtg-instruction-methods-errors.td:244:35: error: expected ']'
#endif

#ifdef TEST_ANYTYPE_MUST_CONTAIN_LABEL
def ErrorAnyTypeMustContainLabelOp : TestOp<"error_anytype_must_contain_label"> {
  let arguments = (ins AnyTypeOf<[TestRegType, ImmediateOfWidth<5>]>:$src);
  let isaBinaryFormat = "$src 0b1010";
  let isaAssemblyFormat = "mnemonic ` ` unsigned($src)";
}
// ANYTYPE_MUST_CONTAIN_LABEL: rtg-instruction-methods-errors.td:253:26: error: one of the operand types must be a label in binary format if there are 2 operand types possible
#endif
