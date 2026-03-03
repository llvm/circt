// RUN: circt-opt --rtg-emit-isa-assembly %s 2>&1 >/dev/null | FileCheck %s --check-prefix=CHECK-ALLOWED --match-full-lines --strict-whitespace
// RUN: circt-opt --rtg-emit-isa-assembly="unsupported-instructions=rtgtest.zero_operand_instr,rtgtest.two_register_instr unsupported-instructions-file=%S/unsupported-instr.txt" %s 2>&1 >/dev/null | FileCheck %s --match-full-lines --strict-whitespace

// CHECK:    # Begin of test0
// CHECK-ALLOWED:    # Begin of test0

emit.file "" {
  %str_begin = rtg.constant "Begin of test0" : !rtg.string
  rtg.comment %str_begin

  %idx8 = index.constant 8
  // CHECK-ALLOWED-NEXT:    .space 8
  // CHECK-NEXT:    .space 8
  rtg.isa.space %idx8

  // CHECK-ALLOWED-NEXT:.data
  // CHECK-NEXT:.data
  rtg.isa.segment data {
    // CHECK-ALLOWED-NEXT:    # data segment
    // CHECK-NEXT:    # data segment
    %str_data = rtg.constant "data segment" : !rtg.string
    rtg.comment %str_data
  }

  // CHECK-ALLOWED-NEXT:.text
  // CHECK-NEXT:.text
  rtg.isa.segment text {
    // CHECK-ALLOWED-NEXT:    # text segment
    // CHECK-NEXT:    # text segment
    %str_text = rtg.constant "text segment" : !rtg.string
    rtg.comment %str_text
  }

  // CHECK-ALLOWED-NEXT:    .asciz "hello world\n\t\\\""
  // CHECK-NEXT:    .asciz "hello world\n\t\\\""
  %str_hello = rtg.constant "hello world\n\t\\\"" : !rtg.string
  rtg.isa.string_data %str_hello

  %rd = rtg.constant #rtgtest.ra
  %rs = rtg.constant #rtgtest.s0
  %imm32 = rtg.constant #rtg.isa.immediate<32, 0>

  // CHECK-ALLOWED-NEXT:    zero_operand_instr
  // CHECK-NEXT:    # zero_operand_instr
  // CHECK-NEXT:    .word 0x12345678
  rtgtest.zero_operand_instr

  // CHECK-ALLOWED-NEXT:    two_register_instr ra, s0
  // CHECK-NEXT:    # two_register_instr ra, s0
  // CHECK-NEXT:    .word 0x40080
  rtgtest.two_register_instr %rd, %rs

  // CHECK-ALLOWED-NEXT:    immediate_instr ra, 0
  // CHECK-NEXT:    immediate_instr ra, 0
  rtgtest.immediate_instr %rd, %imm32

  %str_end = rtg.constant "End of test0" : !rtg.string
  rtg.comment %str_end
}

// CHECK-NEXT:    # End of test0
// CHECK-ALLOWED-NEXT:    # End of test0
