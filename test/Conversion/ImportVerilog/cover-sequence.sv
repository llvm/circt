// RUN: circt-verilog --ir-moore %s | FileCheck %s --check-prefix=MOORE
// RUN: circt-verilog --ir-llhd %s | FileCheck %s --check-prefix=LOWERED
// RUN: circt-verilog --ir-hw %s | FileCheck %s --check-prefix=LOWERED
// REQUIRES: slang
// UNSUPPORTED: valgrind

// A Boolean concurrent cover retains its sampling clock.
// MOORE-LABEL: moore.module @simple(
// MOORE: [[VALID_INT:%.+]] = moore.logic_to_int %valid : l1
// MOORE: [[VALID:%.+]] = moore.to_builtin_int [[VALID_INT]] : i1
// MOORE: [[CLK_INT:%.+]] = moore.logic_to_int %clk : l1
// MOORE: [[CLK:%.+]] = moore.to_builtin_int [[CLK_INT]] : i1
// MOORE: verif.clocked_cover [[VALID]], posedge [[CLK]] : i1
// LOWERED-LABEL: hw.module @simple(
// LOWERED: verif.clocked_cover %valid, posedge %clk : i1
module simple(input logic clk, valid);
  cover sequence (@(posedge clk) valid);
endmodule

// The disable condition is negated to form the cover's enable operand.
// Also check that a falling-edge sampling clock is preserved.
// MOORE-LABEL: moore.module @disabled(
// MOORE: [[NOT_RST:%.+]] = moore.not %rst : l1
// MOORE: [[ENABLE_INT:%.+]] = moore.logic_to_int [[NOT_RST]] : l1
// MOORE: [[ENABLE:%.+]] = moore.to_builtin_int [[ENABLE_INT]] : i1
// MOORE: [[VALID_INT:%.+]] = moore.logic_to_int %valid : l1
// MOORE: [[VALID:%.+]] = moore.to_builtin_int [[VALID_INT]] : i1
// MOORE: [[CLK_INT:%.+]] = moore.logic_to_int %clk : l1
// MOORE: [[CLK:%.+]] = moore.to_builtin_int [[CLK_INT]] : i1
// MOORE: verif.clocked_cover [[VALID]] if [[ENABLE]], negedge [[CLK]] : i1
// LOWERED-LABEL: hw.module @disabled(
// LOWERED: [[TRUE:%.+]] = hw.constant true
// LOWERED: [[ENABLE:%.+]] = comb.xor %rst, [[TRUE]] : i1
// LOWERED: verif.clocked_cover %valid if [[ENABLE]], negedge %clk : i1
module disabled(input logic clk, rst, valid);
  cover sequence (@(negedge clk) disable iff (rst) valid);
endmodule

// Multi-cycle coverage retains both the delay and consecutive repetition.
// MOORE-LABEL: moore.module @sequence_cover(
// MOORE: [[NOT_RST:%.+]] = moore.not %rst : l1
// MOORE: [[ENABLE_INT:%.+]] = moore.logic_to_int [[NOT_RST]] : l1
// MOORE: [[ENABLE:%.+]] = moore.to_builtin_int [[ENABLE_INT]] : i1
// MOORE: [[VALID_INT:%.+]] = moore.logic_to_int %valid : l1
// MOORE: [[VALID:%.+]] = moore.to_builtin_int [[VALID_INT]] : i1
// MOORE: [[FIRST:%.+]] = ltl.delay [[VALID]], 0, 0 : i1
// MOORE: [[DONE_INT:%.+]] = moore.logic_to_int %done : l1
// MOORE: [[DONE:%.+]] = moore.to_builtin_int [[DONE_INT]] : i1
// MOORE: [[REPEAT:%.+]] = ltl.repeat [[DONE]], 2, 0 : i1
// MOORE: [[DELAY:%.+]] = ltl.delay [[REPEAT]], 1, 0 : !ltl.sequence
// MOORE: [[SEQ:%.+]] = ltl.concat [[FIRST]], [[DELAY]] : !ltl.sequence, !ltl.sequence
// MOORE: [[CLK_INT:%.+]] = moore.logic_to_int %clk : l1
// MOORE: [[CLK:%.+]] = moore.to_builtin_int [[CLK_INT]] : i1
// MOORE: verif.clocked_cover [[SEQ]] if [[ENABLE]], posedge [[CLK]] : !ltl.sequence
// LOWERED-LABEL: hw.module @sequence_cover(
// LOWERED: [[TRUE:%.+]] = hw.constant true
// LOWERED: [[ENABLE:%.+]] = comb.xor %rst, [[TRUE]] : i1
// LOWERED: [[FIRST:%.+]] = ltl.delay %valid, 0, 0 : i1
// LOWERED: [[REPEAT:%.+]] = ltl.repeat %done, 2, 0 : i1
// LOWERED: [[DELAY:%.+]] = ltl.delay [[REPEAT]], 1, 0 : !ltl.sequence
// LOWERED: [[SEQ:%.+]] = ltl.concat [[FIRST]], [[DELAY]] : !ltl.sequence, !ltl.sequence
// LOWERED: verif.clocked_cover [[SEQ]] if [[ENABLE]], posedge %clk : !ltl.sequence
module sequence_cover(input logic clk, rst, valid, done);
  cover sequence (@(posedge clk) disable iff (rst) valid ##1 done[*2]);
endmodule
