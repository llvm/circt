// RUN: circt-translate --import-verilog %s | FileCheck %s
// RUN: circt-verilog --ir-moore %s
// REQUIRES: slang

// Internal issue in Slang v3 about jump depending on uninitialised value.
// UNSUPPORTED: valgrind

// CHECK-LABEL: moore.module @and_prim()
// CHECK: [[A:%.+]] = moore.net wire : <l1>
// CHECK: [[B:%.+]] = moore.net wire : <l1>
// CHECK: [[Q:%.+]] = moore.net wire : <l1>
// CHECK: [[RD_A:%.+]] = moore.read [[A]] : ref<l1>
// CHECK: [[RD_B:%.+]] = moore.read [[B]] : ref<l1>
// CHECK: [[AND:%.+]] = moore.and [[RD_A]], [[RD_B]] : l1
// CHECK: moore.assign [[Q]], [[AND]] : ref<l1>

module and_prim;
    wire A, B, Q;
    and a (Q, A, B);
endmodule

// CHECK-LABEL: moore.module @or_prim()
// CHECK: [[A:%.+]] = moore.net wire : <l1>
// CHECK: [[B:%.+]] = moore.net wire : <l1>
// CHECK: [[Q:%.+]] = moore.net wire : <l1>
// CHECK: [[RD_A:%.+]] = moore.read [[A]] : ref<l1>
// CHECK: [[RD_B:%.+]] = moore.read [[B]] : ref<l1>
// CHECK: [[OR:%.+]] = moore.or [[RD_A]], [[RD_B]] : l1
// CHECK: moore.assign [[Q]], [[OR]] : ref<l1>

module or_prim;
    wire A, B, Q;
    or a (Q, A, B);
endmodule

// CHECK-LABEL: moore.module @xor_prim()
// CHECK: [[A:%.+]] = moore.net wire : <l1>
// CHECK: [[B:%.+]] = moore.net wire : <l1>
// CHECK: [[Q:%.+]] = moore.net wire : <l1>
// CHECK: [[RD_A:%.+]] = moore.read [[A]] : ref<l1>
// CHECK: [[RD_B:%.+]] = moore.read [[B]] : ref<l1>
// CHECK: [[XOR:%.+]] = moore.xor [[RD_A]], [[RD_B]] : l1
// CHECK: moore.assign [[Q]], [[XOR]] : ref<l1>

module xor_prim;
    wire A, B, Q;
    xor c (Q, A, B);
endmodule

// CHECK-LABEL: moore.module @multi_and_prim()
// CHECK: [[A:%.+]] = moore.net wire : <l1>
// CHECK: [[B:%.+]] = moore.net wire : <l1>
// CHECK: [[C:%.+]] = moore.net wire : <l1>
// CHECK: [[D:%.+]] = moore.net wire : <l1>
// CHECK: [[Q:%.+]] = moore.net wire : <l1>
// CHECK: [[RD_A:%.+]] = moore.read [[A]] : ref<l1>
// CHECK: [[RD_B:%.+]] = moore.read [[B]] : ref<l1>
// CHECK: [[RD_C:%.+]] = moore.read [[C]] : ref<l1>
// CHECK: [[RD_D:%.+]] = moore.read [[D]] : ref<l1>
// CHECK: [[AND0:%.+]] = moore.and [[RD_A]], [[RD_B]] : l1
// CHECK: [[AND1:%.+]] = moore.and [[AND0]], [[RD_C]] : l1
// CHECK: [[AND2:%.+]] = moore.and [[AND1]], [[RD_D]] : l1
// CHECK: moore.assign [[Q]], [[AND2]] : ref<l1>

module multi_and_prim;
    wire A, B, C, D, Q;
    and a (Q, A, B, C, D);
endmodule

// CHECK-LABEL: moore.module @nand_prim()
// CHECK: [[A:%.+]] = moore.net wire : <l1>
// CHECK: [[B:%.+]] = moore.net wire : <l1>
// CHECK: [[Q:%.+]] = moore.net wire : <l1>
// CHECK: [[RD_A:%.+]] = moore.read [[A]] : ref<l1>
// CHECK: [[RD_B:%.+]] = moore.read [[B]] : ref<l1>
// CHECK: [[AND:%.+]] = moore.and [[RD_A]], [[RD_B]] : l1
// CHECK: [[NOT_AND:%.+]] = moore.not [[AND]] : l1
// CHECK: moore.assign [[Q]], [[NOT_AND]] : ref<l1>

module nand_prim;
    wire A, B, Q;
    nand a (Q, A, B);
endmodule

// CHECK-LABEL: moore.module @nor_prim()
// CHECK: [[A:%.+]] = moore.net wire : <l1>
// CHECK: [[B:%.+]] = moore.net wire : <l1>
// CHECK: [[Q:%.+]] = moore.net wire : <l1>
// CHECK: [[RD_A:%.+]] = moore.read [[A]] : ref<l1>
// CHECK: [[RD_B:%.+]] = moore.read [[B]] : ref<l1>
// CHECK: [[OR:%.+]] = moore.or [[RD_A]], [[RD_B]] : l1
// CHECK: [[NOT_OR:%.+]] = moore.not [[OR]] : l1
// CHECK: moore.assign [[Q]], [[NOT_OR]] : ref<l1>

module nor_prim;
    wire A, B, Q;
    nor a (Q, A, B);
endmodule

// CHECK-LABEL: moore.module @xnor_prim()
// CHECK: [[A:%.+]] = moore.net wire : <l1>
// CHECK: [[B:%.+]] = moore.net wire : <l1>
// CHECK: [[Q:%.+]] = moore.net wire : <l1>
// CHECK: [[RD_A:%.+]] = moore.read [[A]] : ref<l1>
// CHECK: [[RD_B:%.+]] = moore.read [[B]] : ref<l1>
// CHECK: [[XOR:%.+]] = moore.xor [[RD_A]], [[RD_B]] : l1
// CHECK: [[NOT_XOR:%.+]] = moore.not [[XOR]] : l1
// CHECK: moore.assign [[Q]], [[NOT_XOR]] : ref<l1>

module xnor_prim;
    wire A, B, Q;
    xnor c (Q, A, B);
endmodule

// CHECK-LABEL: moore.module @multi_nand_prim()
// CHECK: [[A:%.+]] = moore.net wire : <l1>
// CHECK: [[B:%.+]] = moore.net wire : <l1>
// CHECK: [[C:%.+]] = moore.net wire : <l1>
// CHECK: [[D:%.+]] = moore.net wire : <l1>
// CHECK: [[Q:%.+]] = moore.net wire : <l1>
// CHECK: [[RD_A:%.+]] = moore.read [[A]] : ref<l1>
// CHECK: [[RD_B:%.+]] = moore.read [[B]] : ref<l1>
// CHECK: [[RD_C:%.+]] = moore.read [[C]] : ref<l1>
// CHECK: [[RD_D:%.+]] = moore.read [[D]] : ref<l1>
// CHECK: [[AND0:%.+]] = moore.and [[RD_A]], [[RD_B]] : l1
// CHECK: [[AND1:%.+]] = moore.and [[AND0]], [[RD_C]] : l1
// CHECK: [[AND2:%.+]] = moore.and [[AND1]], [[RD_D]] : l1
// CHECK: [[NOT_AND:%.+]] = moore.not [[AND2]] : l1
// CHECK: moore.assign [[Q]], [[NOT_AND]] : ref<l1>

module multi_nand_prim;
    wire A, B, C, D, Q;
    nand a (Q, A, B, C, D);
endmodule

// CHECK-LABEL: moore.module @delayed_nand_prim()
// CHECK: [[A:%.+]] = moore.net wire : <l1>
// CHECK: [[B:%.+]] = moore.net wire : <l1>
// CHECK: [[Q:%.+]] = moore.net wire : <l1>
// CHECK: [[RD_A:%.+]] = moore.read [[A]] : ref<l1>
// CHECK: [[RD_B:%.+]] = moore.read [[B]] : ref<l1>
// CHECK: [[AND:%.+]] = moore.and [[RD_A]], [[RD_B]] : l1
// CHECK: [[NOT_AND:%.+]] = moore.not [[AND]] : l1
// CHECK: [[DELAYCONST:%.+]] = moore.constant_time 5000000 fs
// CHECK: moore.delayed_assign [[Q]], [[NOT_AND]], [[DELAYCONST]] : ref<l1>

module delayed_nand_prim;
    wire A, B, Q;
    nand #5 a (Q, A, B);
endmodule

// CHECK-LABEL: moore.module @delayed3_nand_prim()
// CHECK: [[A:%.+]] = moore.net wire : <l1>
// CHECK: [[B:%.+]] = moore.net wire : <l1>
// CHECK: [[Q:%.+]] = moore.net wire : <l1>
// CHECK: [[RD_A:%.+]] = moore.read [[A]] : ref<l1>
// CHECK: [[RD_B:%.+]] = moore.read [[B]] : ref<l1>
// CHECK: [[AND:%.+]] = moore.and [[RD_A]], [[RD_B]] : l1
// CHECK: [[NOT_AND:%.+]] = moore.not [[AND]] : l1
// CHECK: [[DELAYCONST:%.+]] = moore.constant_time 5000000 fs
// CHECK: moore.delayed_assign [[Q]], [[NOT_AND]], [[DELAYCONST]] : ref<l1>

module delayed3_nand_prim;
    wire A, B, Q;
    nand #(5) a (Q, A, B);
endmodule

// CHECK-LABEL: moore.module @not_prim()
// CHECK: [[A:%.+]] = moore.net wire : <l1>
// CHECK: [[Q:%.+]] = moore.net wire : <l1>
// CHECK: [[RD_A:%.+]] = moore.read [[A]] : ref<l1>
// CHECK: [[NOT:%.+]] = moore.not [[RD_A]] : l1
// CHECK: moore.assign [[Q]], [[NOT]] : ref<l1>

module not_prim;
    wire A, Q;
    not n (Q, A);
endmodule

// CHECK-LABEL: moore.module @multi_not_prim()
// CHECK: [[A:%.+]] = moore.net wire : <l1>
// CHECK: [[Q0:%.+]] = moore.net wire : <l1>
// CHECK: [[Q1:%.+]] = moore.net wire : <l1>
// CHECK: [[RD_A:%.+]] = moore.read [[A]] : ref<l1>
// CHECK: [[NOT:%.+]] = moore.not [[RD_A]] : l1
// CHECK: moore.assign [[Q0]], [[NOT]] : ref<l1>
// CHECK: moore.assign [[Q1]], [[NOT]] : ref<l1>

module multi_not_prim;
    wire A, Q0, Q1;
    not n (Q0, Q1, A);
endmodule

// CHECK-LABEL: moore.module @buf_prim()
// CHECK: [[A:%.+]] = moore.net wire : <l1>
// CHECK: [[Q:%.+]] = moore.net wire : <l1>
// CHECK: [[RD_A:%.+]] = moore.read [[A]] : ref<l1>
// CHECK: [[NOT:%.+]] = moore.bool_cast [[RD_A]] : l1
// CHECK: moore.assign [[Q]], [[NOT]] : ref<l1>

module buf_prim;
    wire A, Q;
    buf n (Q, A);
endmodule

// CHECK-LABEL: moore.module @delayed_buf_prim()
// CHECK: [[A:%.+]] = moore.net wire : <l1>
// CHECK: [[Q:%.+]] = moore.net wire : <l1>
// CHECK: [[RD_A:%.+]] = moore.read [[A]] : ref<l1>
// CHECK: [[NOT:%.+]] = moore.bool_cast [[RD_A]] : l1
// CHECK: [[DELAYCONST:%.+]] = moore.constant_time 5000000 fs
// CHECK: moore.delayed_assign [[Q]], [[NOT]], [[DELAYCONST]] : ref<l1>


module delayed_buf_prim;
    wire A, Q;
    buf #5 n (Q, A);
endmodule

// CHECK-LABEL: moore.module @delayed3_buf_prim()
// CHECK: [[A:%.+]] = moore.net wire : <l1>
// CHECK: [[Q:%.+]] = moore.net wire : <l1>
// CHECK: [[RD_A:%.+]] = moore.read [[A]] : ref<l1>
// CHECK: [[NOT:%.+]] = moore.bool_cast [[RD_A]] : l1
// CHECK: [[DELAYCONST:%.+]] = moore.constant_time 5000000 fs
// CHECK: moore.delayed_assign [[Q]], [[NOT]], [[DELAYCONST]] : ref<l1>

module delayed3_buf_prim;
    wire A, Q;
    buf #(5) n (Q, A);
endmodule


// CHECK-LABEL: moore.module @pullup_prim()
// CHECK: [[A:%.+]] = moore.net wire : <l1>
// CHECK: [[ONE:%.+]] = moore.constant 1 : l1
// CHECK: moore.assign [[A]], [[ONE]] : ref<l1>

module pullup_prim;
    wire A;
    pullup n (A);
endmodule

// CHECK-LABEL: moore.module @pulldown_prim()
// CHECK: [[A:%.+]] = moore.net wire : <l1>
// CHECK: [[ZERO:%.+]] = moore.constant 0 : l1
// CHECK: moore.assign [[A]], [[ZERO]] : ref<l1>

module pulldown_prim;
    wire A;
    pulldown n (A);
endmodule

// CHECK-LABEL: moore.module @wide_pullup_prim()
// CHECK: [[A:%.+]] = moore.net wire : <l4>
// CHECK: [[ONES:%.+]] = moore.constant -1 : l4
// CHECK: moore.assign [[A]], [[ONES]] : ref<l4>

module wide_pullup_prim;
    wire [3:0] A;
    pullup n (A);
endmodule

// CHECK-LABEL: moore.module @TestBufif0
module TestBufif0(
  input wire data_in,
  input wire en_n,
  output wire data_out
);
  // CHECK: [[DATA_IN:%.+]] = moore.net name "data_in" wire : <l1>
  // CHECK: [[EN_N:%.+]] = moore.net name "en_n" wire : <l1>
  // CHECK: [[DATA_OUT:%.+]] = moore.net wire : <l1>
  // CHECK: [[IN:%.+]] = moore.read [[DATA_IN]] : ref<l1>
  // CHECK: [[EN:%.+]] = moore.read [[EN_N]] : ref<l1>
  // CHECK: [[INACTIVE:%.+]] = moore.constant 1 : l1
  // CHECK: [[Z:%.+]] = moore.constant bZ : l1
  // CHECK: [[COND:%.+]] = moore.case_eq [[EN]], [[INACTIVE]] : l1
  // CHECK: [[RESULT:%.+]] = moore.conditional [[COND]] : i1 -> l1
  // CHECK: moore.yield [[Z]] : l1
  // CHECK: [[INNERX:%.+]] = moore.constant bX : l1
  // CHECK: [[INNERZ:%.+]] = moore.constant bZ : l1
  // CHECK: [[ISZ:%.+]] = moore.case_eq [[IN]], [[INNERZ]] : l1
  // CHECK: [[INNERRESULT:%.+]] = moore.conditional [[ISZ]] : i1 -> l1
  // CHECK: moore.yield [[INNERX]] : l1
  // CHECK: moore.yield [[IN]] : l1
  // CHECK: moore.yield [[INNERRESULT]] : l1
  // CHECK: moore.assign [[DATA_OUT]], [[RESULT]] : ref<l1>
  bufif0 bufif0_inst(data_out, data_in, en_n);

endmodule

// CHECK-LABEL: moore.module @TestBufif1
module TestBufif1(
  input wire data_in,
  input wire en_n,
  output wire data_out
);
  // CHECK: [[DATA_IN:%.+]] = moore.net name "data_in" wire : <l1>
  // CHECK: [[EN_N:%.+]] = moore.net name "en_n" wire : <l1>
  // CHECK: [[DATA_OUT:%.+]] = moore.net wire : <l1>
  // CHECK: [[IN:%.+]] = moore.read [[DATA_IN]] : ref<l1>
  // CHECK: [[EN:%.+]] = moore.read [[EN_N]] : ref<l1>
  // CHECK: [[INACTIVE:%.+]] = moore.constant 0 : l1
  // CHECK: [[Z:%.+]] = moore.constant bZ : l1
  // CHECK: [[COND:%.+]] = moore.case_eq [[EN]], [[INACTIVE]] : l1
  // CHECK: [[RESULT:%.+]] = moore.conditional [[COND]] : i1 -> l1
  // CHECK: moore.yield [[Z]] : l1
  // CHECK: [[INNERX:%.+]] = moore.constant bX : l1
  // CHECK: [[INNERZ:%.+]] = moore.constant bZ : l1
  // CHECK: [[ISZ:%.+]] = moore.case_eq [[IN]], [[INNERZ]] : l1
  // CHECK: [[INNERRESULT:%.+]] = moore.conditional [[ISZ]] : i1 -> l1
  // CHECK: moore.yield [[INNERX]] : l1
  // CHECK: moore.yield [[IN]] : l1
  // CHECK: moore.yield [[INNERRESULT]] : l1
  // CHECK: moore.assign [[DATA_OUT]], [[RESULT]] : ref<l1>
  bufif1 bufif1_inst(data_out, data_in, en_n);

endmodule

// CHECK-LABEL: moore.module @TestNotif0
module TestNotif0(
  input wire data_in,
  input wire en_n,
  output wire data_out
);
  // CHECK: [[DATA_IN:%.+]] = moore.net name "data_in" wire : <l1>
  // CHECK: [[EN_N:%.+]] = moore.net name "en_n" wire : <l1>
  // CHECK: [[DATA_OUT:%.+]] = moore.net wire : <l1>
  // CHECK: [[IN:%.+]] = moore.read [[DATA_IN]] : ref<l1>
  // CHECK: [[EN:%.+]] = moore.read [[EN_N]] : ref<l1>
  // CHECK: [[NOT_IN:%.+]] = moore.not [[IN]] : l1
  // CHECK: [[INACTIVE:%.+]] = moore.constant 1 : l1
  // CHECK: [[Z:%.+]] = moore.constant bZ : l1
  // CHECK: [[COND:%.+]] = moore.case_eq [[EN]], [[INACTIVE]] : l1
  // CHECK: [[RESULT:%.+]] = moore.conditional [[COND]] : i1 -> l1
  // CHECK: moore.yield [[Z]] : l1
  // CHECK: [[INNERX:%.+]] = moore.constant bX : l1
  // CHECK: [[INNERZ:%.+]] = moore.constant bZ : l1
  // CHECK: [[ISZ:%.+]] = moore.case_eq [[NOT_IN]], [[INNERZ]] : l1
  // CHECK: [[INNERRESULT:%.+]] = moore.conditional [[ISZ]] : i1 -> l1
  // CHECK: moore.yield [[INNERX]] : l1
  // CHECK: moore.yield [[NOT_IN]] : l1
  // CHECK: moore.yield [[INNERRESULT]] : l1
  // CHECK: moore.assign [[DATA_OUT]], [[RESULT]] : ref<l1>
  notif0 notif0_inst(data_out, data_in, en_n);

endmodule

// CHECK-LABEL: moore.module @TestNotif1
module TestNotif1(
  input wire data_in,
  input wire en_n,
  output wire data_out
);
  // CHECK: [[DATA_IN:%.+]] = moore.net name "data_in" wire : <l1>
  // CHECK: [[EN_N:%.+]] = moore.net name "en_n" wire : <l1>
  // CHECK: [[DATA_OUT:%.+]] = moore.net wire : <l1>
  // CHECK: [[IN:%.+]] = moore.read [[DATA_IN]] : ref<l1>
  // CHECK: [[EN:%.+]] = moore.read [[EN_N]] : ref<l1>
  // CHECK: [[NOT_IN:%.+]] = moore.not [[IN]] : l1
  // CHECK: [[INACTIVE:%.+]] = moore.constant 0 : l1
  // CHECK: [[Z:%.+]] = moore.constant bZ : l1
  // CHECK: [[COND:%.+]] = moore.case_eq [[EN]], [[INACTIVE]] : l1
  // CHECK: [[RESULT:%.+]] = moore.conditional [[COND]] : i1 -> l1
  // CHECK: moore.yield [[Z]] : l1
  // CHECK: [[INNERX:%.+]] = moore.constant bX : l1
  // CHECK: [[INNERZ:%.+]] = moore.constant bZ : l1
  // CHECK: [[ISZ:%.+]] = moore.case_eq [[NOT_IN]], [[INNERZ]] : l1
  // CHECK: [[INNERRESULT:%.+]] = moore.conditional [[ISZ]] : i1 -> l1
  // CHECK: moore.yield [[INNERX]] : l1
  // CHECK: moore.yield [[NOT_IN]] : l1
  // CHECK: moore.yield [[INNERRESULT]] : l1
  // CHECK: moore.assign [[DATA_OUT]], [[RESULT]] : ref<l1>
  notif1 notif1_inst(data_out, data_in, en_n);

endmodule

// CHECK-LABEL: moore.module @TestNmos
module TestNmos(input wire data_in, input wire en, output wire data_out);
// CHECK: %[[DATA_IN:.*]] = moore.net name "data_in" wire : <l1>
// CHECK: %[[EN:.*]] = moore.net name "en" wire : <l1>
// CHECK: %[[IN:.*]] = moore.read %[[DATA_IN]] : ref<l1>
// CHECK: %[[ENV:.*]] = moore.read %[[EN]] : ref<l1>
// CHECK: %[[OFF:.*]] = moore.constant 0 : l1
// CHECK: %[[COND:.*]] = moore.case_eq %[[ENV]], %[[OFF]] : l1
// CHECK: %[[Z:.*]] = moore.constant bZ : l1
// CHECK: %[[RESULT:.*]] = moore.conditional %[[COND]] : i1 -> l1
// CHECK: moore.yield %[[Z]] : l1
// CHECK: moore.yield %[[IN]] : l1
nmos n0(data_out, data_in, en);
endmodule

// CHECK-LABEL: moore.module @TestPmos
module TestPmos(input wire data_in, input wire en, output wire data_out);
// CHECK: %[[DATA_IN:.*]] = moore.net name "data_in" wire : <l1>
// CHECK: %[[EN:.*]] = moore.net name "en" wire : <l1>
// CHECK: %[[IN:.*]] = moore.read %[[DATA_IN]] : ref<l1>
// CHECK: %[[ENV:.*]] = moore.read %[[EN]] : ref<l1>
// CHECK: %[[OFF:.*]] = moore.constant 1 : l1
// CHECK: %[[COND:.*]] = moore.case_eq %[[ENV]], %[[OFF]] : l1
// CHECK: %[[Z:.*]] = moore.constant bZ : l1
// CHECK: %[[RESULT:.*]] = moore.conditional %[[COND]] : i1 -> l1
// CHECK: moore.yield %[[Z]] : l1
// CHECK: moore.yield %[[IN]] : l1
pmos p0(data_out, data_in, en);
endmodule

// CHECK-LABEL: moore.module @TestCmos
module TestCmos(input wire data_in, input wire n_en, input wire p_en,
                 output wire data_out);
// CHECK: %[[DATA_IN:.*]] = moore.net name "data_in" wire : <l1>
// CHECK: %[[N_EN:.*]] = moore.net name "n_en" wire : <l1>
// CHECK: %[[P_EN:.*]] = moore.net name "p_en" wire : <l1>
// CHECK: %[[IN:.*]] = moore.read %[[DATA_IN]] : ref<l1>
// CHECK: %[[NEN:.*]] = moore.read %[[N_EN]] : ref<l1>
// CHECK: %[[PEN:.*]] = moore.read %[[P_EN]] : ref<l1>
// CHECK: %[[Z:.*]] = moore.constant bZ : l1
// CHECK: %[[X:.*]] = moore.constant bX : l1
// CHECK: %[[NOFF:.*]] = moore.constant 0 : l1
// CHECK: %[[NCOND:.*]] = moore.case_eq %[[NEN]], %[[NOFF]] : l1
// CHECK: %[[NRESULT:.*]] = moore.conditional %[[NCOND]] : i1 -> l1
// CHECK: moore.yield %[[Z]] : l1
// CHECK: moore.yield %[[IN]] : l1
// CHECK: %[[POFF:.*]] = moore.constant 1 : l1
// CHECK: %[[PCOND:.*]] = moore.case_eq %[[PEN]], %[[POFF]] : l1
// CHECK: %[[PRESULT:.*]] = moore.conditional %[[PCOND]] : i1 -> l1
// CHECK: moore.yield %[[Z]] : l1
// CHECK: moore.yield %[[IN]] : l1
// CHECK: %[[AGREE:.*]] = moore.case_eq %[[NRESULT]], %[[PRESULT]] : l1
// CHECK: %[[NISZ:.*]] = moore.case_eq %[[NRESULT]], %[[Z]] : l1
// CHECK: %[[PISZ:.*]] = moore.case_eq %[[PRESULT]], %[[Z]] : l1
// CHECK: %[[RESULT:.*]] = moore.conditional %[[AGREE]] : i1 -> l1 {
// CHECK: moore.yield %[[NRESULT]] : l1
// CHECK: } {
// CHECK: %[[WHENNISZ:.*]] = moore.conditional %[[NISZ]] : i1 -> l1 {
// CHECK: moore.yield %[[PRESULT]] : l1
// CHECK: } {
// CHECK: %[[WHENDISAGREE:.*]] = moore.conditional %[[PISZ]] : i1 -> l1 {
// CHECK: moore.yield %[[NRESULT]] : l1
// CHECK: } {
// CHECK: moore.yield %[[X]] : l1
// CHECK: }
// CHECK: moore.yield %[[WHENDISAGREE]] : l1
// CHECK: }
// CHECK: moore.yield %[[WHENNISZ]] : l1
// CHECK: }
cmos c0(data_out, data_in, n_en, p_en);
endmodule

// CHECK-LABEL: moore.module @TestRnmos
module TestRnmos(input wire data_in, input wire en, output wire data_out);
// CHECK: %[[DATA_IN:.*]] = moore.net name "data_in" wire : <l1>
// CHECK: %[[EN:.*]] = moore.net name "en" wire : <l1>
// CHECK: %[[IN:.*]] = moore.read %[[DATA_IN]] : ref<l1>
// CHECK: %[[ENV:.*]] = moore.read %[[EN]] : ref<l1>
// CHECK: %[[OFF:.*]] = moore.constant 0 : l1
// CHECK: %[[COND:.*]] = moore.case_eq %[[ENV]], %[[OFF]] : l1
// CHECK: %[[Z:.*]] = moore.constant bZ : l1
// CHECK: %[[RESULT:.*]] = moore.conditional %[[COND]] : i1 -> l1
// CHECK: moore.yield %[[Z]] : l1
// CHECK: moore.yield %[[IN]] : l1
rnmos rn0(data_out, data_in, en);
endmodule

// CHECK-LABEL: moore.module @TestRpmos
module TestRpmos(input wire data_in, input wire en, output wire data_out);
// CHECK: %[[DATA_IN:.*]] = moore.net name "data_in" wire : <l1>
// CHECK: %[[EN:.*]] = moore.net name "en" wire : <l1>
// CHECK: %[[IN:.*]] = moore.read %[[DATA_IN]] : ref<l1>
// CHECK: %[[ENV:.*]] = moore.read %[[EN]] : ref<l1>
// CHECK: %[[OFF:.*]] = moore.constant 1 : l1
// CHECK: %[[COND:.*]] = moore.case_eq %[[ENV]], %[[OFF]] : l1
// CHECK: %[[Z:.*]] = moore.constant bZ : l1
// CHECK: %[[RESULT:.*]] = moore.conditional %[[COND]] : i1 -> l1
// CHECK: moore.yield %[[Z]] : l1
// CHECK: moore.yield %[[IN]] : l1
rpmos rp0(data_out, data_in, en);
endmodule

// CHECK-LABEL: moore.module @TestRcmos
module TestRcmos(input wire data_in, input wire n_en, input wire p_en,
                  output wire data_out);
// CHECK: %[[DATA_IN:.*]] = moore.net name "data_in" wire : <l1>
// CHECK: %[[N_EN:.*]] = moore.net name "n_en" wire : <l1>
// CHECK: %[[P_EN:.*]] = moore.net name "p_en" wire : <l1>
// CHECK: %[[IN:.*]] = moore.read %[[DATA_IN]] : ref<l1>
// CHECK: %[[NEN:.*]] = moore.read %[[N_EN]] : ref<l1>
// CHECK: %[[PEN:.*]] = moore.read %[[P_EN]] : ref<l1>
// CHECK: %[[Z:.*]] = moore.constant bZ : l1
// CHECK: %[[X:.*]] = moore.constant bX : l1
// CHECK: %[[NOFF:.*]] = moore.constant 0 : l1
// CHECK: %[[NCOND:.*]] = moore.case_eq %[[NEN]], %[[NOFF]] : l1
// CHECK: %[[NRESULT:.*]] = moore.conditional %[[NCOND]] : i1 -> l1
// CHECK: moore.yield %[[Z]] : l1
// CHECK: moore.yield %[[IN]] : l1
// CHECK: %[[POFF:.*]] = moore.constant 1 : l1
// CHECK: %[[PCOND:.*]] = moore.case_eq %[[PEN]], %[[POFF]] : l1
// CHECK: %[[PRESULT:.*]] = moore.conditional %[[PCOND]] : i1 -> l1
// CHECK: moore.yield %[[Z]] : l1
// CHECK: moore.yield %[[IN]] : l1
// CHECK: %[[AGREE:.*]] = moore.case_eq %[[NRESULT]], %[[PRESULT]] : l1
// CHECK: %[[NISZ:.*]] = moore.case_eq %[[NRESULT]], %[[Z]] : l1
// CHECK: %[[PISZ:.*]] = moore.case_eq %[[PRESULT]], %[[Z]] : l1
// CHECK: %[[RESULT:.*]] = moore.conditional %[[AGREE]] : i1 -> l1 {
// CHECK: moore.yield %[[NRESULT]] : l1
// CHECK: } {
// CHECK: %[[WHENNISZ:.*]] = moore.conditional %[[NISZ]] : i1 -> l1 {
// CHECK: moore.yield %[[PRESULT]] : l1
// CHECK: } {
// CHECK: %[[WHENDISAGREE:.*]] = moore.conditional %[[PISZ]] : i1 -> l1 {
// CHECK: moore.yield %[[NRESULT]] : l1
// CHECK: } {
// CHECK: moore.yield %[[X]] : l1
// CHECK: }
// CHECK: moore.yield %[[WHENDISAGREE]] : l1
// CHECK: }
// CHECK: moore.yield %[[WHENNISZ]] : l1
// CHECK: }
rcmos c0(data_out, data_in, n_en, p_en);
endmodule

// Ensure Slang automatically inserts conversions for MOS switch inputs that
// aren't 1-bit
// CHECK-LABEL: moore.module @TestNmosWideInputs
module TestNmosWideInputs(input wire [1:0] data_in, input wire [1:0] en,
                          output wire data_out);
// CHECK: %[[DATA_IN:.*]] = moore.net name "data_in" wire : <l2>
// CHECK: %[[EN:.*]] = moore.net name "en" wire : <l2>
// CHECK: %[[INW:.*]] = moore.read %[[DATA_IN]] : ref<l2>
// CHECK: %[[IN:.*]] = moore.trunc %[[INW]] : l2 -> l1
// CHECK: %[[ENW:.*]] = moore.read %[[EN]] : ref<l2>
// CHECK: %[[ENV:.*]] = moore.trunc %[[ENW]] : l2 -> l1
// CHECK: %[[OFF:.*]] = moore.constant 0 : l1
// CHECK: %[[COND:.*]] = moore.case_eq %[[ENV]], %[[OFF]] : l1
// CHECK: %[[Z:.*]] = moore.constant bZ : l1
// CHECK: %[[RESULT:.*]] = moore.conditional %[[COND]] : i1 -> l1
// CHECK: moore.yield %[[Z]] : l1
// CHECK: moore.yield %[[IN]] : l1
nmos n0(data_out, data_in, en);
endmodule

// CHECK-LABEL: moore.module @TestCmosWideInputs
module TestCmosWideInputs(input wire [1:0] data_in, input wire [1:0] n_en,
                          input wire [1:0] p_en, output wire data_out);
// CHECK: %[[DATA_IN:.*]] = moore.net name "data_in" wire : <l2>
// CHECK: %[[N_EN:.*]] = moore.net name "n_en" wire : <l2>
// CHECK: %[[P_EN:.*]] = moore.net name "p_en" wire : <l2>
// CHECK: %[[INW:.*]] = moore.read %[[DATA_IN]] : ref<l2>
// CHECK: %[[IN:.*]] = moore.trunc %[[INW]] : l2 -> l1
// CHECK: %[[NENW:.*]] = moore.read %[[N_EN]] : ref<l2>
// CHECK: %[[NEN:.*]] = moore.trunc %[[NENW]] : l2 -> l1
// CHECK: %[[PENW:.*]] = moore.read %[[P_EN]] : ref<l2>
// CHECK: %[[PEN:.*]] = moore.trunc %[[PENW]] : l2 -> l1
// CHECK: %[[Z:.*]] = moore.constant bZ : l1
// CHECK: %[[NOFF:.*]] = moore.constant 0 : l1
// CHECK: %[[NCOND:.*]] = moore.case_eq %[[NEN]], %[[NOFF]] : l1
// CHECK: moore.conditional %[[NCOND]] : i1 -> l1
// CHECK: moore.yield %[[Z]] : l1
// CHECK: moore.yield %[[IN]] : l1
// CHECK: %[[POFF:.*]] = moore.constant 1 : l1
// CHECK: %[[PCOND:.*]] = moore.case_eq %[[PEN]], %[[POFF]] : l1
// CHECK: moore.conditional %[[PCOND]] : i1 -> l1
// CHECK: moore.yield %[[Z]] : l1
// CHECK: moore.yield %[[IN]] : l1
cmos c0(data_out, data_in, n_en, p_en);
endmodule

// CHECK-LABEL: moore.module @TestDelayedPmos
module TestDelayedPmos(input wire data_in, input wire en, output wire data_out);
// CHECK: %[[DATA_OUT:.*]] = moore.net wire : <l1>
// CHECK: %[[RESULT:.*]] = moore.conditional
// CHECK: %[[DELAY:.*]] = moore.constant_time 5000000 fs
// CHECK: moore.delayed_assign %[[DATA_OUT]], %[[RESULT]], %[[DELAY]] : ref<l1>
pmos #5 p0(data_out, data_in, en);
endmodule

// CHECK-LABEL: moore.module @TestDelayedCmos
module TestDelayedCmos(input wire data_in, input wire n_en, input wire p_en,
                       output wire data_out);
// CHECK: %[[DATA_OUT:.*]] = moore.net wire : <l1>
// CHECK: %[[NRESULT:.*]] = moore.conditional
// CHECK: %[[PRESULT:.*]] = moore.conditional
// CHECK: %[[AGREE:.*]] = moore.case_eq %[[NRESULT]], %[[PRESULT]] : l1
// CHECK: %[[RESULT:.*]] = moore.conditional %[[AGREE]] : i1 -> l1
// CHECK: %[[DELAY:.*]] = moore.constant_time 5000000 fs
// CHECK: moore.delayed_assign %[[DATA_OUT]], %[[RESULT]], %[[DELAY]] : ref<l1>
cmos #5 c0(data_out, data_in, n_en, p_en);
endmodule

// CHECK-LABEL: moore.module @TestPmosArrayConcatLiteral
module TestPmosArrayConcatLiteral;
  wire [1:0] ad;
  reg regff;
  reg en;

  // Regression test: bit-blasting an array primitive instantiation whose
  // data connection is a concatenation containing a two-state literal
  // (1'b0) used to leave that bit's extracted value two-state (i1),
  // causing a 'moore.yield' type mismatch against the (4-state)
  // primitive output.
  // CHECK: %ad = moore.net wire : <l2>
  // CHECK: %regff = moore.variable : <l1>
  // CHECK: %en = moore.variable : <l1>
  // CHECK: %[[REGFFREAD:.*]] = moore.read %regff : ref<l1>
  // CHECK: %[[ENREAD0:.*]] = moore.read %en : ref<l1>
  // CHECK: %[[OFF0:.*]] = moore.constant 1 : l1
  // CHECK: %[[COND0:.*]] = moore.case_eq %[[ENREAD0]], %[[OFF0]] : l1
  // CHECK: %[[Z0:.*]] = moore.constant bZ : l1
  // CHECK: moore.conditional %[[COND0]] : i1 -> l1 {
  // CHECK:   moore.yield %[[Z0]] : l1
  // CHECK: } {
  // CHECK:   moore.yield %[[REGFFREAD]] : l1
  // CHECK: }
  // CHECK: %[[ENREAD1:.*]] = moore.read %en : ref<l1>
  // CHECK: %[[OFF1:.*]] = moore.constant 1 : l1
  // CHECK: %[[COND1:.*]] = moore.case_eq %[[ENREAD1]], %[[OFF1]] : l1
  // CHECK: %[[ZEROBIT:.*]] = moore.constant 0 : l1
  // CHECK: %[[Z1:.*]] = moore.constant bZ : l1
  // CHECK: moore.conditional %[[COND1]] : i1 -> l1 {
  // CHECK:   moore.yield %[[Z1]] : l1
  // CHECK: } {
  // CHECK:   moore.yield %[[ZEROBIT]] : l1
  // CHECK: }
  pmos ad_drv[1:0] (ad, {1'b0, regff}, en);
endmodule

// Ensure Slang automatically inserts conversions for enable signals that aren't 1-bit
// CHECK-LABEL: moore.module @TestBufif1WideEnable
module TestBufif1WideEnable(
  input wire data_in,
  input wire [1:0] en,
  output wire data_out
);
  // CHECK: [[DATA_IN:%.+]] = moore.net name "data_in" wire : <l1>
  // CHECK: [[EN_W:%.+]] = moore.net name "en" wire : <l2>
  // CHECK: [[DATA_OUT:%.+]] = moore.net wire : <l1>
  // CHECK: [[IN:%.+]] = moore.read [[DATA_IN]] : ref<l1>
  // CHECK: [[EN_READ:%.+]] = moore.read [[EN_W]] : ref<l2>
  // CHECK: [[EN:%.+]] = moore.trunc [[EN_READ]] : l2 -> l1
  // CHECK: [[INACTIVE:%.+]] = moore.constant 0 : l1
  // CHECK: [[Z:%.+]] = moore.constant bZ : l1
  // CHECK: [[COND:%.+]] = moore.case_eq [[EN]], [[INACTIVE]] : l1
  // CHECK: [[RESULT:%.+]] = moore.conditional [[COND]] : i1 -> l1
  // CHECK: moore.yield [[Z]] : l1
  // CHECK: [[INNERX:%.+]] = moore.constant bX : l1
  // CHECK: [[INNERZ:%.+]] = moore.constant bZ : l1
  // CHECK: [[ISZ:%.+]] = moore.case_eq [[IN]], [[INNERZ]] : l1
  // CHECK: [[INNERRESULT:%.+]] = moore.conditional [[ISZ]] : i1 -> l1
  // CHECK: moore.yield [[INNERX]] : l1
  // CHECK: moore.yield [[IN]] : l1
  // CHECK: moore.yield [[INNERRESULT]] : l1
  // CHECK: moore.assign [[DATA_OUT]], [[RESULT]] : ref<l1>
  bufif1 bufif1_inst(data_out, data_in, en);
endmodule

// CHECK-LABEL: moore.module @udp_comb_and(out out : !moore.l1, in %a : !moore.l1, in %b : !moore.l1)
// CHECK: [[AND:%.+]] = moore.and %a, %b : l1
// CHECK: moore.output [[AND]] : !moore.l1
primitive udp_comb_and(out, a, b);
    output out;
    input a, b;
    table
        1 1 : 1 ;
        0 ? : 0 ;
        ? 0 : 0 ;
    endtable
endprimitive

// CHECK-LABEL: moore.module @udp_comb_prim()
// CHECK: [[A:%.+]] = moore.net wire : <l1>
// CHECK: [[B:%.+]] = moore.net wire : <l1>
// CHECK: [[Q:%.+]] = moore.net wire : <l1>
// CHECK: [[RD_A:%.+]] = moore.read [[A]] : ref<l1>
// CHECK: [[RD_B:%.+]] = moore.read [[B]] : ref<l1>
// CHECK: [[INST:%.+]] = moore.instance "u1" @udp_comb_and(a: [[RD_A]]: !moore.l1, b: [[RD_B]]: !moore.l1) -> (out: !moore.l1)
// CHECK: moore.assign [[Q]], [[INST]] : ref<l1>
module udp_comb_prim;
    wire A, B, Q;
    udp_comb_and u1 (Q, A, B);
endmodule

// CHECK-LABEL: moore.module @udp_comb_delayed_prim()
// CHECK: [[A:%.+]] = moore.net wire : <l1>
// CHECK: [[B:%.+]] = moore.net wire : <l1>
// CHECK: [[Q:%.+]] = moore.net wire : <l1>
// CHECK: [[RD_A:%.+]] = moore.read [[A]] : ref<l1>
// CHECK: [[RD_B:%.+]] = moore.read [[B]] : ref<l1>
// CHECK: [[INST:%.+]] = moore.instance "u1" @udp_comb_and(a: [[RD_A]]: !moore.l1, b: [[RD_B]]: !moore.l1) -> (out: !moore.l1)
// CHECK: [[DELAYCONST:%.+]] = moore.constant_time 5000000 fs
// CHECK: moore.delayed_assign [[Q]], [[INST]], [[DELAYCONST]] : ref<l1>
module udp_comb_delayed_prim;
    wire A, B, Q;
    udp_comb_and #5 u1 (Q, A, B);
endmodule

//===----------------------------------------------------------------------===//
// Sequential UDP Primitives
//===----------------------------------------------------------------------===//

// CHECK-LABEL: moore.module @udp_seq_dff(out q : !moore.l1, in %clk : !moore.l1, in %d : !moore.l1)
// CHECK: [[Q_VAR:%.+]] = moore.variable : <i1>
// CHECK: [[CLK_CVT:%.+]] = moore.logic_to_int %clk : l1
// CHECK: [[D_CVT:%.+]] = moore.logic_to_int %d : l1
// CHECK: [[CLK_NET:%.+]] = moore.net wire [[CLK_CVT]] : <i1>
// CHECK: moore.procedure always {
// CHECK:   moore.wait_event {
// CHECK:     [[CLK_RD:%.+]] = moore.read [[CLK_NET]] : {{ref<i1>|<i1>}}
// CHECK:     moore.detect_event posedge [[CLK_RD]] : i1
// CHECK:   }
// CHECK:   moore.nonblocking_assign [[Q_VAR]], [[D_CVT]] : i1
// CHECK:   moore.return
// CHECK: }
// CHECK: [[Q_RD:%.+]] = moore.read [[Q_VAR]] : {{ref<i1>|<i1>}}
// CHECK: [[Q_LOGIC:%.+]] = moore.int_to_logic [[Q_RD]] : i1
// CHECK: moore.output [[Q_LOGIC]] : !moore.l1
primitive udp_seq_dff(q, clk, d);
    output reg q;
    input clk, d;
    table
        (01) 0 : ? : 0 ;
        (01) 1 : ? : 1 ;
        (10) ? : ? : - ;
        ?   (??): ? : - ;
    endtable
endprimitive

// CHECK-LABEL: moore.module @udp_seq_dff_neg(out q : !moore.l1, in %clk : !moore.l1, in %d : !moore.l1)
// CHECK: [[Q_VAR:%.+]] = moore.variable : <i1>
// CHECK: [[CLK_CVT:%.+]] = moore.logic_to_int %clk : l1
// CHECK: [[D_CVT:%.+]] = moore.logic_to_int %d : l1
// CHECK: [[CLK_NET:%.+]] = moore.net wire [[CLK_CVT]] : <i1>
// CHECK: moore.procedure always {
// CHECK:   moore.wait_event {
// CHECK:     [[CLK_RD:%.+]] = moore.read [[CLK_NET]] : {{ref<i1>|<i1>}}
// CHECK:     moore.detect_event negedge [[CLK_RD]] : i1
// CHECK:   }
// CHECK:   moore.nonblocking_assign [[Q_VAR]], [[D_CVT]] : i1
// CHECK:   moore.return
// CHECK: }
// CHECK: [[Q_RD:%.+]] = moore.read [[Q_VAR]] : {{ref<i1>|<i1>}}
// CHECK: [[Q_LOGIC:%.+]] = moore.int_to_logic [[Q_RD]] : i1
// CHECK: moore.output [[Q_LOGIC]] : !moore.l1
primitive udp_seq_dff_neg(q, clk, d);
    output reg q;
    input clk, d;
    table
        (10) 0 : ? : 0 ;
        (10) 1 : ? : 1 ;
        (01) ? : ? : - ;
        ?   (??): ? : - ;
    endtable
endprimitive

// CHECK-LABEL: moore.module @udp_seq_dff_en(out q : !moore.l1, in %clk : !moore.l1, in %en : !moore.l1, in %d : !moore.l1)
// CHECK: [[Q_VAR:%.+]] = moore.variable : <i1>
// CHECK: [[CLK_CVT:%.+]] = moore.logic_to_int %clk : l1
// CHECK: [[EN_CVT:%.+]] = moore.logic_to_int %en : l1
// CHECK: [[D_CVT:%.+]] = moore.logic_to_int %d : l1
// CHECK: [[CLK_NET:%.+]] = moore.net wire [[CLK_CVT]] : <i1>
// CHECK: moore.procedure always {
// CHECK:   moore.wait_event {
// CHECK:     [[CLK_RD:%.+]] = moore.read [[CLK_NET]] : {{ref<i1>|<i1>}}
// CHECK:     moore.detect_event posedge [[CLK_RD]] : i1
// CHECK:   }
// CHECK:   [[COND:%.+]] = moore.to_builtin_int [[EN_CVT]] : i1
// CHECK:   cf.cond_br [[COND]], ^[[TRUE_BB:.+]], ^[[FALSE_BB:.+]]
// CHECK: ^[[TRUE_BB]]:
// CHECK:   moore.nonblocking_assign [[Q_VAR]], [[D_CVT]] : i1
// CHECK:   cf.br ^[[FALSE_BB]]
// CHECK: ^[[FALSE_BB]]:
// CHECK:   moore.return
// CHECK: }
// CHECK: [[Q_RD:%.+]] = moore.read [[Q_VAR]] : {{ref<i1>|<i1>}}
// CHECK: [[Q_LOGIC:%.+]] = moore.int_to_logic [[Q_RD]] : i1
// CHECK: moore.output [[Q_LOGIC]] : !moore.l1
primitive udp_seq_dff_en(q, clk, en, d);
    output reg q;
    input clk, en, d;
    table
        (01) 1 0 : ? : 0 ;
        (01) 1 1 : ? : 1 ;
        (01) 0 ? : ? : - ;
        (10) ? ? : ? : - ;
        ?    ? (??): ? : - ;
    endtable
endprimitive

// Verify that benign hold rows and inactive clock edges in an edge-triggered
// UDP are safely skipped.
// CHECK-LABEL: moore.module @udp_seq_dff_hold(out q : !moore.l1, in %clk : !moore.l1, in %d : !moore.l1)
// CHECK: [[Q_VAR:%.+]] = moore.variable : <i1>
// CHECK: [[CLK_CVT:%.+]] = moore.logic_to_int %clk : l1
// CHECK: [[D_CVT:%.+]] = moore.logic_to_int %d : l1
// CHECK: [[CLK_NET:%.+]] = moore.net wire [[CLK_CVT]] : <i1>
// CHECK: moore.procedure always {
// CHECK:   moore.wait_event {
// CHECK:     [[CLK_RD:%.+]] = moore.read [[CLK_NET]] : {{ref<i1>|<i1>}}
// CHECK:     moore.detect_event posedge [[CLK_RD]] : i1
// CHECK:   }
// CHECK:   moore.nonblocking_assign [[Q_VAR]], [[D_CVT]] : i1
// CHECK:   moore.return
// CHECK: }
// CHECK: [[Q_RD:%.+]] = moore.read [[Q_VAR]] : {{ref<i1>|<i1>}}
// CHECK: [[Q_LOGIC:%.+]] = moore.int_to_logic [[Q_RD]] : i1
// CHECK: moore.output [[Q_LOGIC]] : !moore.l1
primitive udp_seq_dff_hold(q, clk, d);
    output reg q;
    input clk, d;
    table
        (01) 0 : ? : 0 ;
        (01) 1 : ? : 1 ;
        (10) ? : ? : - ; // Skipped: inactive clock edge (output is '-')
        ?    0 : 0 : 0 ; // Skipped: steady-state hold (state == output)
        ?    1 : 1 : 1 ; // Skipped: steady-state hold (state == output)
    endtable
endprimitive

// CHECK-LABEL: moore.module @udp_seq_dual_edge_dff(out q : !moore.l1, in %clk : !moore.l1, in %d : !moore.l1)
// CHECK: [[Q_POS_VAR:%.+]] = moore.variable : <i1>
// CHECK: [[Q_NEG_VAR:%.+]] = moore.variable : <i1>
// CHECK: [[CLK_CVT:%.+]] = moore.logic_to_int %clk : l1
// CHECK: [[D_CVT:%.+]] = moore.logic_to_int %d : l1
// CHECK: [[CLK_NET:%.+]] = moore.net wire [[CLK_CVT]] : <i1>
// CHECK: moore.procedure always {
// CHECK:   moore.wait_event {
// CHECK:     [[CLK_RD:%.+]] = moore.read [[CLK_NET]] : {{ref<i1>|<i1>}}
// CHECK:     moore.detect_event posedge [[CLK_RD]] : i1
// CHECK:   }
// CHECK:   moore.nonblocking_assign [[Q_POS_VAR]], [[D_CVT]] : i1
// CHECK:   moore.return
// CHECK: }
// CHECK: [[CLK_NET2:%.+]] = moore.net wire [[CLK_CVT]] : <i1>
// CHECK: moore.procedure always {
// CHECK:   moore.wait_event {
// CHECK:     [[CLK_RD2:%.+]] = moore.read [[CLK_NET2]] : {{ref<i1>|<i1>}}
// CHECK:     moore.detect_event negedge [[CLK_RD2]] : i1
// CHECK:   }
// CHECK:   moore.nonblocking_assign [[Q_NEG_VAR]], [[D_CVT]] : i1
// CHECK:   moore.return
// CHECK: }
// CHECK: [[Q_POS_RD:%.+]] = moore.read [[Q_POS_VAR]] : {{ref<i1>|<i1>}}
// CHECK: [[Q_NEG_RD:%.+]] = moore.read [[Q_NEG_VAR]] : {{ref<i1>|<i1>}}
// CHECK: [[TERM_POS:%.+]] = moore.and [[CLK_CVT]], [[Q_POS_RD]] : i1
// CHECK: [[NOT_CLK:%.+]] = moore.not [[CLK_CVT]] : i1
// CHECK: [[TERM_NEG:%.+]] = moore.and [[NOT_CLK]], [[Q_NEG_RD]] : i1
// CHECK: [[OR:%.+]] = moore.or [[TERM_POS]], [[TERM_NEG]] : i1
// CHECK: [[Q_LOGIC:%.+]] = moore.int_to_logic [[OR]] : i1
// CHECK: moore.output [[Q_LOGIC]] : !moore.l1
primitive udp_seq_dual_edge_dff(q, clk, d);
    output reg q;
    input clk, d;
    table
        (01) 0 : ? : 0 ;
        (01) 1 : ? : 1 ;
        (10) 0 : ? : 0 ;
        (10) 1 : ? : 1 ;
    endtable
endprimitive

// CHECK-LABEL: moore.module @udp_seq_dual_edge_mux(out q : !moore.l1, in %clk : !moore.l1, in %da : !moore.l1, in %db : !moore.l1)
// CHECK: [[Q_POS_VAR:%.+]] = moore.variable : <i1>
// CHECK: [[Q_NEG_VAR:%.+]] = moore.variable : <i1>
// CHECK: [[CLK_CVT:%.+]] = moore.logic_to_int %clk : l1
// CHECK: [[DA_CVT:%.+]] = moore.logic_to_int %da : l1
// CHECK: [[DB_CVT:%.+]] = moore.logic_to_int %db : l1
// CHECK: [[CLK_NET:%.+]] = moore.net wire [[CLK_CVT]] : <i1>
// CHECK: moore.procedure always {
// CHECK:   moore.wait_event {
// CHECK:     [[CLK_RD:%.+]] = moore.read [[CLK_NET]] : {{ref<i1>|<i1>}}
// CHECK:     moore.detect_event posedge [[CLK_RD]] : i1
// CHECK:   }
// CHECK:   moore.nonblocking_assign [[Q_POS_VAR]], [[DA_CVT]] : i1
// CHECK:   moore.return
// CHECK: }
// CHECK: [[CLK_NET2:%.+]] = moore.net wire [[CLK_CVT]] : <i1>
// CHECK: moore.procedure always {
// CHECK:   moore.wait_event {
// CHECK:     [[CLK_RD2:%.+]] = moore.read [[CLK_NET2]] : {{ref<i1>|<i1>}}
// CHECK:     moore.detect_event negedge [[CLK_RD2]] : i1
// CHECK:   }
// CHECK:   moore.nonblocking_assign [[Q_NEG_VAR]], [[DB_CVT]] : i1
// CHECK:   moore.return
// CHECK: }
// CHECK: [[Q_POS_RD:%.+]] = moore.read [[Q_POS_VAR]] : {{ref<i1>|<i1>}}
// CHECK: [[Q_NEG_RD:%.+]] = moore.read [[Q_NEG_VAR]] : {{ref<i1>|<i1>}}
// CHECK: [[TERM_POS:%.+]] = moore.and [[CLK_CVT]], [[Q_POS_RD]] : i1
// CHECK: [[NOT_CLK:%.+]] = moore.not [[CLK_CVT]] : i1
// CHECK: [[TERM_NEG:%.+]] = moore.and [[NOT_CLK]], [[Q_NEG_RD]] : i1
// CHECK: [[OR:%.+]] = moore.or [[TERM_POS]], [[TERM_NEG]] : i1
// CHECK: [[Q_LOGIC:%.+]] = moore.int_to_logic [[OR]] : i1
// CHECK: moore.output [[Q_LOGIC]] : !moore.l1
primitive udp_seq_dual_edge_mux(q, clk, da, db);
    output reg q;
    input clk, da, db;
    table
        (01) 0 ? : ? : 0 ;
        (01) 1 ? : ? : 1 ;
        (10) ? 0 : ? : 0 ;
        (10) ? 1 : ? : 1 ;
    endtable
endprimitive

// CHECK-LABEL: moore.module @udp_seq_asym_cond(out q : !moore.l1, in %clk : !moore.l1, in %clr_n : !moore.l1, in %pre_n : !moore.l1, in %d : !moore.l1, in %n : !moore.l1)
// CHECK: [[Q_VAR:%.+]] = moore.variable : <i1>
// CHECK: [[CLK_CVT:%.+]] = moore.logic_to_int %clk : l1
// CHECK: [[CLR_CVT:%.+]] = moore.logic_to_int %clr_n : l1
// CHECK: [[PRE_CVT:%.+]] = moore.logic_to_int %pre_n : l1
// CHECK: [[D_CVT:%.+]] = moore.logic_to_int %d : l1
// CHECK: [[N_CVT:%.+]] = moore.logic_to_int %n : l1
// CHECK: [[Q_HOLD:%.+]] = moore.read [[Q_VAR]] : {{ref<i1>|<i1>}}
// CHECK: [[C1:%.+]] = moore.constant 1 : i1
// CHECK: [[SET_COND:%.+]] = moore.and [[CLR_CVT]], [[D_CVT]] : i1
// CHECK: [[MUX1:%.+]] = moore.conditional [[SET_COND]] : i1 -> i1 {
// CHECK:   moore.yield [[C1]] : i1
// CHECK: } {
// CHECK:   moore.yield [[Q_HOLD]] : i1
// CHECK: }
// CHECK: [[C0:%.+]] = moore.constant 0 : i1
// CHECK: [[NOT_D:%.+]] = moore.not [[D_CVT]] : i1
// CHECK: [[CLR_COND:%.+]] = moore.and [[PRE_CVT]], [[NOT_D]] : i1
// CHECK: [[NEXT:%.+]] = moore.conditional [[CLR_COND]] : i1 -> i1 {
// CHECK:   moore.yield [[C0]] : i1
// CHECK: } {
// CHECK:   moore.yield [[MUX1]] : i1
// CHECK: }
// CHECK: [[CLK_NET:%.+]] = moore.net wire [[CLK_CVT]] : <i1>
// CHECK: moore.procedure always {
// CHECK:   moore.wait_event {
// CHECK:     [[CLK_RD:%.+]] = moore.read [[CLK_NET]] : {{ref<i1>|<i1>}}
// CHECK:     moore.detect_event posedge [[CLK_RD]] : i1
// CHECK:   }
// CHECK:   moore.nonblocking_assign [[Q_VAR]], [[NEXT]] : i1
// CHECK:   moore.return
// CHECK: }
// CHECK: [[Q_RD:%.+]] = moore.read [[Q_VAR]] : {{ref<i1>|<i1>}}
// CHECK: [[Q_LOGIC:%.+]] = moore.int_to_logic [[Q_RD]] : i1
// CHECK: moore.output [[Q_LOGIC]] : !moore.l1
primitive udp_seq_asym_cond(q, clk, clr_n, pre_n, d, n);
    output reg q;
    input clk, clr_n, pre_n, d, n;
    table
        (01) ? 1 0 ? : ? : 0 ;
        (01) 1 ? 1 ? : ? : 1 ;
    endtable
endprimitive

// CHECK-LABEL: moore.module @udp_seq_dff_rst(out q : !moore.l1, in %clk : !moore.l1, in %rst : !moore.l1, in %d : !moore.l1)
// CHECK: [[Q_VAR:%.+]] = moore.variable : <i1>
// CHECK: [[CLK_CVT:%.+]] = moore.logic_to_int %clk : l1
// CHECK: [[RST_CVT:%.+]] = moore.logic_to_int %rst : l1
// CHECK: [[D_CVT:%.+]] = moore.logic_to_int %d : l1
// CHECK: [[CLK_NET:%.+]] = moore.net wire [[CLK_CVT]] : <i1>
// CHECK: [[RST_NET:%.+]] = moore.net wire [[RST_CVT]] : <i1>
// CHECK: moore.procedure always {
// CHECK:   moore.wait_event {
// CHECK:     [[CLK_RD:%.+]] = moore.read [[CLK_NET]] : {{ref<i1>|<i1>}}
// CHECK:     moore.detect_event posedge [[CLK_RD]] : i1
// CHECK:     [[RST_RD:%.+]] = moore.read [[RST_NET]] : {{ref<i1>|<i1>}}
// CHECK:     moore.detect_event posedge [[RST_RD]] : i1
// CHECK:   }
// CHECK:   [[RST_COND:%.+]] = moore.read [[RST_NET]] : {{ref<i1>|<i1>}}
// CHECK:   [[RST_I1:%.+]] = moore.to_builtin_int [[RST_COND]] : i1
// CHECK:   cf.cond_br [[RST_I1]], ^[[TRUE_BB:.+]], ^[[FALSE_BB:.+]]
// CHECK: ^[[TRUE_BB]]:
// CHECK:   moore.nonblocking_assign [[Q_VAR]], {{%.+}} : i1
// CHECK:   cf.br ^[[EXIT_BB:.+]]
// CHECK: ^[[FALSE_BB]]:
// CHECK:   moore.nonblocking_assign [[Q_VAR]], [[D_CVT]] : i1
// CHECK:   cf.br ^[[EXIT_BB]]
// CHECK: ^[[EXIT_BB]]:
// CHECK:   moore.return
// CHECK: }
// CHECK: [[Q_RD:%.+]] = moore.read [[Q_VAR]] : {{ref<i1>|<i1>}}
// CHECK: [[Q_LOGIC:%.+]] = moore.int_to_logic [[Q_RD]] : i1
// CHECK: moore.output [[Q_LOGIC]] : !moore.l1
primitive udp_seq_dff_rst(q, clk, rst, d);
    output reg q;
    input clk, rst, d;
    table
        ?    1 ? : ? : 0 ;
        (01) 0 0 : ? : 0 ;
        (01) 0 1 : ? : 1 ;
        (10) ? ? : ? : - ;
        ?   (10) ? : ? : - ;
    endtable
endprimitive

// CHECK-LABEL: moore.module @udp_seq_dff_sr(out q : !moore.l1, in %clk : !moore.l1, in %rst : !moore.l1, in %set : !moore.l1, in %d : !moore.l1)
// CHECK: [[Q_VAR:%.+]] = moore.variable : <i1>
// CHECK: [[CLK_CVT:%.+]] = moore.logic_to_int %clk : l1
// CHECK: [[RST_CVT:%.+]] = moore.logic_to_int %rst : l1
// CHECK: [[SET_CVT:%.+]] = moore.logic_to_int %set : l1
// CHECK: [[D_CVT:%.+]] = moore.logic_to_int %d : l1
// CHECK: [[NOT_RST:%.+]] = moore.not [[RST_CVT]] : i1
// CHECK: [[SET_TERM:%.+]] = moore.and [[NOT_RST]], [[SET_CVT]] : i1
// CHECK: [[SYNC_C0:%.+]] = moore.constant 0 : i1
// CHECK: [[SYNC_C1:%.+]] = moore.constant 1 : i1
// CHECK: [[SYNC_INNER:%.+]] = moore.conditional [[SET_TERM]] : i1 -> i1 {
// CHECK:   moore.yield [[SYNC_C1]] : i1
// CHECK: } {
// CHECK:   moore.yield [[D_CVT]] : i1
// CHECK: }
// CHECK: [[SYNC_VAL:%.+]] = moore.conditional [[RST_CVT]] : i1 -> i1 {
// CHECK:   moore.yield [[SYNC_C0]] : i1
// CHECK: } {
// CHECK:   moore.yield [[SYNC_INNER]] : i1
// CHECK: }
// CHECK: [[CLK_NET:%.+]] = moore.net wire [[CLK_CVT]] : <i1>
// CHECK: moore.procedure always {
// CHECK:   moore.wait_event {
// CHECK:     [[CLK_RD:%.+]] = moore.read [[CLK_NET]] : {{ref<i1>|<i1>}}
// CHECK:     moore.detect_event posedge [[CLK_RD]] : i1
// CHECK:   }
// CHECK:   moore.nonblocking_assign [[Q_VAR]], [[SYNC_VAL]] : i1
// CHECK:   moore.return
// CHECK: }
// CHECK: [[Q_RD:%.+]] = moore.read [[Q_VAR]] : {{ref<i1>|<i1>}}
// CHECK: [[OUT_C0:%.+]] = moore.constant 0 : i1
// CHECK: [[OUT_C1:%.+]] = moore.constant 1 : i1
// CHECK: [[OUT_INNER:%.+]] = moore.conditional [[SET_TERM]] : i1 -> i1 {
// CHECK:   moore.yield [[OUT_C1]] : i1
// CHECK: } {
// CHECK:   moore.yield [[Q_RD]] : i1
// CHECK: }
// CHECK: [[OUT_VAL:%.+]] = moore.conditional [[RST_CVT]] : i1 -> i1 {
// CHECK:   moore.yield [[OUT_C0]] : i1
// CHECK: } {
// CHECK:   moore.yield [[OUT_INNER]] : i1
// CHECK: }
// CHECK: [[Q_LOGIC:%.+]] = moore.int_to_logic [[OUT_VAL]] : i1
// CHECK: moore.output [[Q_LOGIC]] : !moore.l1
primitive udp_seq_dff_sr(q, clk, rst, set, d);
    output reg q;
    input clk, rst, set, d;
    table
        ?    1 ? ? : ? : 0 ;
        ?    0 1 ? : ? : 1 ;
        (01) 0 0 0 : ? : 0 ;
        (01) 0 0 1 : ? : 1 ;
        (10) ? ? ? : ? : - ;
    endtable
endprimitive

// CHECK-LABEL: moore.module @udp_seq_dff_clear_preset(out q : !moore.l1, in %clk : !moore.l1, in %clr_n : !moore.l1, in %pre_n : !moore.l1, in %d : !moore.l1)
// CHECK: [[Q_VAR:%.+]] = moore.variable : <i1>
// CHECK: [[CLK_CVT:%.+]] = moore.logic_to_int %clk : l1
// CHECK: [[CLR_CVT:%.+]] = moore.logic_to_int %clr_n : l1
// CHECK: [[PRE_CVT:%.+]] = moore.logic_to_int %pre_n : l1
// CHECK: [[D_CVT:%.+]] = moore.logic_to_int %d : l1
// CHECK: [[NOT_CLR:%.+]] = moore.not [[CLR_CVT]] : i1
// CHECK: [[NOT_PRE:%.+]] = moore.not [[PRE_CVT]] : i1
// CHECK: [[SET_TERM:%.+]] = moore.and [[CLR_CVT]], [[NOT_PRE]] : i1
// CHECK: [[SYNC_C0:%.+]] = moore.constant 0 : i1
// CHECK: [[SYNC_C1:%.+]] = moore.constant 1 : i1
// CHECK: [[SYNC_INNER:%.+]] = moore.conditional [[SET_TERM]] : i1 -> i1 {
// CHECK:   moore.yield [[SYNC_C1]] : i1
// CHECK: } {
// CHECK:   moore.yield [[D_CVT]] : i1
// CHECK: }
// CHECK: [[SYNC_VAL:%.+]] = moore.conditional [[NOT_CLR]] : i1 -> i1 {
// CHECK:   moore.yield [[SYNC_C0]] : i1
// CHECK: } {
// CHECK:   moore.yield [[SYNC_INNER]] : i1
// CHECK: }
// CHECK: [[CLK_NET:%.+]] = moore.net wire [[CLK_CVT]] : <i1>
// CHECK: moore.procedure always {
// CHECK:   moore.wait_event {
// CHECK:     [[CLK_RD:%.+]] = moore.read [[CLK_NET]] : {{ref<i1>|<i1>}}
// CHECK:     moore.detect_event posedge [[CLK_RD]] : i1
// CHECK:   }
// CHECK:   moore.nonblocking_assign [[Q_VAR]], [[SYNC_VAL]] : i1
// CHECK:   moore.return
// CHECK: }
// CHECK: [[Q_RD:%.+]] = moore.read [[Q_VAR]] : {{ref<i1>|<i1>}}
// CHECK: [[OUT_C0:%.+]] = moore.constant 0 : i1
// CHECK: [[OUT_C1:%.+]] = moore.constant 1 : i1
// CHECK: [[OUT_INNER:%.+]] = moore.conditional [[SET_TERM]] : i1 -> i1 {
// CHECK:   moore.yield [[OUT_C1]] : i1
// CHECK: } {
// CHECK:   moore.yield [[Q_RD]] : i1
// CHECK: }
// CHECK: [[OUT_VAL:%.+]] = moore.conditional [[NOT_CLR]] : i1 -> i1 {
// CHECK:   moore.yield [[OUT_C0]] : i1
// CHECK: } {
// CHECK:   moore.yield [[OUT_INNER]] : i1
// CHECK: }
// CHECK: [[Q_LOGIC:%.+]] = moore.int_to_logic [[OUT_VAL]] : i1
// CHECK: moore.output [[Q_LOGIC]] : !moore.l1
primitive udp_seq_dff_clear_preset (q, clk, clr_n, pre_n, d);
   output reg q; input clk, clr_n, pre_n, d;
   table
      ?    0 ? ? : ? : 0 ;
      ?    1 0 ? : ? : 1 ;
      (01) 1 1 0 : ? : 0 ;
      (01) 1 1 1 : ? : 1 ;
      (10) ? ? ? : ? : - ; // Skipped: inactive clock edge (output is '-')
      0    1 1 ? : 0 : 0 ; // Skipped: steady-clock hold row (state == output)
      0    1 1 ? : 1 : 1 ; // Skipped: steady-clock hold row (state == output)
   endtable
endprimitive

//===----------------------------------------------------------------------===//
// Sequential UDP Instantiations
//===----------------------------------------------------------------------===//

// CHECK-LABEL: moore.module @udp_seq_prim()
// CHECK: [[CLK:%.+]] = moore.net wire : <l1>
// CHECK: [[D:%.+]] = moore.net wire : <l1>
// CHECK: [[Q:%.+]] = moore.net wire : <l1>
// CHECK: [[RD_CLK:%.+]] = moore.read [[CLK]] : {{ref<l1>|<l1>}}
// CHECK: [[RD_D:%.+]] = moore.read [[D]] : {{ref<l1>|<l1>}}
// CHECK: [[INST:%.+]] = moore.instance "u2" @udp_seq_dff(clk: [[RD_CLK]]: !moore.l1, d: [[RD_D]]: !moore.l1) -> (q: !moore.l1)
// CHECK: moore.assign [[Q]], [[INST]] : {{ref<l1>|l1}}
module udp_seq_prim;
    wire CLK, D, Q;
    udp_seq_dff u2 (Q, CLK, D);
endmodule
