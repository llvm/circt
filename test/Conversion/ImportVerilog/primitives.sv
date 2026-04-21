// RUN: circt-translate --import-verilog %s | FileCheck %s
// RUN: circt-verilog --ir-moore %s
// REQUIRES: slang

// Internal issue in Slang v3 about jump depending on uninitialised value.
// UNSUPPORTED: valgrind

// CHECK-LABEL: moore.module @and_prim()
// CHECK: [[A:%.+]] = moore.net wire : <l1>
// CHECK: [[B:%.+]] = moore.net wire : <l1>
// CHECK: [[Q:%.+]] = moore.net wire : <l1>
// CHECK: [[RD_A:%.+]] = moore.read [[A]] : <l1>
// CHECK: [[RD_B:%.+]] = moore.read [[B]] : <l1>
// CHECK: [[AND:%.+]] = moore.and [[RD_A]], [[RD_B]] : l1
// CHECK: moore.assign [[Q]], [[AND]] : l1

module and_prim;
    wire A, B, Q;
    and a (Q, A, B);
endmodule

// CHECK-LABEL: moore.module @or_prim()
// CHECK: [[A:%.+]] = moore.net wire : <l1>
// CHECK: [[B:%.+]] = moore.net wire : <l1>
// CHECK: [[Q:%.+]] = moore.net wire : <l1>
// CHECK: [[RD_A:%.+]] = moore.read [[A]] : <l1>
// CHECK: [[RD_B:%.+]] = moore.read [[B]] : <l1>
// CHECK: [[OR:%.+]] = moore.or [[RD_A]], [[RD_B]] : l1
// CHECK: moore.assign [[Q]], [[OR]] : l1

module or_prim;
    wire A, B, Q;
    or a (Q, A, B);
endmodule

// CHECK-LABEL: moore.module @xor_prim()
// CHECK: [[A:%.+]] = moore.net wire : <l1>
// CHECK: [[B:%.+]] = moore.net wire : <l1>
// CHECK: [[Q:%.+]] = moore.net wire : <l1>
// CHECK: [[RD_A:%.+]] = moore.read [[A]] : <l1>
// CHECK: [[RD_B:%.+]] = moore.read [[B]] : <l1>
// CHECK: [[XOR:%.+]] = moore.xor [[RD_A]], [[RD_B]] : l1
// CHECK: moore.assign [[Q]], [[XOR]] : l1

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
// CHECK: [[RD_A:%.+]] = moore.read [[A]] : <l1>
// CHECK: [[RD_B:%.+]] = moore.read [[B]] : <l1>
// CHECK: [[RD_C:%.+]] = moore.read [[C]] : <l1>
// CHECK: [[RD_D:%.+]] = moore.read [[D]] : <l1>
// CHECK: [[AND0:%.+]] = moore.and [[RD_A]], [[RD_B]] : l1
// CHECK: [[AND1:%.+]] = moore.and [[AND0]], [[RD_C]] : l1
// CHECK: [[AND2:%.+]] = moore.and [[AND1]], [[RD_D]] : l1
// CHECK: moore.assign [[Q]], [[AND2]] : l1

module multi_and_prim;
    wire A, B, C, D, Q;
    and a (Q, A, B, C, D);
endmodule

// CHECK-LABEL: moore.module @nand_prim()
// CHECK: [[A:%.+]] = moore.net wire : <l1>
// CHECK: [[B:%.+]] = moore.net wire : <l1>
// CHECK: [[Q:%.+]] = moore.net wire : <l1>
// CHECK: [[RD_A:%.+]] = moore.read [[A]] : <l1>
// CHECK: [[RD_B:%.+]] = moore.read [[B]] : <l1>
// CHECK: [[AND:%.+]] = moore.and [[RD_A]], [[RD_B]] : l1
// CHECK: [[NOT_AND:%.+]] = moore.not [[AND]] : l1
// CHECK: moore.assign [[Q]], [[NOT_AND]] : l1

module nand_prim;
    wire A, B, Q;
    nand a (Q, A, B);
endmodule

// CHECK-LABEL: moore.module @nor_prim()
// CHECK: [[A:%.+]] = moore.net wire : <l1>
// CHECK: [[B:%.+]] = moore.net wire : <l1>
// CHECK: [[Q:%.+]] = moore.net wire : <l1>
// CHECK: [[RD_A:%.+]] = moore.read [[A]] : <l1>
// CHECK: [[RD_B:%.+]] = moore.read [[B]] : <l1>
// CHECK: [[OR:%.+]] = moore.or [[RD_A]], [[RD_B]] : l1
// CHECK: [[NOT_OR:%.+]] = moore.not [[OR]] : l1
// CHECK: moore.assign [[Q]], [[NOT_OR]] : l1

module nor_prim;
    wire A, B, Q;
    nor a (Q, A, B);
endmodule

// CHECK-LABEL: moore.module @xnor_prim()
// CHECK: [[A:%.+]] = moore.net wire : <l1>
// CHECK: [[B:%.+]] = moore.net wire : <l1>
// CHECK: [[Q:%.+]] = moore.net wire : <l1>
// CHECK: [[RD_A:%.+]] = moore.read [[A]] : <l1>
// CHECK: [[RD_B:%.+]] = moore.read [[B]] : <l1>
// CHECK: [[XOR:%.+]] = moore.xor [[RD_A]], [[RD_B]] : l1
// CHECK: [[NOT_XOR:%.+]] = moore.not [[XOR]] : l1
// CHECK: moore.assign [[Q]], [[NOT_XOR]] : l1

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
// CHECK: [[RD_A:%.+]] = moore.read [[A]] : <l1>
// CHECK: [[RD_B:%.+]] = moore.read [[B]] : <l1>
// CHECK: [[RD_C:%.+]] = moore.read [[C]] : <l1>
// CHECK: [[RD_D:%.+]] = moore.read [[D]] : <l1>
// CHECK: [[AND0:%.+]] = moore.and [[RD_A]], [[RD_B]] : l1
// CHECK: [[AND1:%.+]] = moore.and [[AND0]], [[RD_C]] : l1
// CHECK: [[AND2:%.+]] = moore.and [[AND1]], [[RD_D]] : l1
// CHECK: [[NOT_AND:%.+]] = moore.not [[AND2]] : l1
// CHECK: moore.assign [[Q]], [[NOT_AND]] : l1

module multi_nand_prim;
    wire A, B, C, D, Q;
    nand a (Q, A, B, C, D);
endmodule

// CHECK-LABEL: moore.module @delayed_nand_prim()
// CHECK: [[A:%.+]] = moore.net wire : <l1>
// CHECK: [[B:%.+]] = moore.net wire : <l1>
// CHECK: [[Q:%.+]] = moore.net wire : <l1>
// CHECK: [[RD_A:%.+]] = moore.read [[A]] : <l1>
// CHECK: [[RD_B:%.+]] = moore.read [[B]] : <l1>
// CHECK: [[AND:%.+]] = moore.and [[RD_A]], [[RD_B]] : l1
// CHECK: [[NOT_AND:%.+]] = moore.not [[AND]] : l1
// CHECK: [[DELAYCONST:%.+]] = moore.constant_time 5000000 fs
// CHECK: moore.delayed_assign [[Q]], [[NOT_AND]], [[DELAYCONST]] : l1

module delayed_nand_prim;
    wire A, B, Q;
    nand #5 a (Q, A, B);
endmodule

// CHECK-LABEL: moore.module @delayed3_nand_prim()
// CHECK: [[A:%.+]] = moore.net wire : <l1>
// CHECK: [[B:%.+]] = moore.net wire : <l1>
// CHECK: [[Q:%.+]] = moore.net wire : <l1>
// CHECK: [[RD_A:%.+]] = moore.read [[A]] : <l1>
// CHECK: [[RD_B:%.+]] = moore.read [[B]] : <l1>
// CHECK: [[AND:%.+]] = moore.and [[RD_A]], [[RD_B]] : l1
// CHECK: [[NOT_AND:%.+]] = moore.not [[AND]] : l1
// CHECK: [[DELAYCONST:%.+]] = moore.constant_time 5000000 fs
// CHECK: moore.delayed_assign [[Q]], [[NOT_AND]], [[DELAYCONST]] : l1

module delayed3_nand_prim;
    wire A, B, Q;
    nand #(5) a (Q, A, B);
endmodule


// CHECK-LABEL: moore.module @not_prim()
// CHECK: [[A:%.+]] = moore.net wire : <l1>
// CHECK: [[Q:%.+]] = moore.net wire : <l1>
// CHECK: [[RD_A:%.+]] = moore.read [[A]] : <l1>
// CHECK: [[NOT:%.+]] = moore.not [[RD_A]] : l1
// CHECK: moore.assign [[Q]], [[NOT]] : l1

module not_prim;
    wire A, Q;
    not n (Q, A);
endmodule

// CHECK-LABEL: moore.module @multi_not_prim()
// CHECK: [[A:%.+]] = moore.net wire : <l1>
// CHECK: [[Q0:%.+]] = moore.net wire : <l1>
// CHECK: [[Q1:%.+]] = moore.net wire : <l1>
// CHECK: [[RD_A:%.+]] = moore.read [[A]] : <l1>
// CHECK: [[NOT:%.+]] = moore.not [[RD_A]] : l1
// CHECK: moore.assign [[Q0]], [[NOT]] : l1
// CHECK: moore.assign [[Q1]], [[NOT]] : l1

module multi_not_prim;
    wire A, Q0, Q1;
    not n (Q0, Q1, A);
endmodule

// CHECK-LABEL: moore.module @buf_prim()
// CHECK: [[A:%.+]] = moore.net wire : <l1>
// CHECK: [[Q:%.+]] = moore.net wire : <l1>
// CHECK: [[RD_A:%.+]] = moore.read [[A]] : <l1>
// CHECK: [[NOT:%.+]] = moore.bool_cast [[RD_A]] : l1
// CHECK: moore.assign [[Q]], [[NOT]] : l1

module buf_prim;
    wire A, Q;
    buf n (Q, A);
endmodule

// CHECK-LABEL: moore.module @pullup_prim()
// CHECK: [[A:%.+]] = moore.net wire : <l1>
// CHECK: [[ONE:%.+]] = moore.constant 1 : l1
// CHECK: moore.assign [[A]], [[ONE]] : l1

module pullup_prim;
    wire A;
    pullup n (A);
endmodule

// CHECK-LABEL: moore.module @pulldown_prim()
// CHECK: [[A:%.+]] = moore.net wire : <l1>
// CHECK: [[ZERO:%.+]] = moore.constant 0 : l1
// CHECK: moore.assign [[A]], [[ZERO]] : l1

module pulldown_prim;
    wire A;
    pulldown n (A);
endmodule

// CHECK-LABEL: moore.module @wide_pullup_prim()
// CHECK: [[A:%.+]] = moore.net wire : <l4>
// CHECK: [[ONES:%.+]] = moore.constant -1 : l4
// CHECK: moore.assign [[A]], [[ONES]] : l4

module wide_pullup_prim;
    wire [3:0] A;
    pullup n (A);
endmodule
