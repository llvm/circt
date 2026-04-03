// RUN: circt-translate --import-verilog %s | FileCheck %s
// RUN: circt-verilog --ir-moore %s
// REQUIRES: slang

// Internal issue in Slang v3 about jump depending on uninitialised value.
// UNSUPPORTED: valgrind

// CHECK-LABEL: moore.module @and_prim()
// CHECK: [[A:%.+]] = moore.variable : <l1>
// CHECK: [[B:%.+]] = moore.variable : <l1>
// CHECK: [[Q:%.+]] = moore.variable : <l1>
// CHECK: [[RD_A:%.+]] = moore.read [[A]] : <l1>
// CHECK: [[RD_B:%.+]] = moore.read [[B]] : <l1>
// CHECK: [[AND:%.+]] = moore.and [[RD_A]], [[RD_B]] : l1
// CHECK: moore.assign [[Q]], [[AND]] : l1

module and_prim;
    logic A, B, Q;
    and a (Q, A, B);
endmodule

// CHECK-LABEL: moore.module @or_prim()
// CHECK: [[A:%.+]] = moore.variable : <l1>
// CHECK: [[B:%.+]] = moore.variable : <l1>
// CHECK: [[Q:%.+]] = moore.variable : <l1>
// CHECK: [[RD_A:%.+]] = moore.read [[A]] : <l1>
// CHECK: [[RD_B:%.+]] = moore.read [[B]] : <l1>
// CHECK: [[OR:%.+]] = moore.or [[RD_A]], [[RD_B]] : l1
// CHECK: moore.assign [[Q]], [[OR]] : l1

module or_prim;
    logic A, B, Q;
    or a (Q, A, B);
endmodule

// CHECK-LABEL: moore.module @xor_prim()
// CHECK: [[A:%.+]] = moore.variable : <l1>
// CHECK: [[B:%.+]] = moore.variable : <l1>
// CHECK: [[Q:%.+]] = moore.variable : <l1>
// CHECK: [[RD_A:%.+]] = moore.read [[A]] : <l1>
// CHECK: [[RD_B:%.+]] = moore.read [[B]] : <l1>
// CHECK: [[XOR:%.+]] = moore.xor [[RD_A]], [[RD_B]] : l1
// CHECK: moore.assign [[Q]], [[XOR]] : l1

module xor_prim;
    logic A, B, Q;
    xor c (Q, A, B);
endmodule

// CHECK-LABEL: moore.module @multi_and_prim()
// CHECK: [[A:%.+]] = moore.variable : <l1>
// CHECK: [[B:%.+]] = moore.variable : <l1>
// CHECK: [[C:%.+]] = moore.variable : <l1>
// CHECK: [[D:%.+]] = moore.variable : <l1>
// CHECK: [[Q:%.+]] = moore.variable : <l1>
// CHECK: [[RD_A:%.+]] = moore.read [[A]] : <l1>
// CHECK: [[RD_B:%.+]] = moore.read [[B]] : <l1>
// CHECK: [[RD_C:%.+]] = moore.read [[C]] : <l1>
// CHECK: [[RD_D:%.+]] = moore.read [[D]] : <l1>
// CHECK: [[AND0:%.+]] = moore.and [[RD_A]], [[RD_B]] : l1
// CHECK: [[AND1:%.+]] = moore.and [[AND0]], [[RD_C]] : l1
// CHECK: [[AND2:%.+]] = moore.and [[AND1]], [[RD_D]] : l1
// CHECK: moore.assign [[Q]], [[AND2]] : l1

module multi_and_prim;
    logic A, B, C, D, Q;
    and a (Q, A, B, C, D);
endmodule
