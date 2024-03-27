// RUN: circt-translate --import-verilog %s | FileCheck %s
// REQUIRES: slang

// CHECK-LABEL: moore.module @Top1
module Top1;
    // CHECK-NEXT: moore.instance "child1" @Child
    // CHECK-NEXT: moore.instance "child2" @Child_0
    Child child1();
    Child child2();
endmodule

// CHECK-LABEL: moore.module @Top2
module Top2;
endmodule

// CHECK-LABEL: moore.module @Child
// CHECK-NEXT:    moore.instance "p1" @Parametrized
// CHECK-NEXT:    moore.instance "p2" @Parametrized_1
// CHECK-LABEL: moore.module @Child_0
// CHECK-NEXT:    moore.instance "p1" @Parametrized_2
// CHECK-NEXT:    moore.instance "p2" @Parametrized_3
module Child;
    Parametrized #(42) p1();
    Parametrized #(9001) p2();
endmodule

// CHECK-LABEL: moore.module @Parametrized
// CHECK-NEXT:    %x = moore.variable : !moore.packed<range<logic, 41:0>>
// CHECK-LABEL: moore.module @Parametrized_1
// CHECK-NEXT:    %x = moore.variable : !moore.packed<range<logic, 9000:0>>
// CHECK-LABEL: moore.module @Parametrized_2
// CHECK-NEXT:    %x = moore.variable : !moore.packed<range<logic, 41:0>>
// CHECK-LABEL: moore.module @Parametrized_3
// CHECK-NEXT:    %x = moore.variable : !moore.packed<range<logic, 9000:0>>
module Parametrized #(int N);
    logic [N-1:0] x;
endmodule
