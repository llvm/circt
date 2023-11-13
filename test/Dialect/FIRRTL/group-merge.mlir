// RUN: circt-opt -pass-pipeline="builtin.module(firrtl.circuit(firrtl.module(firrtl-group-merge)))" %s | FileCheck %s

// CHECK-LABEL: firrtl.circuit "SimpleMerge"
firrtl.circuit "SimpleMerge" {
  firrtl.declgroup @A  bind {
    firrtl.declgroup @B  bind {
    }
  }
  // CHECK: firrtl.module @SimpleMerge
  firrtl.module @SimpleMerge(in %a: !firrtl.uint<1>, in %b: !firrtl.uint<1>) {
    // CHECK-NEXT: firrtl.group @A {
    // CHECK-NEXT:   %A_a = firrtl.node %a
    // CHECK-NEXT:   %A_b = firrtl.node %b
    // CHECK-NEXT:   firrtl.group @A::@B {
    // CHECK-NEXT:     %A_B_a = firrtl.node %A_a
    // CHECK-NEXT:     %A_B_b = firrtl.node %A_b
    // CHECK-NEXT:   }
    // CHECK-NEXT: }
    // CHECK-NOT:  firrtl.group
    firrtl.group @A {
      %A_a = firrtl.node %a : !firrtl.uint<1>
      firrtl.group @A::@B {
        %A_B_a = firrtl.node %A_a : !firrtl.uint<1>
      }
    }
    firrtl.group @A {
      %A_b = firrtl.node %b : !firrtl.uint<1>
      firrtl.group @A::@B {
        %A_B_b = firrtl.node %A_b : !firrtl.uint<1>
      }
    }
  }
}
