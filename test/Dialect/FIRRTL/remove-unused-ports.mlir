// RUN: circt-opt -pass-pipeline='firrtl.circuit(firrtl-remove-unused-ports)' %s | FileCheck %s
firrtl.circuit "Top"   {
  // CHECK-LABEL: firrtl.module @Top(in %a: !firrtl.uint<1>, in %b: !firrtl.uint<1>, out %c: !firrtl.uint<1>,
  // CHECK-SAME :                    out %d_unused: !firrtl.uint<1>, out %d_invalid: !firrtl.uint<1>, out %d_constant: !firrtl.uint<1>)
  firrtl.module @Top(in %a: !firrtl.uint<1>, in %b: !firrtl.uint<1>, out %c: !firrtl.uint<1>,
                     out %d_unused: !firrtl.uint<1>, out %d_invalid: !firrtl.uint<1>, out %d_constant: !firrtl.uint<1>) {
    %A_a, %A_b, %A_c, %A_d_unused, %A_d_invalid, %A_d_constant = firrtl.instance A  @UseBar(in a: !firrtl.uint<1>, in b: !firrtl.uint<1>, out c: !firrtl.uint<1>, out d_unused: !firrtl.uint<1>, out d_invalid: !firrtl.uint<1>, out d_constant: !firrtl.uint<1>)
    // CHECK: %A_b, %A_c = firrtl.instance A @UseBar(in b: !firrtl.uint<1>, out c: !firrtl.uint<1>)
    // CHECK-NEXT: firrtl.connect %A_b, %b
    // CHECK-NEXT: firrtl.connect %c, %A_c
    // CHECK-NEXT: firrtl.connect %d_unused, %{{invalid_ui1.*}}
    // CHECK-NEXT: firrtl.connect %d_invalid, %{{invalid_ui1.*}}
    // CHECK-NEXT: firrtl.connect %d_constant, %{{c1_ui1.*}}
    firrtl.connect %A_a, %a : !firrtl.uint<1>, !firrtl.uint<1>
    firrtl.connect %A_b, %b : !firrtl.uint<1>, !firrtl.uint<1>
    firrtl.connect %c, %A_c : !firrtl.uint<1>, !firrtl.uint<1>
    firrtl.connect %d_unused, %A_d_unused : !firrtl.uint<1>, !firrtl.uint<1>
    firrtl.connect %d_invalid, %A_d_invalid : !firrtl.uint<1>, !firrtl.uint<1>
    firrtl.connect %d_constant, %A_d_constant : !firrtl.uint<1>, !firrtl.uint<1>
  }

  // Check that %a, %d_unused, %d_invalid and %d_constant are removed.
  // CHECK-LABEL: firrtl.module @Bar(in %b: !firrtl.uint<1>, out %c: !firrtl.uint<1>)
  // CHECK-NEXT:    firrtl.connect %c, %b
  // CHECK-NEXT:  }
  firrtl.module @Bar(in %a: !firrtl.uint<1>, in %b: !firrtl.uint<1>, out %c: !firrtl.uint<1>,
                     out %d_unused: !firrtl.uint<1>, out %d_invalid: !firrtl.uint<1>, out %d_constant: !firrtl.uint<1>) {
    firrtl.connect %c, %b : !firrtl.uint<1>, !firrtl.uint<1>

    %invalid_ui1 = firrtl.invalidvalue : !firrtl.uint<1>
    firrtl.connect %d_invalid, %invalid_ui1 : !firrtl.uint<1>, !firrtl.uint<1>
    %c1_i1 = firrtl.constant 1 : !firrtl.uint<1>
    firrtl.connect %d_constant, %c1_i1 : !firrtl.uint<1>, !firrtl.uint<1>
  }

  // Check that %a, %d_unused, %d_invalid and %d_constant are removed.
  // CHECK-LABEL: firrtl.module @UseBar(in %b: !firrtl.uint<1>, out %c: !firrtl.uint<1>) {
  firrtl.module @UseBar(in %a: !firrtl.uint<1>, in %b: !firrtl.uint<1>, out %c: !firrtl.uint<1>,
                        out %d_unused: !firrtl.uint<1>, out %d_invalid: !firrtl.uint<1>, out %d_constant: !firrtl.uint<1>) {
    %A_a, %A_b, %A_c, %A_d_unused, %A_d_invalid, %A_d_constant = firrtl.instance A  @Bar(in a: !firrtl.uint<1>, in b: !firrtl.uint<1>, out c: !firrtl.uint<1>, out d_unused: !firrtl.uint<1>, out d_invalid: !firrtl.uint<1>, out d_constant: !firrtl.uint<1>)
    firrtl.connect %A_a, %a : !firrtl.uint<1>, !firrtl.uint<1>
    // CHECK: %A_b, %A_c = firrtl.instance A  @Bar(in b: !firrtl.uint<1>, out c: !firrtl.uint<1>)
    firrtl.connect %A_b, %b : !firrtl.uint<1>, !firrtl.uint<1>
    firrtl.connect %c, %A_c : !firrtl.uint<1>, !firrtl.uint<1>
    firrtl.connect %d_unused, %A_d_unused : !firrtl.uint<1>, !firrtl.uint<1>
    firrtl.connect %d_invalid, %A_d_invalid : !firrtl.uint<1>, !firrtl.uint<1>
    firrtl.connect %d_constant, %A_d_constant : !firrtl.uint<1>, !firrtl.uint<1>
  }

  // Make sure that %a, %b and %c are not erased because they have an annotation or a symbol.
  // CHECK-LABEL: firrtl.module @Foo(in %a: !firrtl.uint<1> sym @dntSym, in %b: !firrtl.uint<1> [{a = "a"}], out %c: !firrtl.uint<1> sym @dntSym)
  firrtl.module @Foo(in %a: !firrtl.uint<1>, in %b: !firrtl.uint<1>, out %c: !firrtl.uint<1>) attributes {
    portAnnotations = [[], [{a = "a"}], []], portSyms = ["dntSym", "", "dntSym"]}
  {
    // CHECK: firrtl.connect %c, %{{invalid_ui1.*}}
    %invalid_ui1 = firrtl.invalidvalue : !firrtl.uint<1>
    firrtl.connect %c, %invalid_ui1 : !firrtl.uint<1>, !firrtl.uint<1>
  }

  // CHECK-LABEL: firrtl.module @UseFoo(in %a: !firrtl.uint<1>, in %b: !firrtl.uint<1>, out %c: !firrtl.uint<1>)
  firrtl.module @UseFoo(in %a: !firrtl.uint<1>, in %b: !firrtl.uint<1>, out %c: !firrtl.uint<1>) {
    %A_a, %A_b, %A_c = firrtl.instance A  @Foo(in a: !firrtl.uint<1>, in b: !firrtl.uint<1>, out c: !firrtl.uint<1>)
    // CHECK: %A_a, %A_b, %A_c = firrtl.instance A @Foo(in a: !firrtl.uint<1>, in b: !firrtl.uint<1>, out c: !firrtl.uint<1>)
    firrtl.connect %A_a, %a : !firrtl.uint<1>, !firrtl.uint<1>
    firrtl.connect %A_b, %b : !firrtl.uint<1>, !firrtl.uint<1>
    firrtl.connect %c, %A_c : !firrtl.uint<1>, !firrtl.uint<1>
  }
}