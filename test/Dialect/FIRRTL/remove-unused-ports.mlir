// RUN: circt-opt -pass-pipeline='builtin.module(firrtl.circuit(firrtl-remove-unused-ports))' %s -split-input-file | FileCheck %s
firrtl.circuit "Top"   {
  // CHECK-LABEL: firrtl.module @Top(in %a: !firrtl.uint<1>, in %b: !firrtl.uint<1>, out %c: !firrtl.uint<1>,
  // CHECK-SAME:                     out %d_unused: !firrtl.uint<1>, out %d_invalid: !firrtl.uint<1>, out %d_constant: !firrtl.uint<1>)
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
  // CHECK-LABEL: firrtl.module private @Bar(in %b: !firrtl.uint<1>, out %c: !firrtl.uint<1>)
  // CHECK-NEXT:    firrtl.connect %c, %b
  // CHECK-NEXT:  }
  firrtl.module private @Bar(in %a: !firrtl.uint<1>, in %b: !firrtl.uint<1>, out %c: !firrtl.uint<1>,
                     out %d_unused: !firrtl.uint<1>, out %d_invalid: !firrtl.uint<1>, out %d_constant: !firrtl.uint<1>) {
    firrtl.connect %c, %b : !firrtl.uint<1>, !firrtl.uint<1>

    %invalid_ui1 = firrtl.invalidvalue : !firrtl.uint<1>
    firrtl.connect %d_invalid, %invalid_ui1 : !firrtl.uint<1>, !firrtl.uint<1>
    %c1_i1 = firrtl.constant 1 : !firrtl.uint<1>
    firrtl.connect %d_constant, %c1_i1 : !firrtl.uint<1>, !firrtl.uint<1>
  }

  // Check that %a, %d_unused, %d_invalid and %d_constant are removed.
  // CHECK-LABEL: firrtl.module private @UseBar(in %b: !firrtl.uint<1>, out %c: !firrtl.uint<1>) {
  firrtl.module private @UseBar(in %a: !firrtl.uint<1>, in %b: !firrtl.uint<1>, out %c: !firrtl.uint<1>,
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
  // CHECK-LABEL: firrtl.module private @Foo(in %a: !firrtl.uint<1> sym @dntSym, in %b: !firrtl.uint<1> [{a = "a"}], out %c: !firrtl.uint<1> sym @dntSym2)
  firrtl.module private @Foo(in %a: !firrtl.uint<1> sym @dntSym, in %b: !firrtl.uint<1>, out %c: !firrtl.uint<1> sym @dntSym2) attributes {
    portAnnotations = [[], [{a = "a"}], []]}
  {
    // CHECK: firrtl.connect %c, %{{invalid_ui1.*}}
    %invalid_ui1 = firrtl.invalidvalue : !firrtl.uint<1>
    firrtl.connect %c, %invalid_ui1 : !firrtl.uint<1>, !firrtl.uint<1>
  }

  // CHECK-LABEL: firrtl.module private @UseFoo(in %a: !firrtl.uint<1>, in %b: !firrtl.uint<1>, out %c: !firrtl.uint<1>)
  firrtl.module private @UseFoo(in %a: !firrtl.uint<1>, in %b: !firrtl.uint<1>, out %c: !firrtl.uint<1>) {
    %A_a, %A_b, %A_c = firrtl.instance A  @Foo(in a: !firrtl.uint<1>, in b: !firrtl.uint<1>, out c: !firrtl.uint<1>)
    // CHECK: %A_a, %A_b, %A_c = firrtl.instance A @Foo(in a: !firrtl.uint<1>, in b: !firrtl.uint<1>, out c: !firrtl.uint<1>)
    firrtl.connect %A_a, %a : !firrtl.uint<1>, !firrtl.uint<1>
    firrtl.connect %A_b, %b : !firrtl.uint<1>, !firrtl.uint<1>
    firrtl.connect %c, %A_c : !firrtl.uint<1>, !firrtl.uint<1>
  }
}

// -----

// Matching connect version.
firrtl.circuit "Top"   {
  // CHECK-LABEL: firrtl.module @Top(in %a: !firrtl.uint<1>, in %b: !firrtl.uint<1>, out %c: !firrtl.uint<1>,
  // CHECK-SAME:                     out %d_unused: !firrtl.uint<1>, out %d_invalid: !firrtl.uint<1>, out %d_constant: !firrtl.uint<1>)
  firrtl.module @Top(in %a: !firrtl.uint<1>, in %b: !firrtl.uint<1>, out %c: !firrtl.uint<1>,
                     out %d_unused: !firrtl.uint<1>, out %d_invalid: !firrtl.uint<1>, out %d_constant: !firrtl.uint<1>) {
    %A_a, %A_b, %A_c, %A_d_unused, %A_d_invalid, %A_d_constant = firrtl.instance A  @UseBar(in a: !firrtl.uint<1>, in b: !firrtl.uint<1>, out c: !firrtl.uint<1>, out d_unused: !firrtl.uint<1>, out d_invalid: !firrtl.uint<1>, out d_constant: !firrtl.uint<1>)
    // CHECK: %A_b, %A_c = firrtl.instance A @UseBar(in b: !firrtl.uint<1>, out c: !firrtl.uint<1>)
    // CHECK-NEXT: firrtl.matchingconnect %A_b, %b
    // CHECK-NEXT: firrtl.matchingconnect %c, %A_c
    // CHECK-NEXT: firrtl.matchingconnect %d_unused, %{{invalid_ui1.*}}
    // CHECK-NEXT: firrtl.matchingconnect %d_invalid, %{{invalid_ui1.*}}
    // CHECK-NEXT: firrtl.matchingconnect %d_constant, %{{c1_ui1.*}}
    firrtl.matchingconnect %A_a, %a : !firrtl.uint<1>
    firrtl.matchingconnect %A_b, %b : !firrtl.uint<1>
    firrtl.matchingconnect %c, %A_c : !firrtl.uint<1>
    firrtl.matchingconnect %d_unused, %A_d_unused : !firrtl.uint<1>
    firrtl.matchingconnect %d_invalid, %A_d_invalid : !firrtl.uint<1>
    firrtl.matchingconnect %d_constant, %A_d_constant : !firrtl.uint<1>
  }

  // Check that %a, %d_unused, %d_invalid and %d_constant are removed.
  // CHECK-LABEL: firrtl.module private @Bar(in %b: !firrtl.uint<1>, out %c: !firrtl.uint<1>)
  // CHECK-NEXT:    firrtl.matchingconnect %c, %b
  // CHECK-NEXT:  }
  firrtl.module private @Bar(in %a: !firrtl.uint<1>, in %b: !firrtl.uint<1>, out %c: !firrtl.uint<1>,
                     out %d_unused: !firrtl.uint<1>, out %d_invalid: !firrtl.uint<1>, out %d_constant: !firrtl.uint<1>) {
    firrtl.matchingconnect %c, %b : !firrtl.uint<1>

    %invalid_ui1 = firrtl.invalidvalue : !firrtl.uint<1>
    firrtl.matchingconnect %d_invalid, %invalid_ui1 : !firrtl.uint<1>
    %c1_i1 = firrtl.constant 1 : !firrtl.uint<1>
    firrtl.matchingconnect %d_constant, %c1_i1 : !firrtl.uint<1>
  }

  // Check that %a, %d_unused, %d_invalid and %d_constant are removed.
  // CHECK-LABEL: firrtl.module private @UseBar(in %b: !firrtl.uint<1>, out %c: !firrtl.uint<1>) {
  firrtl.module private @UseBar(in %a: !firrtl.uint<1>, in %b: !firrtl.uint<1>, out %c: !firrtl.uint<1>,
                        out %d_unused: !firrtl.uint<1>, out %d_invalid: !firrtl.uint<1>, out %d_constant: !firrtl.uint<1>) {
    %A_a, %A_b, %A_c, %A_d_unused, %A_d_invalid, %A_d_constant = firrtl.instance A  @Bar(in a: !firrtl.uint<1>, in b: !firrtl.uint<1>, out c: !firrtl.uint<1>, out d_unused: !firrtl.uint<1>, out d_invalid: !firrtl.uint<1>, out d_constant: !firrtl.uint<1>)
    firrtl.matchingconnect %A_a, %a : !firrtl.uint<1>
    // CHECK: %A_b, %A_c = firrtl.instance A  @Bar(in b: !firrtl.uint<1>, out c: !firrtl.uint<1>)
    firrtl.matchingconnect %A_b, %b : !firrtl.uint<1>
    firrtl.matchingconnect %c, %A_c : !firrtl.uint<1>
    firrtl.matchingconnect %d_unused, %A_d_unused : !firrtl.uint<1>
    firrtl.matchingconnect %d_invalid, %A_d_invalid : !firrtl.uint<1>
    firrtl.matchingconnect %d_constant, %A_d_constant : !firrtl.uint<1>
  }

  // Make sure that %a, %b and %c are not erased because they have an annotation or a symbol.
  // CHECK-LABEL: firrtl.module private @Foo(in %a: !firrtl.uint<1> sym @dntSym, in %b: !firrtl.uint<1> [{a = "a"}], out %c: !firrtl.uint<1> sym @dntSym2)
  firrtl.module private @Foo(in %a: !firrtl.uint<1> sym @dntSym, in %b: !firrtl.uint<1>, out %c: !firrtl.uint<1> sym @dntSym2) attributes {
    portAnnotations = [[], [{a = "a"}], []]}
  {
    // CHECK: firrtl.matchingconnect %c, %{{invalid_ui1.*}}
    %invalid_ui1 = firrtl.invalidvalue : !firrtl.uint<1>
    firrtl.matchingconnect %c, %invalid_ui1 : !firrtl.uint<1>
  }

  // CHECK-LABEL: firrtl.module private @UseFoo(in %a: !firrtl.uint<1>, in %b: !firrtl.uint<1>, out %c: !firrtl.uint<1>)
  firrtl.module private @UseFoo(in %a: !firrtl.uint<1>, in %b: !firrtl.uint<1>, out %c: !firrtl.uint<1>) {
    %A_a, %A_b, %A_c = firrtl.instance A  @Foo(in a: !firrtl.uint<1>, in b: !firrtl.uint<1>, out c: !firrtl.uint<1>)
    // CHECK: %A_a, %A_b, %A_c = firrtl.instance A @Foo(in a: !firrtl.uint<1>, in b: !firrtl.uint<1>, out c: !firrtl.uint<1>)
    firrtl.matchingconnect %A_a, %a : !firrtl.uint<1>
    firrtl.matchingconnect %A_b, %b : !firrtl.uint<1>
    firrtl.matchingconnect %c, %A_c : !firrtl.uint<1>
  }
}

// -----

// Ensure that the "output_file" attribute isn't destroyed by RemoveUnusedPorts.
// This matters for interactions between Grand Central (which sets these) and
// RemoveUnusedPorts which may clone modules with stripped ports.
//
// CHECK-LABEL: "PreserveOutputFile"
firrtl.circuit "PreserveOutputFile" {
  // CHECK-NEXT: firrtl.module {{.+}}@Sub
  // CHECK-SAME:   output_file
  firrtl.module private @Sub(in %a: !firrtl.uint<1>) attributes {output_file = #hw.output_file<"hello">} {}
  // CHECK: firrtl.module @PreserveOutputFile
  firrtl.module @PreserveOutputFile() {
    // CHECK-NEXT: firrtl.instance sub
    // CHECK-SAME: output_file
    firrtl.instance sub {output_file = #hw.output_file<"hello">} @Sub(in a: !firrtl.uint<1>)
  }
}

// -----

// CHECK-LABEL: "UnusedOutput"
firrtl.circuit "UnusedOutput"  {
  // CHECK: firrtl.module {{.+}}@SingleDriver
  // CHECK-NOT:     out %c
  firrtl.module private @SingleDriver(in %a: !firrtl.uint<1>, out %b: !firrtl.uint<1>, out %c: !firrtl.uint<1>) {
    // CHECK-NEXT: %[[c_wire:.+]] = firrtl.wire
    // CHECK-NEXT: firrtl.matchingconnect %b, %[[c_wire]]
    firrtl.matchingconnect %b, %c : !firrtl.uint<1>
    // CHECK-NEXT: %[[not_a:.+]] = firrtl.not %a
    %0 = firrtl.not %a : (!firrtl.uint<1>) -> !firrtl.uint<1>
    // CHECK-NEXT: firrtl.matchingconnect %[[c_wire]], %[[not_a]]
    firrtl.matchingconnect %c, %0 : !firrtl.uint<1>
  }
  // CHECK-LABEL: @UnusedOutput
  firrtl.module @UnusedOutput(in %a: !firrtl.uint<1>, out %b: !firrtl.uint<1>) {
    // CHECK: %singleDriver_a, %singleDriver_b = firrtl.instance singleDriver
    %singleDriver_a, %singleDriver_b, %singleDriver_c = firrtl.instance singleDriver @SingleDriver(in a: !firrtl.uint<1>, out b: !firrtl.uint<1>, out c: !firrtl.uint<1>)
    firrtl.matchingconnect %singleDriver_a, %a : !firrtl.uint<1>
    firrtl.matchingconnect %b, %singleDriver_b : !firrtl.uint<1>
  }
}

// -----

// OMIR annotations should not block removal.
//   - See: https://github.com/llvm/circt/issues/6199
//
// CHECK-LABEL: firrtl.circuit "OMIRRemoval"
firrtl.circuit "OMIRRemoval" {
  // CHECK-NOT: %a
  // CHECK-NOT: %b
  // CHECK-NOT: %c
  // CHECK:     %d
  firrtl.module private @Foo(
    out %a: !firrtl.uint<1> [
      {
         class = "freechips.rocketchip.objectmodel.OMIRTracker",
         id = 0 : i64,
         type = "OMReferenceTarget"
      }
    ],
    out %b: !firrtl.uint<2> [
      {
         class = "freechips.rocketchip.objectmodel.OMIRTracker",
         id = 1 : i64,
         type = "OMMemberReferenceTarget"
      }
    ],
    in %c: !firrtl.uint<3> [
      {
         class = "freechips.rocketchip.objectmodel.OMIRTracker",
         id = 3 : i64,
         type = "OMMemberInstanceTarget"
      }
    ],
    in %d: !firrtl.uint<4> [
      {
         class = "freechips.rocketchip.objectmodel.OMIRTracker",
         id = 4 : i64,
         type = "OMMemberInstanceTarget"
      },
      // Adding one additional annotation will block removal.
      {
         class = "circt.test"
      }
    ]
  ) {}
  firrtl.module @OMIRRemoval() {
    %foo_a, %foo_b, %foo_c, %foo_d = firrtl.instance foo @Foo(
      out a: !firrtl.uint<1>,
      out b: !firrtl.uint<2>,
      in  c: !firrtl.uint<3>,
      in  d: !firrtl.uint<4>
    )
  }
}
