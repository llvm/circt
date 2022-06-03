// RUN: circt-opt -hw-remove-unused-ports --split-input-file %s | FileCheck %s

module {
  // CHECK-LABEL: private @Child2() {
  hw.module private @Child2(%in: i1) {
    // CHECK-NEXT: hw.output
    hw.output
  }

  // CHECK-LABEL: private @Child1() {
  hw.module private @Child1(%in: i1) -> (out: i1) {
    // CHECK-NEXT: hw.output
    hw.output %in : i1
  }

  // CHECK-LABEL: hw.module @Parent(%in: i1)
  hw.module @Parent(%in: i1) {
    // CHECK-NEXT: hw.instance "child1" @Child1() -> ()
    %child1.out = hw.instance "child1" @Child1(in: %in: i1) -> (out: i1)
    // CHECK-NEXT: hw.instance "child2" @Child2() -> ()
    hw.instance "child2" @Child2(in: %child1.out: i1) -> ()
    // CHECK-NEXT: hw.output
    hw.output
  }
}

// -----
module {
  // CHECK-LABEL: hw.module @Top(%a: i1, %b: i1) -> (c: i1, d_unused: i1, d_invalid: i1, d_constant: i1)
  hw.module @Top(%a: i1, %b: i1) -> (c: i1, d_unused: i1, d_invalid: i1, d_constant: i1) {
    // Check that constants are forwarded to the caller.
    // CHECK-NEXT: %x_i1 = sv.constantX
    // CHECK-NEXT: %false = hw.constant false
    // CHECK-NEXT: %A.c = hw.instance "A" @UseBar(b: %b: i1) -> (c: i1)
    // CHECK-NEXT: hw.instance "B" @UseFoo(a: %a: i1)
    // CHECK-NEXT: hw.output %A.c, %A.c, %x_i1, %false
    %A.c, %A.d_unused, %A.d_invalid, %A.d_constant = hw.instance "A" @UseBar(a: %a: i1, b: %b: i1) -> (c: i1, d_unused: i1, d_invalid: i1, d_constant: i1)
    %B.c = hw.instance "B" @UseFoo(a: %a: i1) -> (c: i1)
    hw.output %A.c, %A.c, %A.d_invalid, %A.d_constant : i1, i1, i1, i1
  }

  // CHECK-LABEL: hw.module private @Bar(%b: i1) -> (c: i1)
  hw.module private @Bar(%a: i1, %b: i1) -> (c: i1, d_unused: i1, d_invalid: i1, d_constant: i1) {
    // CHECK-NEXT: hw.output %b : i1
    %x = sv.constantX : i1
    %false = hw.constant false
    hw.output %b, %b, %x, %false : i1, i1, i1, i1
  }

  // CHECK-LABEL: hw.module private @UseBar(%b: i1) -> (c: i1)
  hw.module private @UseBar(%a: i1, %b: i1) -> (c: i1, d_unused: i1, d_invalid: i1, d_constant: i1) {
    // CHECK-NEXT: %A.c = hw.instance "A" @Bar(b: %b: i1) -> (c: i1)
    // CHECK-NEXT: hw.output %A.c : i1
    %A.c, %A.d_unused, %A.d_invalid, %A.d_constant = hw.instance "A" @Bar(a: %a: i1, b: %b: i1) -> (c: i1, d_unused: i1, d_invalid: i1, d_constant: i1)
    hw.output %A.c, %A.d_unused, %A.d_invalid, %A.d_constant : i1, i1, i1, i1
  }
  // Check that %a and %c are not erased.
  // CHECK-LABEL: hw.module private @Foo(%a: i1 {hw.exportPort = @dntSym}) -> (c: i1 {hw.exportPort = @dntSym})
  hw.module private @Foo(%a: i1 {hw.exportPort = @dntSym}) -> (c: i1 {hw.exportPort = @dntSym}) {
    %false = hw.constant false
    // CHECK:  hw.output %false
    hw.output %false : i1
  }

  // CHECK-LABEL: hw.module private @UseFoo(%a: i1) {
  hw.module private @UseFoo(%a: i1) -> (c: i1) {
    // CHECK-NEXT:  %A.c = hw.instance "A" @Foo(a: %a: i1) -> (c: i1)
    // CHECK-NEXT:  hw.output
    %A.c = hw.instance "A" @Foo(a: %a: i1) -> (c: i1)
    hw.output %A.c : i1
  }
}

// -----
// Check that output_file attributes are preserved.
module {
  // CHECK-LABEL: @Sub() attributes {output_file = #hw.output_file<"hello">}
  hw.module private @Sub(%a: i1) attributes {output_file = #hw.output_file<"hello">} {
    hw.output
  }
  hw.module @PreserveOutputFile() {
    %.a.wire = sv.wire  : !hw.inout<i1>
    %0 = sv.read_inout %.a.wire : !hw.inout<i1>
    // CHECK: hw.instance "sub" @Sub() -> () {output_file = #hw.output_file<"hello">}
    hw.instance "sub" @Sub(a: %0: i1) -> () {output_file = #hw.output_file<"hello">}
    hw.output
  }
}
