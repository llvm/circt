// RUN: circt-opt %s --convert-hw-to-btor2 -o %t | FileCheck %s

// Check that assertion/assumption labels are emitted as btor2 symbols on the
// generated bad/constraint instructions.

module {
  // CHECK: [[I1:[0-9]+]] sort bitvec 1
  // CHECK: [[A:[0-9]+]] input [[I1]] a
  // CHECK: [[B:[0-9]+]] input [[I1]] b
  // CHECK: [[EN:[0-9]+]] input [[I1]] en
  hw.module @verif(in %a : i1, in %b : i1, in %en : i1, in %clk : !seq.clock) {
    %clock = seq.from_clock %clk
    // CHECK: [[P:[0-9]+]] and [[I1]] [[A]] [[B]]
    %p = comb.and %a, %b : i1
    // CHECK: [[Q:[0-9]+]] or [[I1]] [[A]] [[B]]
    %q = comb.or %a, %b : i1

    // CHECK: [[NOT0:[0-9]+]] not [[I1]] [[P]]
    // CHECK-NEXT: {{[0-9]+}} bad [[NOT0]] my_assert{{$}}
    verif.assert %p label "my_assert" : i1

    // Whitespace is not allowed in btor2 symbols, so it gets replaced.
    // CHECK: [[NOT1:[0-9]+]] not [[I1]] [[P]]
    // CHECK-NEXT: {{[0-9]+}} bad [[NOT1]] assert_with_spaces{{$}}
    verif.assert %p label "assert with spaces" : i1

    // Unlabeled and empty-labeled properties get no symbol.
    // CHECK: [[NOT2:[0-9]+]] not [[I1]] [[P]]
    // CHECK-NEXT: {{[0-9]+}} bad [[NOT2]]{{$}}
    verif.assert %p : i1
    // CHECK: [[NOT3:[0-9]+]] not [[I1]] [[P]]
    // CHECK-NEXT: {{[0-9]+}} bad [[NOT3]]{{$}}
    verif.assert %p label "" : i1

    // Labels are preserved when an enable is folded into the property.
    // CHECK: [[IMPL0:[0-9]+]] implies [[I1]] [[EN]] [[P]]
    // CHECK-NEXT: [[NOT4:[0-9]+]] not [[I1]] [[IMPL0]]
    // CHECK-NEXT: {{[0-9]+}} bad [[NOT4]] enabled_assert{{$}}
    verif.assert %p if %en label "enabled_assert" : i1

    // CHECK: [[NOT5:[0-9]+]] not [[I1]] [[P]]
    // CHECK-NEXT: {{[0-9]+}} bad [[NOT5]] clocked_assert{{$}}
    verif.clocked_assert %p, posedge %clock label "clocked_assert" : i1

    // CHECK: {{[0-9]+}} constraint [[Q]] my_assume{{$}}
    verif.assume %q label "my_assume" : i1

    // CHECK: {{[0-9]+}} constraint [[Q]]{{$}}
    verif.assume %q : i1

    // CHECK: [[IMPL1:[0-9]+]] implies [[I1]] [[EN]] [[Q]]
    // CHECK-NEXT: {{[0-9]+}} constraint [[IMPL1]] enabled_assume{{$}}
    verif.assume %q if %en label "enabled_assume" : i1
  }

  // CHECK: [[SVI1:[0-9]+]] sort bitvec 1
  // CHECK: [[SVA:[0-9]+]] input [[SVI1]] a
  // CHECK: [[SVB:[0-9]+]] input [[SVI1]] b
  hw.module @sv(in %a : i1, in %b : i1, in %clk : !seq.clock) {
    %clock = seq.from_clock %clk
    // CHECK: [[SVP:[0-9]+]] and [[SVI1]] [[SVA]] [[SVB]]
    %p = comb.and %a, %b : i1
    // CHECK: [[SVQ:[0-9]+]] or [[SVI1]] [[SVA]] [[SVB]]
    %q = comb.or %a, %b : i1
    sv.always posedge %clock {
      // CHECK: [[SVNOT0:[0-9]+]] not [[SVI1]] [[SVP]]
      // CHECK-NEXT: {{[0-9]+}} bad [[SVNOT0]] sv_assert{{$}}
      sv.assert %p, immediate label "sv_assert"
      // CHECK: [[SVNOT1:[0-9]+]] not [[SVI1]] [[SVP]]
      // CHECK-NEXT: {{[0-9]+}} bad [[SVNOT1]]{{$}}
      sv.assert %p, immediate
      // CHECK: {{[0-9]+}} constraint [[SVQ]] sv_assume{{$}}
      sv.assume %q, immediate label "sv_assume"
      // CHECK: {{[0-9]+}} constraint [[SVQ]]{{$}}
      sv.assume %q, immediate
    }
  }
}
