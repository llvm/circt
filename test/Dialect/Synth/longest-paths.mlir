// RUN: circt-opt %s --split-input-file --pass-pipeline='builtin.module(synth-print-longest-path-analysis{test=true})' --verify-diagnostics

// expected-remark-re @below {{endPoint=Object($root.x[0]), startPoint=Object($root.a[0], delay=4, history=[{{.+}}])}}
// expected-remark-re @below {{endPoint=Object($root.x[0]), startPoint=Object($root.b[0], delay=4, history=[{{.+}}])}}
hw.module private @basic(in %a : i1, in %b : i1, out x : i1) {
  // These operations are considered to have a delay of 1.
  %p = synth.aig.and_inv not %a, %b : i1 // p[0] := max(a[0], b[0]) + 1 = 1
  %q = comb.and %p, %a : i1  // q[0] := max(p[0], a[0]) + 1 = 2
  %r = comb.or %q, %a : i1  // r[0] := max(q[0], a[0]) + 1 = 3
  %s = comb.xor %r, %a : i1 // s[0] := max(r[0], a[0]) + 1 = 4
  hw.output %s : i1
}

// expected-remark-re @below {{endPoint=Object($root.x[0]), startPoint=Object($root.a[0], delay=1, history=[{{.+}}])}}
// expected-remark-re @below {{endPoint=Object($root.x[0]), startPoint=Object($root.b[0], delay=1, history=[{{.+}}])}}
// expected-remark-re @below {{endPoint=Object($root.x[1]), startPoint=Object($root.a[1], delay=1, history=[{{.+}}])}}
// expected-remark-re @below {{endPoint=Object($root.x[1]), startPoint=Object($root.b[1], delay=1, history=[{{.+}}])}}
hw.module private @basicWord(in %a : i2, in %b : i2, out x : i2) {
  %r = synth.aig.and_inv not %a, %b : i2 // r[i] := max(a[i], b[i]) + 1 = 1
  hw.output %r : i2
}

// a[2] is delay 2, a[0] and a[1] are delay 1
// expected-remark-re @below {{endPoint=Object($root.x[0]), startPoint=Object($root.a[0], delay=2, history=[{{.+}}])}}
// expected-remark-re @below {{endPoint=Object($root.x[0]), startPoint=Object($root.a[1], delay=2, history=[{{.+}}])}}
// expected-remark-re @below {{endPoint=Object($root.x[0]), startPoint=Object($root.a[2], delay=1, history=[{{.+}}])}}
hw.module private @extract(in %a : i3, out x : i1) {
  %0 = comb.extract %a from 0 : (i3) -> i1
  %1 = comb.extract %a from 1 : (i3) -> i1
  %2 = comb.extract %a from 2 : (i3) -> i1
  %q = synth.aig.and_inv %0, %1 : i1 // q[0] := max(a[0], a[1]) + 1 = 1
  %r = synth.aig.and_inv %q, %2 : i1 // r[0] := max(q[0], a[2]) + 1 = 2
  hw.output %r : i1
}

// expected-remark-re @below {{endPoint=Object($root.x[0]), startPoint=Object($root.a[1], delay=0, history=[{{.+}})])}}
hw.module private @concat(in %a : i3, in %b : i1, out x : i1) {
  %0 = comb.concat %a, %b : i3, i1
  %1 = comb.extract %0 from 2 : (i4) -> i1
  hw.output %1 : i1
}

// expected-remark-re @below {{endPoint=Object($root.x[0]), startPoint=Object($root.a[1], delay=0, history=[{{.+}})])}}
hw.module private @replicate(in %a : i3, in %b : i1, out x : i1) {
  %0 = comb.replicate %a : (i3) -> i24
  %1 = comb.extract %0 from 13 : (i24) -> i1
  hw.output %1 : i1
}

// Make sure bound modules are skipped.
hw.module private @bound(in %a: i1) {
  %clock = seq.to_clock %a
  %r = seq.compreg %and, %clock : i1
  %and = synth.aig.and_inv %r, %a : i1
  hw.output
}

hw.module private @pass(in %a : i1, out x : i1) {
  %r = synth.aig.and_inv %a {sv.namehint = "foo"} : i1
  hw.output %r : i1
}

hw.module private @child(in %a : i1, in %b : i1, out x : i1) {
  %r = synth.aig.and_inv %a, %b : i1 // r[0] := max(a[0], b[0]) + 1 = 1
  %r1 = hw.instance "pass" @pass(a: %r: i1) -> (x: i1)
  hw.instance "bound" @bound(a: %a: i1) -> () {doNotPrint}
  hw.output %r1 : i1
}

// Check history.
// expected-remark @below {{endPoint=Object($root.x[0]), startPoint=Object($root.a[0], delay=2, history=[Object($root.c2.x[0], delay=2, comment="output port"), Object($root/c2:child.pass.x[0], delay=2, comment="output port"), Object($root/c2:child/pass:pass.foo[0], delay=2, comment="namehint"), Object($root/c2:child/pass:pass.a[0], delay=2, comment="input port"), Object($root/c2:child.a[0], delay=1, comment="input port"), Object($root.c1.x[0], delay=1, comment="output port"), Object($root/c1:child.pass.x[0], delay=1, comment="output port"), Object($root/c1:child/pass:pass.foo[0], delay=1, comment="namehint"), Object($root/c1:child/pass:pass.a[0], delay=1, comment="input port"), Object($root/c1:child.a[0], delay=0, comment="input port"), Object($root.a[0], delay=0, comment="input port")])}}
// expected-remark @below {{endPoint=Object($root.x[0]), startPoint=Object($root.b[0], delay=2, history=[Object($root.c2.x[0], delay=2, comment="output port"), Object($root/c2:child.pass.x[0], delay=2, comment="output port"), Object($root/c2:child/pass:pass.foo[0], delay=2, comment="namehint"), Object($root/c2:child/pass:pass.a[0], delay=2, comment="input port"), Object($root/c2:child.a[0], delay=1, comment="input port"), Object($root.c1.x[0], delay=1, comment="output port"), Object($root/c1:child.pass.x[0], delay=1, comment="output port"), Object($root/c1:child/pass:pass.foo[0], delay=1, comment="namehint"), Object($root/c1:child/pass:pass.a[0], delay=1, comment="input port"), Object($root/c1:child.b[0], delay=0, comment="input port"), Object($root.b[0], delay=0, comment="input port")])}}
hw.module private @parent(in %a : i1, in %b : i1, out x : i1) {
  %0 = hw.instance "c1" @child(a: %a: i1, b: %b: i1) -> (x: i1)
  %1 = hw.instance "c2" @child(a: %0: i1, b: %b: i1) -> (x: i1)
  hw.output %1 : i1
}

// expected-remark @below {{endPoint=Object($root.x[0]), startPoint=Object($root.r[0], delay=1)}}
// expected-remark-re @below {{endPoint=Object($root.x[0]), startPoint=Object($root.a[0], delay=1, history=[{{.+}}])}}
hw.module private @firreg(in %a : i1, in %clk : !seq.clock, out x : i1) {
  // expected-remark @below {{root=firreg, endPoint=Object($root.r[0]), startPoint=Object($root.r[0], delay=1)}}
  // expected-remark-re @below {{root=firreg, endPoint=Object($root.r[0]), startPoint=Object($root.a[0], delay=1, history=[{{.+}}])}}
  %r = seq.firreg %flip clock %clk : i1
  %flip = synth.aig.and_inv not %r, %a : i1
  hw.output %flip : i1
}

// expected-remark @below {{endPoint=Object($root.x[0]), startPoint=Object($root.r[0], delay=1)}}
// expected-remark-re @below {{endPoint=Object($root.x[0]), startPoint=Object($root.a[0], delay=1, history=[{{.+}}])}}
hw.module private @compreg(in %a : i1, in %clk : !seq.clock, out x : i1) {
  // expected-remark @below {{root=compreg, endPoint=Object($root.r[0]), startPoint=Object($root.r[0], delay=1)}}
  // expected-remark-re @below {{root=compreg, endPoint=Object($root.r[0]), startPoint=Object($root.a[0], delay=1, history=[{{.+}}])}}
  %r = seq.compreg %flip, %clk : i1
  %flip = synth.aig.and_inv not %r, %a : i1
  hw.output %flip : i1
}

// expected-remark-re @below {{endPoint=Object($root.x[0]), startPoint=Object($root.cond[0], delay=1, history=[{{.+}}])}}
// expected-remark-re @below {{endPoint=Object($root.x[0]), startPoint=Object($root.a[0], delay=1, history=[{{.+}}])}}
// expected-remark-re @below {{endPoint=Object($root.x[0]), startPoint=Object($root.b[0], delay=1, history=[{{.+}}])}}
// expected-remark-re @below {{endPoint=Object($root.y[0]), startPoint=Object($root.a[0], delay=1, history=[{{.+}}])}}
// expected-remark-re @below {{endPoint=Object($root.y[0]), startPoint=Object($root.b[0], delay=1, history=[{{.+}}])}}
hw.module private @comb(in %cond : i1, in %a : i1, in %b : i1, out x : i1, out y: i1) {
  %r = comb.mux %cond, %a, %b : i1
  %table = comb.truth_table %a, %b -> [true, false, false, true]
  hw.output %r, %table: i1, i1
}

// expected-remark-re @below {{endPoint=Object($root.x[0]), startPoint=Object($root.a[0], delay=2, history=[{{.+}}])}}
// expected-remark-re @below {{endPoint=Object($root.x[0]), startPoint=Object($root.b[0], delay=2, history=[{{.+}}])}}
hw.module private @fix_duplication(in %a : i1, in %b : i1, out x : i1) {
  %0 = synth.aig.and_inv %a, %a, %b : i1
  hw.output %0 : i1
}


// expected-remark-re @below {{endPoint=Object($root.x[0]), startPoint=Object($root.a[0], delay=3, history=[{{.+}}])}}
// expected-remark-re @below {{endPoint=Object($root.x[0]), startPoint=Object($root.a[1], delay=3, history=[{{.+}}])}}
// expected-remark-re @below {{endPoint=Object($root.x[0]), startPoint=Object($root.b[0], delay=3, history=[{{.+}}])}}
// expected-remark-re @below {{endPoint=Object($root.x[0]), startPoint=Object($root.b[1], delay=3, history=[{{.+}}])}}
hw.module private @comb_others(in %a : i2, in %b : i2, out x : i1) {
  %0 = comb.icmp eq %a, %b : i2
  hw.output %0 : i1
}

// expected-remark-re @below {{endPoint=Object($root.x[0]), startPoint=Object($root.a[0], delay=1, history=[{{.+}}])}}
// expected-remark-re @below {{endPoint=Object($root.y[0]), startPoint=Object($root.a[0], delay=2, history=[{{.+}}])}}
hw.module private @mig(in %a : i1, out x : i1, out y : i1) {
  %p = synth.mig.maj_inv %a, %a, %a : i1
  %q = synth.mig.maj_inv %a, %a, %a, %a, %a : i1
  hw.output %p, %q : i1, i1
}
