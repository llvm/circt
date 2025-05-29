// RUN: circt-opt --pass-pipeline="builtin.module(hw-expunge-module{modules={baz,b} port-prefixes={foo:bar2.baz2=meow_,bar:baz1=nya_}})" %s | FileCheck %s --check-prefixes FOO,BAR,BAZ,COMMON
// RUN: circt-opt --pass-pipeline="builtin.module(hw-expunge-module{modules={baz,b} port-prefixes={foo:bar2.baz2=meow_,bar:baz1=nya_}},hw-tree-shake{keep=foo})" %s | FileCheck %s --check-prefixes FOO,BAR,COMMON
// RUN: circt-opt --pass-pipeline="builtin.module(hw-expunge-module{modules={baz,b} port-prefixes={foo:bar2.baz2=meow_,bar:baz1=nya_}},hw-tree-shake{keep=baz})" %s | FileCheck %s --check-prefixes BAZ,COMMON

module {
  hw.module @foo(in %bar1_baz1__out : i2, out test : i1) {
    %bar1.self_out = hw.instance "bar1" @bar(self_in: %0: i1) -> (self_out: i1)
    %bar2.self_out = hw.instance "bar2" @bar(self_in: %bar1.self_out: i1) -> (self_out: i1)
    %0 = comb.extract %bar1_baz1__out from 0 : (i2) -> i1
    hw.output %bar2.self_out : i1
  }
  hw.module private @bar(in %self_in : i1, out self_out : i1) {
    %baz1.out = hw.instance "baz1" @baz(in: %self_in: i1) -> (out: i1)
    %baz2.out = hw.instance "baz2" @baz(in: %baz1.out: i1) -> (out: i1)
    hw.output %baz2.out : i1
  }
  hw.module private @baz(in %in : i1, out out : i1) {
    hw.output %in : i1
  }
}

// COMMON: module {
// FOO-NEXT:  hw.module @foo(in %bar1_baz1__out : i2, in %bar1_baz1__out_0 : i1, in %bar1_baz2__out : i1, in %bar2_baz1__out : i1, in %meow_out : i1, out test : i1, out bar1_baz1__in : i1, out bar1_baz2__in : i1, out bar2_baz1__in : i1, out meow_in : i1) {
// FOO-NEXT:    %bar1.self_out, %bar1.nya_in, %bar1.baz2__in = hw.instance "bar1" @bar(self_in: %0: i1, nya_out: %bar1_baz1__out_0: i1, baz2__out: %bar1_baz2__out: i1) -> (self_out: i1, nya_in: i1, baz2__in: i1)
// FOO-NEXT:    %bar2.self_out, %bar2.nya_in, %bar2.baz2__in = hw.instance "bar2" @bar(self_in: %bar1.self_out: i1, nya_out: %bar2_baz1__out: i1, baz2__out: %meow_out: i1) -> (self_out: i1, nya_in: i1, baz2__in: i1)
// FOO-NEXT:    %0 = comb.extract %bar1_baz1__out from 0 : (i2) -> i1
// FOO-NEXT:    hw.output %bar2.self_out, %bar1.nya_in, %bar1.baz2__in, %bar2.nya_in, %bar2.baz2__in : i1, i1, i1, i1, i1
// FOO-NEXT: }
// BAR-NEXT:  hw.module private @bar(in %self_in : i1, in %nya_out : i1, in %baz2__out : i1, out self_out : i1, out nya_in : i1, out baz2__in : i1) {
// BAR-NEXT:    hw.output %baz2__out, %self_in, %nya_out : i1, i1, i1
// BAR-NEXT:  }
// BAZ-NEXT:  hw.module private @baz(in %in : i1, out out : i1) {
// BAZ-NEXT:    hw.output %in : i1
// BAZ-NEXT:  }
