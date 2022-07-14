// For now, just checking very basic cases parse properly.

// Lots of checking around RefType likely warranted:
// * firrtl.ref<uint> -- handling?
// 
// Errors:
// * nested ref
// * Use anywhere other than handful of approved places
// * use in an aggregate type (bundle/vector/etc)

// RUN: circt-opt %s -split-input-file

firrtl.circuit "xmr" {
  firrtl.module private @Test(in %x: !firrtl.ref<uint<2>>) {
    %e = firrtl.xmr.end %x : !firrtl.ref<uint<2>>
  }
  firrtl.module @xmr() {
    %test_x = firrtl.instance test @Test(in x: !firrtl.ref<uint<2>>)
    %x = firrtl.xmr.get %test_x : !firrtl.ref<uint<2>>
  }
}
