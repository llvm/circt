// RUN: circt-opt --firrtl-inliner %s
// XFAIL: *

// Inliner does not support running before expand when's,
// here was crash the reference-handling code because it assumes
// the instance and its uses are in the same block.

module {
  firrtl.circuit "InlinerRefs" {
    firrtl.module private @ChildOut(in %in: !firrtl.bundle<a: uint<1>, b: uint<2>>, out %out: !firrtl.ref<bundle<a: uint<1>, b: uint<2>>>) attributes {annotations = [{class = "firrtl.passes.InlineAnnotation"}]} {
      %0 = firrtl.subfield %in[a] : !firrtl.bundle<a: uint<1>, b: uint<2>>
      firrtl.when %0 : !firrtl.uint<1> {
        %1 = firrtl.ref.send %in : !firrtl.bundle<a: uint<1>, b: uint<2>>
        firrtl.ref.define %out, %1 : !firrtl.ref<bundle<a: uint<1>, b: uint<2>>>
      }
    }
    firrtl.module @InlinerRefs(in %in: !firrtl.bundle<a: uint<1>, b: uint<2>>, out %out: !firrtl.uint<1>) {
      %0 = firrtl.subfield %in[a] : !firrtl.bundle<a: uint<1>, b: uint<2>>
      %co_in, %co_out = firrtl.instance co interesting_name @ChildOut(in in: !firrtl.bundle<a: uint<1>, b: uint<2>>, out out: !firrtl.ref<bundle<a: uint<1>, b: uint<2>>>)
      %1 = firrtl.ref.sub %co_out[0] : !firrtl.ref<bundle<a: uint<1>, b: uint<2>>>
      firrtl.strictconnect %co_in, %in : !firrtl.bundle<a: uint<1>, b: uint<2>>
      firrtl.when %0 : !firrtl.uint<1> {
        %2 = firrtl.ref.resolve %1 : !firrtl.ref<uint<1>>
        firrtl.strictconnect %out, %2 : !firrtl.uint<1>
      }
    }
  }
}
