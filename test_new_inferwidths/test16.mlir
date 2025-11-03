firrtl.circuit "MuxBundleOperands" {
  firrtl.module @MuxBundleOperands(in %a: !firrtl.bundle<a: uint<8>>, in %p: !firrtl.uint<1>, out %c: !firrtl.bundle<a: uint>) {
    // CHECK: %w = firrtl.wire  : !firrtl.bundle<a: uint<8>>
    %w = firrtl.wire  : !firrtl.bundle<a: uint>
    %0 = firrtl.subfield %w[a] : !firrtl.bundle<a: uint>
    %1 = firrtl.subfield %a[a] : !firrtl.bundle<a: uint<8>>
    firrtl.connect %0, %1 : !firrtl.uint, !firrtl.uint<8>
    // CHECK: %2 = firrtl.mux(%p, %a, %w) : (!firrtl.uint<1>, !firrtl.bundle<a: uint<8>>, !firrtl.bundle<a: uint<8>>) -> !firrtl.bundle<a: uint<8>>
    %2 = firrtl.mux(%p, %a, %w) : (!firrtl.uint<1>, !firrtl.bundle<a: uint<8>>, !firrtl.bundle<a: uint>) -> !firrtl.bundle<a: uint>
    firrtl.connect %c, %2 : !firrtl.bundle<a: uint>, !firrtl.bundle<a: uint>
  }
}