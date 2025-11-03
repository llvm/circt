firrtl.circuit "InferBundlePort" {
  firrtl.module @InferBundlePort(in %in: !firrtl.bundle<a: uint<2>, b: uint<3>>, out %out: !firrtl.bundle<a: uint, b: uint>) {
    // CHECK: firrtl.connect %out, %in : !firrtl.bundle<a: uint<2>, b: uint<3>>
    firrtl.connect %out, %in : !firrtl.bundle<a: uint, b: uint>, !firrtl.bundle<a: uint<2>, b: uint<3>>
  }
}