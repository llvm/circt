firrtl.circuit "InferOutput2" {
  firrtl.module @InferOutput2(in %in: !firrtl.uint<2>, out %out: !firrtl.uint) {
    firrtl.connect %out, %in : !firrtl.uint, !firrtl.uint<2>
  }
}