firrtl.circuit "InferOutput" {
  firrtl.module @InferOutput(in %in: !firrtl.uint<2>, out %out: !firrtl.uint) {
    firrtl.connect %out, %in : !firrtl.uint, !firrtl.uint<2>
  }
}