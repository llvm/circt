firrtl.circuit "Issue1110" {
  firrtl.module @Issue1110(in %x: !firrtl.uint<0>, out %y: !firrtl.uint) {
    firrtl.connect %y, %x : !firrtl.uint, !firrtl.uint<0>
  }
}