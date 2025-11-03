firrtl.circuit "Issue1118" {
  firrtl.module @Issue1118(out %x: !firrtl.sint) {
    %c4232_ui = firrtl.constant 4232 : !firrtl.uint
    %0 = firrtl.asSInt %c4232_ui : (!firrtl.uint) -> !firrtl.sint
    firrtl.connect %x, %0 : !firrtl.sint, !firrtl.sint
  }
}