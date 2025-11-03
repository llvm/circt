firrtl.circuit "AttachOne" {
  firrtl.module @AttachOne(in %a0: !firrtl.analog<8>) {
    firrtl.attach %a0 : !firrtl.analog<8>
  }
}