firrtl.circuit "NullaryCat" { 
  // Make sure it doesn't crash.
  firrtl.module @NullaryCat() {
    %0 = firrtl.cat : () -> !firrtl.uint<0>
  }
}