module {
  om.class @A(%arg: i1) {
  }
  om.class @Conflict(){}
  hw.module private @M2() {}
  hw.module @M1(in %a: i1) {
    hw.instance "" @M2() -> ()
  }
}
