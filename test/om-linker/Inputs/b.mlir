module {
  hw.module.extern @M1(in %a: i1)
  hw.module private @M2(in %a: i1) {
    hw.instance "" @M1(a: %a : i1) -> ()
  }
  hw.module @xmr () {}
  hw.module public @Top (in %a: i1){
    hw.instance "" @M2(a: %a: i1) -> ()
  }
  om.class.extern @A(%arg: i1) {
  }
  om.class @B(%arg: i2) {
  }
  om.class @Conflict(){}
}
