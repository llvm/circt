module {
  hw.hierpath private @xmr [@M1::@s1, @M2]
  om.class @A(%arg: i1) {
  }
  om.class @Conflict(){}
  hw.module private @M2() {}
  hw.module @M1(in %a: i1) {

  sv.always posedge %a {
    sv.if %a {
      hw.instance "" sym @s1 @M2() -> ()
    }
  }
  }
}
