module {
  om.class.extern @A(%arg: i1) {}
  om.class @B(%arg: i2) {
    om.class.fields
  }
  om.class @Conflict(){
    om.class.fields
  }
  hw.module.extern @hello()
}
