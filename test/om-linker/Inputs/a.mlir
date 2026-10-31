module {
  om.class @A(%arg: i1) {
    om.class.fields
  }
  om.class @Conflict() -> (items: !om.list<!om.class.type<@Conflict>>) {
    %items = om.constant #om.list<!om.class.type<@Conflict>, []> : !om.list<!om.class.type<@Conflict>>
    om.class.fields %items : !om.list<!om.class.type<@Conflict>>
  }
  hw.module @hello() {}
}
