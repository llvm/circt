module {
  om.class.extern @Child() -> (cond: i1) {}

  om.class @Top() -> (cond: i1) {
    %child = om.object @Child() : () -> !om.class.type<@Child>
    %cond = om.object.field %child["cond"] : (!om.class.type<@Child>) -> i1
    %message = om.constant "linked child condition must hold" : !om.string
    om.property_assert %cond, %message : i1
    om.class.fields %cond : i1
  }
}
