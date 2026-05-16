module {
  om.class.extern @Child() -> (cond: i1) {}

  om.class @Top() -> (cond: i1) {
    %child = om.object @Child() : () -> !om.class.type<@Child>
    %cond = om.object.field %child["cond"] : (!om.class.type<@Child>) -> i1
    om.property_assert %cond, "linked child condition must hold" : i1
    om.class.fields %cond : i1
  }
}
