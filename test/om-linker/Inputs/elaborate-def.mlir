module {
  om.class @Child() -> (cond: i1) {
    %false = om.constant false
    om.class.fields %false : i1
  }
}
