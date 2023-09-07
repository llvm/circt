om.class @Component_OMNode_0(
    %propIn_bore: !om.integer) {
  %0 = om.constant #om.list<!om.string,["MyThing" : !om.string]> : !om.list<!om.string>
  %1 = om.constant "Component.inst1.foo" : !om.string
  om.class.field @field1, %propIn_bore : !om.integer
  om.class.field @field2, %1 : !om.string
  om.class.field @omType, %0 : !om.list<!om.string>
}

om.class @Component_OMNode_1(
    %nodeIn: !om.class.type<@Component_OMNode_0>) {
  %0 = om.constant #om.integer<123> : !om.integer
  %1= om.constant #om.list<!om.string,["MyThing" : !om.string]> : !om.list<!om.string>
  om.class.field @field1, %nodeIn : !om.class.type<@Component_OMNode_0>
  om.class.field @field2, %0 : !om.integer
  om.class.field @omType, %1 : !om.list<!om.string>
}

om.class @Component_OMNode_2() {
  %0 = om.constant "blah" : !om.string
  om.class.field @field1, %0 : !om.string
}

om.class @Component_OMNode_3() {
  %0 = om.constant "blah" : !om.string
  om.class.field @field1, %0 : !om.string
}

om.class @Component_OMIR(
    %propIn_bore : !om.integer) {
  %0 = om.object @Component_OMNode_0(%propIn_bore) : (!om.integer) -> !om.class.type<@Component_OMNode_0>
  %1 = om.object @Component_OMNode_1(%0) : (!om.class.type<@Component_OMNode_0>) -> !om.class.type<@Component_OMNode_1>
  %2 = om.object @Component_OMNode_2() : () -> !om.class.type<@Component_OMNode_2>
  %3 = om.object @Component_OMNode_3() : () -> !om.class.type<@Component_OMNode_3>

  om.class.field @node0_out, %0 : !om.class.type<@Component_OMNode_0>
  om.class.field @node1_out, %1 : !om.class.type<@Component_OMNode_1>
  om.class.field @node2_out, %2 : !om.class.type<@Component_OMNode_2>
  om.class.field @node3_out, %3 : !om.class.type<@Component_OMNode_3>
}

om.class @Component(
    %propIn: !om.integer) {
  %0 = om.object @Component_OMIR(%propIn) : (!om.integer) -> !om.class.type<@Component_OMIR>

  %1 = om.object.field %0, [@node1_out] : (!om.class.type<@Component_OMIR>) -> !om.class.type<@Component_OMNode_1>

  om.class.field @propOut, %1 : !om.class.type<@Component_OMNode_1>
  om.class.field @OmirOut, %0 : !om.class.type<@Component_OMIR>
}

om.class @Client_OMNode_0(
    %propIn_bore: !om.integer,
    %inst1_propOut_bore: !om.class.type<@Component_OMNode_1>,
    %inst2_propOut_bore: !om.class.type<@Component_OMNode_1>) {
  om.class.field @field1, %propIn_bore : !om.integer
  om.class.field @field2, %inst1_propOut_bore : !om.class.type<@Component_OMNode_1>
  om.class.field @field3, %inst2_propOut_bore : !om.class.type<@Component_OMNode_1>
}

om.class @Client_OMIR(
    %propIn_bore : !om.integer,
    %inst1_propOut_bore: !om.class.type<@Component_OMNode_1>,
    %inst2_propOut_bore: !om.class.type<@Component_OMNode_1>) {
  %0 = om.object @Client_OMNode_0(%propIn_bore, %inst1_propOut_bore, %inst1_propOut_bore) : (!om.integer, !om.class.type<@Component_OMNode_1>, !om.class.type<@Component_OMNode_1>) -> !om.class.type<@Client_OMNode_0>

  om.class.field @node0_out, %0 : !om.class.type<@Client_OMNode_0>
}

om.class  @Client(
    %propIn: !om.integer) {
  %0 = om.constant #om.integer<0> : !om.integer
  %1 = om.constant #om.integer<1> : !om.integer

  %inst1 = om.object @Component(%0) : (!om.integer) -> !om.class.type<@Component>
  %inst2 = om.object @Component(%1) : (!om.integer) -> !om.class.type<@Component>

  %inst1.propOut = om.object.field %inst1, [@propOut] : (!om.class.type<@Component>) -> !om.class.type<@Component_OMNode_1>
  %inst1.OmirOut = om.object.field %inst1, [@OmirOut] : (!om.class.type<@Component>) -> !om.class.type<@Component_OMIR>
  %inst2.propOut = om.object.field %inst2, [@propOut] : (!om.class.type<@Component>) -> !om.class.type<@Component_OMNode_1>
  %inst2.OmirOut = om.object.field %inst2, [@OmirOut] : (!om.class.type<@Component>) -> !om.class.type<@Component_OMIR>

  %2 = om.object @Client_OMIR(%propIn, %inst1.propOut, %inst2.propOut) : (!om.integer, !om.class.type<@Component_OMNode_1>, !om.class.type<@Component_OMNode_1>) -> !om.class.type<@Client_OMIR>

  %3 = om.constant #om.integer<456> : !om.integer

  om.class.field @propOut, %3 : !om.integer
  om.class.field @OmirOut, %2 : !om.class.type<@Client_OMIR>
  om.class.field @inst1OmirOut, %inst1.OmirOut : !om.class.type<@Component_OMIR>
  om.class.field @inst2OmirOut, %inst2.OmirOut : !om.class.type<@Component_OMIR>
}

