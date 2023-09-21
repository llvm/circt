// RUN: circt-opt -firrtl-lower-classes %s | FileCheck %s

firrtl.circuit "Component" {
  // CHECK-LABEL: om.class @Class_0
  // CHECK-SAME: %[[REF1:.+]]: !om.class.type<@Class_1>
  firrtl.class private @Class_0(in %someReference_in: !firrtl.class<@Class_1(out someInt: !firrtl.integer)>, out %someReference: !firrtl.class<@Class_1(out someInt: !firrtl.integer)>) {
    // CHECK: om.class.field @someReference, %[[REF1]]
    firrtl.propassign %someReference, %someReference_in : !firrtl.class<@Class_1(out someInt: !firrtl.integer)>
  }

  // CHECK-LABEL: om.class @Class_1
  firrtl.class private @Class_1(out %someInt: !firrtl.integer) {
    // CHECK: %[[C1:.+]] = om.constant #om.integer<1 : si4> : !om.integer
    %0 = firrtl.integer 1
    // CHECK: om.class.field @someInt, %[[C1]]
    firrtl.propassign %someInt, %0 : !firrtl.integer
  }

  // CHECK-LABEL: om.class @Class_2
  firrtl.class private @Class_2(out %someString: !firrtl.string) {
    // CHECK: %[[C2:.+]] = om.constant "fubar" : !om.string
    %0 = firrtl.string "fubar"
    // CHECK: om.class.field @someString, %[[C2]]
    firrtl.propassign %someString, %0 : !firrtl.string
  }

  // CHECK-LABEL: om.class.extern @ExtClass(%input: !om.string) {
  // CHECK-NEXT:    om.class.extern.field @field : !om.string
  // CHECK-NEXT:  }
  firrtl.extclass private @ExtClass(in input: !firrtl.string, out field: !firrtl.string)

  // CHECK-LABEL: om.class @ClassEntrypoint
  firrtl.class private @ClassEntrypoint(out %obj_0_out: !firrtl.class<@Class_1(out someInt: !firrtl.integer)>) {
    // CHECK: %[[OBJ1:.+]] = om.object @Class_1() : () -> !om.class.type<@Class_1>
    %obj1 = firrtl.object @Class_1(out someInt: !firrtl.integer)

    // CHECK: %[[OBJ0:.+]] = om.object @Class_0(%[[OBJ1]]) : (!om.class.type<@Class_1>) -> !om.class.type<@Class_0>
    %obj0 = firrtl.object @Class_0(in someReference_in: !firrtl.class<@Class_1(out someInt: !firrtl.integer)>, out someReference: !firrtl.class<@Class_1(out someInt: !firrtl.integer)>)
    %obj0_someReference_in = firrtl.object.subfield %obj0[someReference_in] : !firrtl.class<@Class_0(in someReference_in: !firrtl.class<@Class_1(out someInt: !firrtl.integer)>, out someReference: !firrtl.class<@Class_1(out someInt: !firrtl.integer)>)>
    firrtl.propassign %obj0_someReference_in, %obj1 : !firrtl.class<@Class_1(out someInt: !firrtl.integer)>

    // CHECK: %[[REF:.+]] = om.object.field %[[OBJ0]], [@someReference] : (!om.class.type<@Class_0>) -> !om.class.type<@Class_1>
    // CHECK: om.class.field @obj_0_out, %[[REF]] : !om.class.type<@Class_1>
    %obj0_someReference = firrtl.object.subfield %obj0[someReference] : !firrtl.class<@Class_0(in someReference_in: !firrtl.class<@Class_1(out someInt: !firrtl.integer)>, out someReference: !firrtl.class<@Class_1(out someInt: !firrtl.integer)>)>
    firrtl.propassign %obj_0_out, %obj0_someReference: !firrtl.class<@Class_1(out someInt: !firrtl.integer)>
  }

  // CHECK-LABEL: om.class @ReadOutputPort()
  firrtl.class @ReadOutputPort(out %output : !firrtl.integer) {
    // CHECK: %[[OBJ:.+]] = om.object @Class_1() : () -> !om.class.type<@Class_1>
    // CHECK: %[[FIELD:.+]] = om.object.field %[[OBJ]], [@someInt] : (!om.class.type<@Class_1>) -> !om.integer
    // CHECK: om.class.field @output, %[[FIELD]] : !om.integer
    %obj = firrtl.object @Class_1(out someInt: !firrtl.integer)
    %0 = firrtl.object.subfield %obj[someInt] : !firrtl.class<@Class_1(out someInt: !firrtl.integer)>
    firrtl.propassign %output, %0 : !firrtl.integer
  }

  firrtl.class @TwoInputs(in %a: !firrtl.integer, in %b: !firrtl.integer) { }

  firrtl.class @AssignInputsInOrder() {
    // CHECK: %0 = om.constant #om.integer<123 : si12> : !om.integer
    // CHECK: %1 = om.constant #om.integer<456 : si12> : !om.integer
    // CHECK: %2 = om.object @TwoInputs(%0, %1) : (!om.integer, !om.integer) -> !om.class.type<@TwoInputs>
    %x = firrtl.integer 123
    %y = firrtl.integer 456
    %obj = firrtl.object @TwoInputs(in a: !firrtl.integer, in b: !firrtl.integer)

    %obj_a = firrtl.object.subfield %obj[a] : !firrtl.class<@TwoInputs(in a: !firrtl.integer, in b: !firrtl.integer)>
    firrtl.propassign %obj_a, %x : !firrtl.integer

    %obj_b = firrtl.object.subfield %obj[b] : !firrtl.class<@TwoInputs(in a: !firrtl.integer, in b: !firrtl.integer)>
    firrtl.propassign %obj_b, %y : !firrtl.integer
  }

  firrtl.class @AssignInputsOutOfOrder() {
    // CHECK: %0 = om.constant #om.integer<123 : si12> : !om.integer
    // CHECK: %1 = om.constant #om.integer<456 : si12> : !om.integer
    // CHECK: %2 = om.object @TwoInputs(%0, %1) : (!om.integer, !om.integer) -> !om.class.type<@TwoInputs>
    %x = firrtl.integer 123
    %y = firrtl.integer 456
    %obj = firrtl.object @TwoInputs(in a: !firrtl.integer, in b: !firrtl.integer)

    %obj_b = firrtl.object.subfield %obj[b] : !firrtl.class<@TwoInputs(in a: !firrtl.integer, in b: !firrtl.integer)>
    firrtl.propassign %obj_b, %y : !firrtl.integer

    %obj_a = firrtl.object.subfield %obj[a] : !firrtl.class<@TwoInputs(in a: !firrtl.integer, in b: !firrtl.integer)>
    firrtl.propassign %obj_a, %x : !firrtl.integer
  }

  firrtl.module @Component(in %input: !firrtl.uint<1>, out %output: !firrtl.uint<1>, out %omir_out: !firrtl.class<@ClassEntrypoint(out obj_0_out: !firrtl.class<@Class_1(out someInt: !firrtl.integer)>)>) attributes {convention = #firrtl<convention scalarized>} {
    %0 = firrtl.object @ClassEntrypoint(out obj_0_out: !firrtl.class<@Class_1(out someInt: !firrtl.integer)>)
    firrtl.propassign %omir_out, %0 : !firrtl.class<@ClassEntrypoint(out obj_0_out: !firrtl.class<@Class_1(out someInt: !firrtl.integer)>)>
    firrtl.strictconnect %output, %input : !firrtl.uint<1>
  }

  // CHECK-LABEL: om.class @ClassTest
  firrtl.class @ClassTest() {}

  // CHECK-LABEL: om.class @ListTest(
  // CHECK-SAME:    %s1: !om.string
  // CHECK-SAME:    %s2: !om.string
  // CHECK-SAME:    %c1: !om.class.type<@ClassTest>
  // CHECK-SAME:    %c2: !om.class.type<@ClassTest>) {
  firrtl.class @ListTest(in %s1: !firrtl.string,
                         in %s2: !firrtl.string,
                         in %c1: !firrtl.class<@ClassTest()>,
                         in %c2: !firrtl.class<@ClassTest()>,
                         out %out_strings: !firrtl.list<string>,
                         out %out_empty: !firrtl.list<string>,
                         out %out_nested: !firrtl.list<list<string>>,
                         out %out_objs: !firrtl.list<class<@ClassTest()>>) {
    // List of basic property types (strings)
    // CHECK-NEXT: %[[STRINGS:.+]] = om.list_create %s1, %s2 : !om.string
    %strings = firrtl.list.create %s1, %s2 : !firrtl.list<string>
    firrtl.propassign %out_strings, %strings : !firrtl.list<string>

    // Empty list
    // CHECK-NEXT: %[[EMPTY:.+]] = om.list_create : !om.string
    %empty = firrtl.list.create : !firrtl.list<string>
    firrtl.propassign %out_empty, %empty : !firrtl.list<string>

    // Nested list
    // CHECK-NEXT: %[[NESTED:.+]] = om.list_create %[[STRINGS]], %[[EMPTY]] : !om.list<!om.string>
    %nested = firrtl.list.create %strings, %empty : !firrtl.list<list<string>>
    firrtl.propassign %out_nested, %nested: !firrtl.list<list<string>>

    // List of objects
    // CHECK-NEXT: %[[OBJS:.+]] = om.list_create %c1, %c2 : !om.class.type<@ClassTest>
    %objs = firrtl.list.create %c1, %c2 : !firrtl.list<class<@ClassTest()>>
    firrtl.propassign %out_objs, %objs : !firrtl.list<class<@ClassTest()>>

    // CHECK-NEXT: om.class.field @out_strings, %[[STRINGS]] : !om.list<!om.string>
    // CHECK-NEXT: om.class.field @out_empty, %[[EMPTY]] : !om.list<!om.string>
    // CHECK-NEXT: om.class.field @out_nested, %[[NESTED]] : !om.list<!om.list<!om.string>>
    // CHECK-NEXT: om.class.field @out_objs, %[[OBJS]] : !om.list<!om.class.type<@ClassTest>
  }

  // CHECK-LABEL: om.class @BoolTest()
  firrtl.class @BoolTest(out %b : !firrtl.bool) {
    // CHECK-NEXT: %[[TRUE:.+]] = om.constant true
    // CHECK-NEXT: om.class.field @b, %[[TRUE]] : i1
    %true = firrtl.bool true
    firrtl.propassign %b, %true : !firrtl.bool
  }

  // CHECK-LABEL: om.class @DoubleTest()
  firrtl.class @DoubleTest(out %d : !firrtl.double) {
    // CHECK-NEXT: %[[DBL:.+]] = om.constant 4.0{{0*[eE]}}-01 : f64
    // CHECK-NEXT: om.class.field @d, %[[DBL]] : f64
    %dbl = firrtl.double 0.4
    firrtl.propassign %d, %dbl: !firrtl.double
  }

  // CHECK-LABEL: om.class @MapTest
  firrtl.class @MapTest(in %s1: !firrtl.string,
                        in %s2: !firrtl.string,
                        in %c1: !firrtl.class<@ClassTest()>,
                        in %c2: !firrtl.class<@ClassTest()>,
                        out %out_strings: !firrtl.map<string, string>,
                        out %out_empty: !firrtl.map<string, string>,
                        out %out_nested: !firrtl.map<string, map<string, string>>,
                        out %out_objs: !firrtl.map<string, class<@ClassTest()>>) {
    // Map of basic property types (strings)
    // CHECK-NEXT: %[[TUPLE1:.+]] = om.tuple_create %s1, %s1
    // CHECK-NEXT: %[[TUPLE2:.+]] = om.tuple_create %s2, %s2
    // CHECK-NEXT: %[[STRINGS:.+]] = om.map_create %[[TUPLE1]], %[[TUPLE2]]
    %strings = firrtl.map.create (%s1 -> %s1, %s2 -> %s2) : !firrtl.map<string, string>
    firrtl.propassign %out_strings, %strings : !firrtl.map<string, string>

    // Empty map
    // CHECK-NEXT: %[[EMPTY:.+]] = om.map_create : !om.string, !om.string
    %empty = firrtl.map.create : !firrtl.map<string, string>
    firrtl.propassign %out_empty, %empty : !firrtl.map<string, string>

    // Nested map
    // CHECK-NEXT: %[[TUPLE1:.+]] = om.tuple_create %s1, %[[STRINGS]]
    // CHECK-NEXT: %[[TUPLE2:.+]] = om.tuple_create %s2, %[[EMPTY]]
    // CHECK-NEXT: %[[NESTED:.+]] = om.map_create %[[TUPLE1]], %[[TUPLE2]]
    %nested = firrtl.map.create (%s1 -> %strings, %s2 -> %empty) : !firrtl.map<string, map<string, string>>
    firrtl.propassign %out_nested, %nested: !firrtl.map<string, map<string, string>>

    // Map of objects
    // CHECK-NEXT: %[[TUPLE1:.+]] = om.tuple_create %s1, %c1
    // CHECK-NEXT: %[[TUPLE2:.+]] = om.tuple_create %s2, %c2
    // CHECK-NEXT: %[[OBJS:.+]] = om.map_create %[[TUPLE1]], %[[TUPLE2]]
    %objs = firrtl.map.create (%s1 -> %c1, %s2 -> %c2) : !firrtl.map<string, class<@ClassTest()>>
    firrtl.propassign %out_objs, %objs : !firrtl.map<string, class<@ClassTest()>>

    // CHECK-NEXT: om.class.field @out_strings, %[[STRINGS]] : !om.map<!om.string, !om.string>
    // CHECK-NEXT: om.class.field @out_empty, %[[EMPTY]] : !om.map<!om.string, !om.string>
    // CHECK-NEXT: om.class.field @out_nested, %[[NESTED]] : !om.map<!om.string, !om.map<!om.string, !om.string>>
    // CHECK-NEXT: om.class.field @out_objs, %[[OBJS]] : !om.map<!om.string, !om.class.type<@ClassTest>>
  }
}

// CHECK-LABEL: firrtl.circuit "PathModule"
firrtl.circuit "PathModule" {
  // CHECK: hw.hierpath private [[PORT_PATH:@.+]] [@PathModule::[[PORT_SYM:@.+]]]
  // CHECK: hw.hierpath private [[WIRE_PATH:@.+]] [@PathModule::[[WIRE_SYM:@.+]]]
  // CHECK: hw.hierpath private [[VECTOR_PATH:@.+]] [@PathModule::[[VECTOR_SYM:@.+]]]
  // CHECK: hw.hierpath private [[INST_PATH:@.+]] [@PathModule::@child]
  // CHECK: hw.hierpath private [[MODULE_PATH:@.+]] [@Child]
  // CHECK: hw.hierpath private [[NONLOCAL_PATH:@.+]] [@PathModule::@child, @Child::[[NONLOCAL_SYM:@.+]]]

  // CHECK: firrtl.module @PathModule(in %in: !firrtl.uint<1> sym [[PORT_SYM]]) {
  firrtl.module @PathModule(in %in : !firrtl.uint<1> [{class = "circt.tracker", id = distinct[0]<>}]) {
    // CHECK: %wire = firrtl.wire sym [[WIRE_SYM]] : !firrtl.uint<8>
    %wire = firrtl.wire {annotations = [{class = "circt.tracker", id = distinct[1]<>}]} : !firrtl.uint<8>
    // CHECK: %vector = firrtl.wire sym [<[[VECTOR_SYM]],1,public>] : !firrtl.vector<uint<8>, 1>
    %vector = firrtl.wire {annotations = [{circt.fieldID = 1 : i32, class = "circt.tracker", id = distinct[2]<>}]} : !firrtl.vector<uint<8>, 1>
    // CHECK: firrtl.instance child sym @child @Child()
    firrtl.instance child sym @child {annotations = [{class = "circt.tracker", id = distinct[4]<>}]} @Child()
  }
  hw.hierpath @NonLocal [@PathModule::@child, @Child]
  // CHECK: firrtl.module @Child() {
  firrtl.module @Child() attributes {annotations = [{class = "circt.tracker", id = distinct[5]<>}]} {
    // CHECK: %non_local = firrtl.wire sym [[NONLOCAL_SYM]] : !firrtl.uint<8>
    %non_local = firrtl.wire {annotations = [{circt.nonlocal = @NonLocal, class = "circt.tracker", id = distinct[3]<>}]} : !firrtl.uint<8>
  }
  // CHECK: om.class @PathTest
  firrtl.class @PathTest() {
    
    // CHECK: om.path reference [[PORT_PATH]]
    %port_path = firrtl.path reference distinct[0]<>

    // CHECK: om.constant #om.path<"OMDeleted"> : !om.path
    %deleted_path = firrtl.path reference distinct[99]<>

    // CHECK: om.path reference [[WIRE_PATH]]
    // CHECK: om.path member_reference [[WIRE_PATH]]
    // CHECK: om.path member_reference [[WIRE_PATH]]
    // CHECK: om.path dont_touch [[WIRE_PATH]]
    %wire_reference = firrtl.path reference distinct[1]<>
    %wire_member_instance = firrtl.path member_instance distinct[1]<>
    %wire_member_reference = firrtl.path member_reference distinct[1]<>
    %wire_dont_touch = firrtl.path dont_touch distinct[1]<>

    // CHECK: om.path reference [[VECTOR_PATH]]
    %vector_reference = firrtl.path reference distinct[2]<>

    // CHECK: om.path reference [[NONLOCAL_PATH]]
    %non_local_path = firrtl.path reference distinct[3]<>

    // CHECK: om.path member_instance [[INST_PATH]]
    // CHECK: om.path member_instance [[INST_PATH]]
    // CHECK: om.path instance [[INST_PATH]]
    %instance_member_instance = firrtl.path member_instance distinct[4]<>
    %instance_member_reference = firrtl.path member_reference distinct[4]<>
    %instance = firrtl.path instance distinct[4]<>

    // CHECK: om.path reference [[MODULE_PATH]]
    %module_path = firrtl.path reference distinct[5]<>
  }
  firrtl.module @ListCreate(in %propIn: !firrtl.integer, out %propOut: !firrtl.list<integer>) attributes {convention = #firrtl<convention scalarized>} {
    %0 = firrtl.integer 123
    %1 = firrtl.list.create %propIn, %0 : !firrtl.list<integer>
    firrtl.propassign %propOut, %1 : !firrtl.list<integer>
    // CHECK:  %[[c0:.+]] = om.constant #om.integer<123 : si12> : !om.integer
    // CHECK:  %[[c1:.+]] = om.list_create %propIn, %[[c0]] : !om.integer
    // CHECK:  om.class.field @propOut, %[[c1]] : !om.list<!om.integer>
  }
}

// CHECK-LABEL: firrtl.circuit "WireProp"
firrtl.circuit "WireProp" {
  // CHECK: om.class @WireProp
  // CHECK-SAME: %[[IN:.+]]: !om.string
  firrtl.module @WireProp(in %in: !firrtl.string, out %out: !firrtl.string) attributes {convention = #firrtl<convention scalarized>} {
    // CHECK-NOT: firrtl.wire
    // CHECK-NOT: firrtl.propassign
    // CHECK: om.class.field @out, %[[IN]] : !om.string
    %s = firrtl.wire : !firrtl.string
    firrtl.propassign %s, %in : !firrtl.string
    firrtl.propassign %out, %s : !firrtl.string
  }
}

// CHECK-LABEL: firrtl.circuit "PublicModule"
firrtl.circuit "PublicModule" {
  // CHECK-NOT: om.class @PrivateModule
  firrtl.module private @PrivateModule() {}

  // CHECK-NOT: om.class @PrivateExtModule
  firrtl.extmodule private @PrivateExtModule()

  // CHECK: om.class @PublicModule
  firrtl.module @PublicModule() {}

  // CHECK: om.class.extern @PublicExtModule
  firrtl.extmodule @PublicExtModule()
}

// CHECK-LABEL: firrtl.circuit "ModuleInstances"
firrtl.circuit "ModuleInstances" {
  // CHECK: firrtl.extmodule private @ExtModule(in inputWire: !firrtl.uint<1>, out outputWire: !firrtl.uint<1>)
  firrtl.extmodule private @ExtModule(in inputWire: !firrtl.uint<1>, in inputProp: !firrtl.string, out outputWire: !firrtl.uint<1>, out outputProp: !firrtl.string)

  // CHECK: firrtl.module private @Module(in %[[IN_WIRE0:.+]]: !firrtl.uint<1>, out %[[OUT_WIRE0:.+]]: !firrtl.uint<1>)
  firrtl.module private @Module(in %inputWire: !firrtl.uint<1>, in %inputProp: !firrtl.string, out %outputWire: !firrtl.uint<1>, out %outputProp: !firrtl.string) {
    // CHECK: firrtl.strictconnect %[[OUT_WIRE0]], %[[IN_WIRE0]]
    firrtl.strictconnect %outputWire, %inputWire : !firrtl.uint<1>
    // CHECK-NEXT: }
    firrtl.propassign %outputProp, %inputProp : !firrtl.string
  }

  // CHECK: firrtl.module @ModuleInstances(in %[[IN_WIRE1:.+]]: !firrtl.uint<1>, out %[[OUT_WIRE1:.+]]: !firrtl.uint<1>)
  firrtl.module @ModuleInstances(in %inputWire: !firrtl.uint<1>, in %inputProp: !firrtl.string, out %outputWire: !firrtl.uint<1>, out %outputProp: !firrtl.string) {
    // CHECK: %[[EXT_IN_WIRE:.+]], %[[EXT_OUT_WIRE:.+]] = firrtl.instance ext @ExtModule
    %ext.inputWire, %ext.inputProp, %ext.outputWire, %ext.outputProp = firrtl.instance ext @ExtModule(in inputWire: !firrtl.uint<1>, in inputProp: !firrtl.string, out outputWire: !firrtl.uint<1>, out outputProp: !firrtl.string)
    // CHECK: %[[MOD_IN_WIRE:.+]], %[[MOD_OUT_WIRE:.+]] = firrtl.instance mod @Module
    %mod.inputWire, %mod.inputProp, %mod.outputWire, %mod.outputProp = firrtl.instance mod @Module(in inputWire: !firrtl.uint<1>, in inputProp: !firrtl.string, out outputWire: !firrtl.uint<1>, out outputProp: !firrtl.string)

    // CHECK: firrtl.strictconnect %[[EXT_IN_WIRE]], %[[IN_WIRE1]]
    firrtl.strictconnect %ext.inputWire, %inputWire : !firrtl.uint<1>
    // CHECK: firrtl.strictconnect %[[MOD_IN_WIRE]], %[[EXT_OUT_WIRE]]
    firrtl.strictconnect %mod.inputWire, %ext.outputWire : !firrtl.uint<1>
    // CHECK: firrtl.strictconnect %[[OUT_WIRE1]], %[[MOD_OUT_WIRE]]
    firrtl.strictconnect %outputWire, %mod.outputWire : !firrtl.uint<1>

    // CHECK-NEXT: }
    firrtl.propassign %ext.inputProp, %inputProp : !firrtl.string
    firrtl.propassign %mod.inputProp, %ext.outputProp : !firrtl.string
    firrtl.propassign %outputProp, %mod.outputProp : !firrtl.string
  }

  // CHECK: om.class.extern @ExtModule_Class(%inputProp: !om.string)
  // CHECK:   om.class.extern.field @outputProp : !om.string

  // CHECK: om.class @Module_Class(%[[IN_PROP0:.+]]: !om.string)
  // CHECK:   om.class.field @outputProp, %[[IN_PROP0]] : !om.string

  // CHECK: om.class @ModuleInstances_Class(%[[IN_PROP1:.+]]: !om.string)
  // CHECK:   %[[O0:.+]] = om.object @ExtModule_Class(%[[IN_PROP1]])
  // CHECK:   %[[F0:.+]] = om.object.field %[[O0]], [@outputProp]
  // CHECK:   %[[O1:.+]] = om.object @Module_Class(%[[F0]])
  // CHECK:   %[[F1:.+]] = om.object.field %[[O1]], [@outputProp]
  // CHECK:   om.class.field @outputProp, %[[F1]]
}

// CHECK-LABEL: firrtl.circuit "AnyCast"
firrtl.circuit "AnyCast" {
  firrtl.class private @Foo() {}

  firrtl.module @AnyCast(out %foo: !firrtl.anyref) {
    // CHECK: %[[OBJ:.+]] = om.object @Foo
    %fooObject = firrtl.object @Foo()
    // CHECK: %[[CAST:.+]] = om.any_cast %[[OBJ]]
    %0 = firrtl.object.anyref_cast %fooObject : !firrtl.class<@Foo()>
    // CHECK: om.class.field @foo, %[[CAST]]
    firrtl.propassign %foo, %0 : !firrtl.anyref
  }
}
