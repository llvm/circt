// RUN: circt-opt -firrtl-lower-classes %s | FileCheck %s

firrtl.circuit "Component" {
  // CHECK-LABEL: om.class @Class_0
  // CHECK-SAME: %[[REF1:[^ ]+]]: !om.class.type<@Class_1>
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

  // CHECK-LABEL: om.class.extern @ExtClass(%basepath: !om.basepath, %input: !om.string) {
  // CHECK-NEXT:    om.class.extern.field @field : !om.string
  // CHECK-NEXT:  }
  firrtl.extclass private @ExtClass(in input: !firrtl.string, out field: !firrtl.string)

  // CHECK-LABEL: om.class @ClassEntrypoint
  firrtl.class private @ClassEntrypoint(out %obj_0_out: !firrtl.class<@Class_1(out someInt: !firrtl.integer)>) {
    // CHECK: %[[OBJ1:.+]] = om.object @Class_1(%basepath) : (!om.basepath) -> !om.class.type<@Class_1>
    %obj1 = firrtl.object @Class_1(out someInt: !firrtl.integer)

    // CHECK: %[[OBJ0:.+]] = om.object @Class_0(%basepath, %[[OBJ1]]) : (!om.basepath, !om.class.type<@Class_1>) -> !om.class.type<@Class_0>
    %obj0 = firrtl.object @Class_0(in someReference_in: !firrtl.class<@Class_1(out someInt: !firrtl.integer)>, out someReference: !firrtl.class<@Class_1(out someInt: !firrtl.integer)>)
    %obj0_someReference_in = firrtl.object.subfield %obj0[someReference_in] : !firrtl.class<@Class_0(in someReference_in: !firrtl.class<@Class_1(out someInt: !firrtl.integer)>, out someReference: !firrtl.class<@Class_1(out someInt: !firrtl.integer)>)>
    firrtl.propassign %obj0_someReference_in, %obj1 : !firrtl.class<@Class_1(out someInt: !firrtl.integer)>

    // CHECK: %[[REF:.+]] = om.object.field %[[OBJ0]], [@someReference] : (!om.class.type<@Class_0>) -> !om.class.type<@Class_1>
    // CHECK: om.class.field @obj_0_out, %[[REF]] : !om.class.type<@Class_1>
    %obj0_someReference = firrtl.object.subfield %obj0[someReference] : !firrtl.class<@Class_0(in someReference_in: !firrtl.class<@Class_1(out someInt: !firrtl.integer)>, out someReference: !firrtl.class<@Class_1(out someInt: !firrtl.integer)>)>
    firrtl.propassign %obj_0_out, %obj0_someReference: !firrtl.class<@Class_1(out someInt: !firrtl.integer)>
  }

  // CHECK-LABEL: om.class @ReadOutputPort(%basepath: !om.basepath)
  firrtl.class @ReadOutputPort(out %output : !firrtl.integer) {
    // CHECK: %[[OBJ:.+]] = om.object @Class_1(%basepath) : (!om.basepath) -> !om.class.type<@Class_1>
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
    // CHECK: %2 = om.object @TwoInputs(%basepath, %0, %1) : (!om.basepath, !om.integer, !om.integer) -> !om.class.type<@TwoInputs>
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
    // CHECK: %2 = om.object @TwoInputs(%basepath, %0, %1) : (!om.basepath, !om.integer, !om.integer) -> !om.class.type<@TwoInputs>
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
    firrtl.matchingconnect %output, %input : !firrtl.uint<1>
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

  // CHECK-LABEL: om.class @BoolTest
  firrtl.class @BoolTest(out %b : !firrtl.bool) {
    // CHECK-NEXT: %[[TRUE:.+]] = om.constant true
    // CHECK-NEXT: om.class.field @b, %[[TRUE]] : i1
    %true = firrtl.bool true
    firrtl.propassign %b, %true : !firrtl.bool
  }

  // CHECK-LABEL: om.class @DoubleTest
  firrtl.class @DoubleTest(out %d : !firrtl.double) {
    // CHECK-NEXT: %[[DBL:.+]] = om.constant 4.0{{0*[eE]}}-01 : f64
    // CHECK-NEXT: om.class.field @d, %[[DBL]] : f64
    %dbl = firrtl.double 0.4
    firrtl.propassign %d, %dbl: !firrtl.double
  }
}

// CHECK-LABEL: firrtl.circuit "PathModule"
firrtl.circuit "PathModule" {
  // CHECK: hw.hierpath private [[PORT_PATH:@.+]] [@PathModule::[[PORT_SYM:@.+]]]
  // CHECK: hw.hierpath private [[WIRE_PATH:@.+]] [@PathModule::[[WIRE_SYM:@.+]]]
  // CHECK: hw.hierpath private [[VECTOR_PATH:@.+]] [@PathModule::[[VECTOR_SYM:@.+]]]
  // CHECK: hw.hierpath private [[INST_PATH:@.+]] [@PathModule::@child]
  // CHECK: hw.hierpath private [[MODULE_PATH:@.+]] [@PathModule::@child, @Child::[[NONLOCAL_SYM:@.+]]]

  // CHECK: firrtl.module @PathModule(in %in: !firrtl.uint<1> sym [[PORT_SYM]]) {
  firrtl.module @PathModule(in %in : !firrtl.uint<1> [{class = "circt.tracker", id = distinct[0]<>}]) {
    // CHECK: %wire = firrtl.wire sym [[WIRE_SYM]] : !firrtl.uint<8>
    %wire = firrtl.wire {annotations = [{class = "circt.tracker", id = distinct[1]<>}]} : !firrtl.uint<8>
    // CHECK: %vector = firrtl.wire sym [<[[VECTOR_SYM]],1,public>] : !firrtl.vector<uint<8>, 1>
    %vector = firrtl.wire {annotations = [{circt.fieldID = 1 : i32, class = "circt.tracker", id = distinct[2]<>}]} : !firrtl.vector<uint<8>, 1>
    // CHECK: firrtl.instance child sym @child @Child()
    firrtl.instance child sym @child {annotations = [{class = "circt.tracker", id = distinct[4]<>}]} @Child()

    %path_test = firrtl.object @PathTest()
  }
  // CHECK: hw.hierpath private [[NONLOCAL_PATH:@.+]] [@PathModule::@child, @Child]
  hw.hierpath private @NonLocal [@PathModule::@child, @Child]
  // CHECK: firrtl.module @Child() {
  firrtl.module @Child() attributes {annotations = [{class = "circt.tracker", id = distinct[5]<>}]} {
    // CHECK: %non_local = firrtl.wire sym [[NONLOCAL_SYM]] : !firrtl.uint<8>
    %non_local = firrtl.wire {annotations = [{circt.nonlocal = @NonLocal, class = "circt.tracker", id = distinct[3]<>}]} : !firrtl.uint<8>
  }
  // CHECK: om.class @PathModule_Class(%basepath: !om.basepath) {
  // CHECK:   om.basepath_create %basepath
  // CHECK:   om.object @Child_Class
  // CHECK:   om.object @PathTest
  // CHECK: om.class @PathTest(%basepath: !om.basepath)
  firrtl.class @PathTest() {
    
    // CHECK: om.path_create reference %basepath [[PORT_PATH]]
    %port_path = firrtl.path reference distinct[0]<>

    // CHECK: om.path_empty
    %deleted_path = firrtl.path reference distinct[99]<>

    // CHECK: om.path_create reference %basepath [[WIRE_PATH]]
    // CHECK: om.path_create member_reference %basepath [[WIRE_PATH]]
    // CHECK: om.path_create member_reference %basepath [[WIRE_PATH]]
    // CHECK: om.path_create dont_touch %basepath [[WIRE_PATH]]
    %wire_reference = firrtl.path reference distinct[1]<>
    %wire_member_instance = firrtl.path member_instance distinct[1]<>
    %wire_member_reference = firrtl.path member_reference distinct[1]<>
    %wire_dont_touch = firrtl.path dont_touch distinct[1]<>

    // CHECK: om.path_create reference %basepath [[VECTOR_PATH]]
    %vector_reference = firrtl.path reference distinct[2]<>

    %non_local_path = firrtl.path reference distinct[3]<>

    // CHECK: om.path_create reference %basepath [[MODULE_PATH]]
    // CHECK: om.path_create member_instance %basepath [[INST_PATH]]
    // CHECK: om.path_create member_instance %basepath [[INST_PATH]]
    // CHECK: om.path_create instance %basepath [[INST_PATH]]
    // CHECK: om.path_create reference %basepath [[NONLOCAL_PATH]]
    %instance_member_instance = firrtl.path member_instance distinct[4]<>
    %instance_member_reference = firrtl.path member_reference distinct[4]<>
    %instance = firrtl.path instance distinct[4]<>

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
  // CHECK-SAME: %[[IN:[^ ]+]]: !om.string
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
  // CHECK: hw.hierpath private @[[EXT_NLA:.+]] [@ModuleInstances::@[[EXT_SYM:.+]]]
  // CHECK: hw.hierpath private @[[EXTDEFNAME_NLA:.+]] [@ModuleInstances::@[[EXTDEFNAME_SYM:.+]]]
  // CHECK: hw.hierpath private @[[MOD_NLA:.+]] [@ModuleInstances::@[[MOD_SYM:.+]]]
  // CHECK: firrtl.extmodule private @ExtModule(in inputWire: !firrtl.uint<1>, out outputWire: !firrtl.uint<1>)
  firrtl.extmodule private @ExtModule(in inputWire: !firrtl.uint<1>, in inputProp: !firrtl.string, out outputWire: !firrtl.uint<1>, out outputProp: !firrtl.string)

  // CHECK: firrtl.extmodule private @ExtModuleDefname
  firrtl.extmodule private @ExtModuleDefname(in inputProp: !firrtl.string, out outputProp: !firrtl.string) attributes {defname = "TheRealName"}

  // CHECK: firrtl.module private @Module(in %[[IN_WIRE0:.+]]: !firrtl.uint<1>, out %[[OUT_WIRE0:.+]]: !firrtl.uint<1>)
  firrtl.module private @Module(in %inputWire: !firrtl.uint<1>, in %inputProp: !firrtl.string, out %outputWire: !firrtl.uint<1>, out %outputProp: !firrtl.string) {
    // CHECK: firrtl.matchingconnect %[[OUT_WIRE0]], %[[IN_WIRE0]]
    firrtl.matchingconnect %outputWire, %inputWire : !firrtl.uint<1>
    // CHECK-NEXT: }
    firrtl.propassign %outputProp, %inputProp : !firrtl.string
  }

  // CHECK: firrtl.module @ModuleInstances(in %[[IN_WIRE1:.+]]: !firrtl.uint<1>, out %[[OUT_WIRE1:.+]]: !firrtl.uint<1>)
  firrtl.module @ModuleInstances(in %inputWire: !firrtl.uint<1>, in %inputProp: !firrtl.string, out %outputWire: !firrtl.uint<1>, out %outputProp: !firrtl.string) {
    // CHECK: %[[EXT_IN_WIRE:.+]], %[[EXT_OUT_WIRE:.+]] = firrtl.instance ext sym @[[EXT_SYM]] @ExtModule
    %ext.inputWire, %ext.inputProp, %ext.outputWire, %ext.outputProp = firrtl.instance ext @ExtModule(in inputWire: !firrtl.uint<1>, in inputProp: !firrtl.string, out outputWire: !firrtl.uint<1>, out outputProp: !firrtl.string)
    // CHECK: firrtl.instance extdefname sym @[[EXTDEFNAME_SYM]] @ExtModuleDefname
    %extdefname.inputProp, %extdefname.outputProp = firrtl.instance extdefname @ExtModuleDefname(in inputProp: !firrtl.string, out outputProp: !firrtl.string)
    // CHECK: %[[MOD_IN_WIRE:.+]], %[[MOD_OUT_WIRE:.+]] = firrtl.instance mod sym @[[MOD_SYM]] @Module
    %mod.inputWire, %mod.inputProp, %mod.outputWire, %mod.outputProp = firrtl.instance mod @Module(in inputWire: !firrtl.uint<1>, in inputProp: !firrtl.string, out outputWire: !firrtl.uint<1>, out outputProp: !firrtl.string)

    // CHECK: firrtl.matchingconnect %[[EXT_IN_WIRE]], %[[IN_WIRE1]]
    firrtl.matchingconnect %ext.inputWire, %inputWire : !firrtl.uint<1>
    // CHECK: firrtl.matchingconnect %[[MOD_IN_WIRE]], %[[EXT_OUT_WIRE]]
    firrtl.matchingconnect %mod.inputWire, %ext.outputWire : !firrtl.uint<1>
    // CHECK: firrtl.matchingconnect %[[OUT_WIRE1]], %[[MOD_OUT_WIRE]]
    firrtl.matchingconnect %outputWire, %mod.outputWire : !firrtl.uint<1>

    // CHECK-NEXT: }
    firrtl.propassign %ext.inputProp, %inputProp : !firrtl.string
    firrtl.propassign %extdefname.inputProp, %inputProp : !firrtl.string
    firrtl.propassign %mod.inputProp, %ext.outputProp : !firrtl.string
    firrtl.propassign %outputProp, %mod.outputProp : !firrtl.string
  }

  // CHECK: om.class.extern @ExtModule_Class(%basepath: !om.basepath, %inputProp: !om.string)
  // CHECK:   om.class.extern.field @outputProp : !om.string

  // CHECK: om.class.extern @TheRealName_Class(%basepath: !om.basepath, %inputProp: !om.string)
  // CHECK:   om.class.extern.field @outputProp : !om.string

  // CHECK: om.class @Module_Class(%basepath: !om.basepath, %[[IN_PROP0:.+]]: !om.string)
  // CHECK:   om.class.field @outputProp, %[[IN_PROP0]] : !om.string

  // CHECK: om.class @ModuleInstances_Class(%basepath: !om.basepath, %[[IN_PROP1:.+]]: !om.string)
  // CHECK:   %[[BASEPATH:.+]] = om.basepath_create %basepath @[[EXT_NLA]]
  // CHECK:   %[[O0:.+]] = om.object @ExtModule_Class(%[[BASEPATH]], %[[IN_PROP1]])
  // CHECK:   %[[F0:.+]] = om.object.field %[[O0]], [@outputProp]
  // CHECK:   %[[BASEPATH:.+]] = om.basepath_create %basepath @[[EXTDEFNAME_NLA]]
  // CHECK:   om.object @TheRealName_Class
  // CHECK:   %[[BASEPATH:.+]] = om.basepath_create %basepath @[[MOD_NLA]]
  // CHECK:   %[[O1:.+]] = om.object @Module_Class(%[[BASEPATH]], %[[F0]])
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

// CHECK-LABEL: firrtl.circuit "ModuleWithPropertySubmodule"
firrtl.circuit "ModuleWithPropertySubmodule" {
  // CHECK: om.class @ModuleWithPropertySubmodule_Class
  firrtl.module private @ModuleWithPropertySubmodule() {
    %c0 = firrtl.integer 0
    // CHECK: om.object @SubmoduleWithProperty_Class
    %inst.prop = firrtl.instance inst @SubmoduleWithProperty(in prop: !firrtl.integer)
    firrtl.propassign %inst.prop, %c0 : !firrtl.integer
  }
  // CHECK: om.class @SubmoduleWithProperty_Class
  firrtl.module private @SubmoduleWithProperty(in %prop: !firrtl.integer) {
  }
}

// CHECK-LABEL: firrtl.circuit "DownwardReferences"
firrtl.circuit "DownwardReferences" {
  firrtl.class @MyClass() {
  }
  firrtl.module @MyClassUser(in %myClassIn: !firrtl.class<@MyClass()>) {
  }
  firrtl.module @DownwardReferences() {
    // CHECK: [[OBJ:%.+]] = om.object @MyClass
    %myClass = firrtl.object @MyClass()
    // CHECK: [[BP:%.+]] = om.basepath_create
    // CHECK: om.object @MyClassUser_Class([[BP]], [[OBJ]])
    %myClassUser.myClassIn = firrtl.instance myClassUser @MyClassUser(in myClassIn: !firrtl.class<@MyClass()>)
    firrtl.propassign %myClassUser.myClassIn, %myClass : !firrtl.class<@MyClass()>
  }
}

// CHECK-LABEL: firrtl.circuit "IntegerArithmetic"
firrtl.circuit "IntegerArithmetic" {
  firrtl.module @IntegerArithmetic() {
    %0 = firrtl.integer 1
    %1 = firrtl.integer 2

    // CHECK: om.integer.add %0, %1 : !om.integer
    %2 = firrtl.integer.add %0, %1 : (!firrtl.integer, !firrtl.integer) -> !firrtl.integer

    // CHECK: om.integer.mul %0, %1 : !om.integer
    %3 = firrtl.integer.mul %0, %1 : (!firrtl.integer, !firrtl.integer) -> !firrtl.integer

    // CHECK: om.integer.shr %0, %1 : !om.integer
    %4 = firrtl.integer.shr %0, %1 : (!firrtl.integer, !firrtl.integer) -> !firrtl.integer
  }
}

// CHECK-LABEL: firrtl.circuit "AltBasePath"
firrtl.circuit "AltBasePath" {
  firrtl.class private @Node(in %path: !firrtl.path) {
  }

  // CHECK: om.class @OMIR(%basepath: !om.basepath, %alt_basepath_0: !om.basepath)
  firrtl.class private @OMIR() {
    %node = firrtl.object @Node(in path: !firrtl.path)
    %0 = firrtl.object.subfield %node[path] : !firrtl.class<@Node(in path: !firrtl.path)>

    // CHECK: om.path_create member_instance %alt_basepath_0
    %1 = firrtl.path member_reference distinct[0]<>
    firrtl.propassign %0, %1 : !firrtl.path
  }

  // CHECK: om.class @DUT_Class(%basepath: !om.basepath, %alt_basepath_0: !om.basepath)
  firrtl.module @DUT(out %omirOut: !firrtl.class<@OMIR()>) attributes {convention = #firrtl<convention scalarized>} {
    // CHECK: om.object @OMIR(%basepath, %alt_basepath_0)
    %omir = firrtl.object @OMIR()
    firrtl.propassign %omirOut, %omir : !firrtl.class<@OMIR()>
  }

  // CHECK: om.class @AltBasePath_Class(%basepath: !om.basepath)
  firrtl.module @AltBasePath() attributes {convention = #firrtl<convention scalarized>} {
    // CHECK: om.object @DUT_Class(%0, %basepath)
    %dut_omirOut = firrtl.instance dut interesting_name @DUT(out omirOut: !firrtl.class<@OMIR()>)
    firrtl.instance foo interesting_name {annotations = [{class = "circt.tracker", id = distinct[0]<>}]} @Foo()
  }

  firrtl.module private @Foo() attributes {annotations = [{class = "circt.tracker", id = distinct[1]<>}]} {
    firrtl.skip
  }
}

// CHECK-LABEL: firrtl.circuit "NonRootedPath"
firrtl.circuit "NonRootedPath" {
  // CHECK: hw.hierpath private [[NLA:@.+]] [@NonRootedPath::[[SYM:@.+]], @Child::@grandchild, @GrandChild::@wire]
  hw.hierpath @nla [@Child::@grandchild, @GrandChild]
  firrtl.module @NonRootedPath() {
    // CHECK: firrtl.instance child sym [[SYM]] @Child
    firrtl.instance child @Child()
    // CHECK: om.path_create reference %basepath [[NLA]]
    firrtl.path reference distinct[0]<>
  }
  firrtl.module @Child() {
    firrtl.instance grandchild sym @grandchild @GrandChild()
  }
  firrtl.module @GrandChild() {
    %foo = firrtl.wire sym @wire {annotations = [{circt.nonlocal = @nla, class = "circt.tracker", id = distinct[0]<>}]} : !firrtl.uint<1>
  }
}

// CHECK-LABEL: firrtl.circuit "OwningModulePrefix"
firrtl.circuit "OwningModulePrefix" {
  // COM: Ensure the hierpath used in the path op starts at the owning module.
  // CHECK: hw.hierpath private [[NLA:@.+]] [@OwningModule::{{.+}}]
  hw.hierpath private @nla [@OwningModulePrefix::@sym0, @OwningModule::@sym1, @OwningModuleChild::@sym2]
  firrtl.module @OwningModulePrefix() {
    firrtl.instance owning_module sym @sym0 @OwningModule()
  }
  firrtl.module @OwningModule() {
    firrtl.instance owning_module_child sym @sym1 @OwningModuleChild()
    firrtl.object @OwningModuleClass()
  }
  firrtl.module @OwningModuleChild() {
    %w = firrtl.wire sym @sym2 {annotations = [{class = "circt.tracker", id = distinct[0]<>, circt.nonlocal = @nla}]} : !firrtl.uint<0>
  }
  firrtl.class @OwningModuleClass() {
    // CHECK: om.path_create reference %basepath [[NLA]]
    firrtl.path reference distinct[0]<>
  }
}
