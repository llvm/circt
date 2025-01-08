// RUN: circt-opt -pass-pipeline='builtin.module(firrtl.circuit(firrtl-grand-central,symbol-dce))' -split-input-file %s | FileCheck %s

// This is the main test that includes different interfaces of different
// types. All the interfaces share a common, simple circuit that provides two
// signals, "foo" and "bar".

firrtl.circuit "InterfaceGroundType" attributes {
  annotations = [
    {
      class = "sifive.enterprise.grandcentral.ExtractGrandCentralAnnotation",
      directory = "gct-dir",
      filename = "bindings.sv"
    },
    {
      class = "sifive.enterprise.grandcentral.GrandCentralHierarchyFileAnnotation",
      filename = "gct.yaml"
    }
  ]
} {
  firrtl.module @Companion() {
    // These are dummy references created for the purposes of the test.
    %_ui0 = firrtl.verbatim.expr "???" : () -> !firrtl.uint<0>
    %_ui1 = firrtl.verbatim.expr "???" : () -> !firrtl.uint<1>
    %_ui2 = firrtl.verbatim.expr "???" : () -> !firrtl.uint<2>
    %ref_ui0 = firrtl.ref.send %_ui0 : !firrtl.uint<0>
    %ref_ui1 = firrtl.ref.send %_ui1 : !firrtl.uint<1>
    %ref_ui2 = firrtl.ref.send %_ui2 : !firrtl.uint<2>

    %ui1 = firrtl.ref.resolve %ref_ui1 : !firrtl.probe<uint<1>>
    %foo = firrtl.node %ui1 : !firrtl.uint<1>

    %ui2 = firrtl.ref.resolve %ref_ui2 : !firrtl.probe<uint<2>>
    %bar = firrtl.node %ui2 : !firrtl.uint<2>

    %ui0 = firrtl.ref.resolve %ref_ui0 : !firrtl.probe<uint<0>>
    %baz = firrtl.node %ui0 : !firrtl.uint<0>

    %c0_ui1 = firrtl.constant 0 : !firrtl.uint<1>
    %c-1_si2 = firrtl.constant -1 : !firrtl.sint<2>

    firrtl.view "GroundView", <{
      class = "sifive.enterprise.grandcentral.AugmentedBundleType",
      defName = "GroundView",
      elements = [
        {
          class = "sifive.enterprise.grandcentral.AugmentedGroundType",
          description = "description of foo",
          name = "foo"
        },
        {
          class = "sifive.enterprise.grandcentral.AugmentedGroundType",
          description = "multi\nline\ndescription\nof\nbar",
          name = "bar"
        }
      ]
    }>, %foo, %bar : !firrtl.uint<1>, !firrtl.uint<2>

    firrtl.view "VectorView", <{
      class = "sifive.enterprise.grandcentral.AugmentedBundleType",
      defName = "VectorView",
      elements = [
        {
          class = "sifive.enterprise.grandcentral.AugmentedVectorType",
          elements = [
            {
              class = "sifive.enterprise.grandcentral.AugmentedGroundType",
              name = "foo"
            },
            {
              class = "sifive.enterprise.grandcentral.AugmentedGroundType",
              name = "bar"
            }
          ],
          name = "vector"
        }
      ]
    }>, %foo, %foo : !firrtl.uint<1>, !firrtl.uint<1>


    firrtl.view "BundleView", <{
      class = "sifive.enterprise.grandcentral.AugmentedBundleType",
      defName = "BundleView",
      elements = [
        {
          class = "sifive.enterprise.grandcentral.AugmentedBundleType",
          defName = "Bundle",
          elements = [
            {
              class = "sifive.enterprise.grandcentral.AugmentedGroundType",
              name = "foo"
            },
            {
              class = "sifive.enterprise.grandcentral.AugmentedGroundType",
              name = "bar"
            }
          ],
          name = "bundle"
        }
      ]
    }>, %foo, %bar : !firrtl.uint<1>, !firrtl.uint<2>

    firrtl.view "VectorOfBundleView", <{
      class = "sifive.enterprise.grandcentral.AugmentedBundleType",
      defName = "VectorOfBundleView",
      elements = [
        {
          class = "sifive.enterprise.grandcentral.AugmentedVectorType",
          elements = [
            {
              class = "sifive.enterprise.grandcentral.AugmentedBundleType",
              defName = "Bundle2",
              elements = [
                {
                  class = "sifive.enterprise.grandcentral.AugmentedGroundType",
                  name = "foo"
                },
                {
                  class = "sifive.enterprise.grandcentral.AugmentedGroundType",
                  name = "bar"
                }
              ],
              name = "bundle2"
            }
          ],
          name = "vector"
        }
      ]
    }>, %foo, %bar : !firrtl.uint<1>, !firrtl.uint<2>

    firrtl.view "VectorOfVectorView", <{
      class = "sifive.enterprise.grandcentral.AugmentedBundleType",
      defName = "VectorOfVectorView",
      elements = [
        {
          class = "sifive.enterprise.grandcentral.AugmentedVectorType",
          elements = [
            {
              class = "sifive.enterprise.grandcentral.AugmentedVectorType",
              defName = "Vector2",
              elements = [
                {
                  class = "sifive.enterprise.grandcentral.AugmentedGroundType",
                  name = "foo"
                },
                {
                  class = "sifive.enterprise.grandcentral.AugmentedGroundType",
                  name = "bar"
                },
                {
                  class = "sifive.enterprise.grandcentral.AugmentedGroundType",
                  name = "baz"
                }
              ],
              name = "vector2"
            },
            {
              class = "sifive.enterprise.grandcentral.AugmentedVectorType",
              defName = "Vector2",
              elements = [
                {
                  class = "sifive.enterprise.grandcentral.AugmentedGroundType",
                  name = "foo"
                },
                {
                  class = "sifive.enterprise.grandcentral.AugmentedGroundType",
                  name = "bar"
                },
                {
                  class = "sifive.enterprise.grandcentral.AugmentedGroundType",
                  name = "baz"
                }
              ],
              name = "vector2"
            }
          ],
          name = "vector"
        }
      ]
    }>, %bar, %bar, %bar, %bar, %bar, %bar : !firrtl.uint<2>, !firrtl.uint<2>, !firrtl.uint<2>, !firrtl.uint<2>, !firrtl.uint<2>, !firrtl.uint<2>

    firrtl.view "ConstantView", <{
      class = "sifive.enterprise.grandcentral.AugmentedBundleType",
      defName = "ConstantView",
      elements = [
        {
          class = "sifive.enterprise.grandcentral.AugmentedGroundType",

          name = "foo"
        },
        {
          class = "sifive.enterprise.grandcentral.AugmentedGroundType",
          name = "bar"
        }
      ]
    }>, %c0_ui1, %c-1_si2 : !firrtl.uint<1>, !firrtl.sint<2>

    firrtl.view "ZeroWidthView", <{
      class = "sifive.enterprise.grandcentral.AugmentedBundleType",
      defName = "ZeroWidthView",
      elements = [
        {
          class = "sifive.enterprise.grandcentral.AugmentedGroundType",
          name = "zerowidth"
        }
      ]
    }>, %baz : !firrtl.uint<0>
  }
  firrtl.module @InterfaceGroundType() {
    firrtl.instance companion @Companion()
  }
}

// All AugmentedBundleType annotations are removed from the circuit.
//
// CHECK-LABEL: firrtl.circuit "InterfaceGroundType" {{.+}} {annotations =
// CHECK-SAME:    class = "sifive.enterprise.grandcentral.ExtractGrandCentralAnnotation"
// CHECK-NOT:     class = "sifive.enterprise.grandcentral.AugmentedBundleType"
// CHECK-SAME: {

// Check YAML Output.
//
// Note: Built-in vector serialization works slightly differently than
// user-defined vector serialization.  This results in the verbose "[ ]" for the
// empty dimensions vector, and the terse "[]" for the empty instances vector.
//
// CHECK:      sv.verbatim
// CHECK-SAME:   - name: GroundView
// CHECK-SAME:     fields:
// CHECK-SAME:       - name: foo
// CHECK-SAME:         description: description of foo
// CHECK-SAME:         dimensions: [ ]
// CHECK-SAME:         width: 1
// CHECK-SAME:       - name: bar
// CHECK-SAME:         description: \22multi\\nline\\ndescription\\nof\\nbar\22
// CHECK-SAME:         dimensions: [ ]
// CHECK-SAME:         width: 2
// CHECK-SAME:     instances: []
// CHECK-SAME:   - name: VectorView
// CHECK-SAME:     fields:
// CHECK-SAME:       - name: vector
// CHECK-SAME:         dimensions: [ 2 ]
// CHECK-SAME:         width: 1
// CHECK-SAME:     instances: []
// CHECK-SAME:   - name: BundleView
// CHECK-SAME:     fields: []
// CHECK-SAME:     instances:
// CHECK-SAME:       - name: bundle
// CHECK-SAME:         dimensions: [ ]
// CHECK-SAME:         interface:
// CHECK-SAME:           name: Bundle
// CHECK-SAME:           fields:
// CHECK-SAME:             - name: foo
// CHECK-SAME:               dimensions: [ ]
// CHECK-SAME:               width: 1
// CHECK-SAME:             - name: bar
// CHECK-SAME:               dimensions: [ ]
// CHECK-SAME:               width: 2
// CHECK-SAME:           instances: []
// CHECK-SAME:   - name: VectorOfBundleView
// CHECK-SAME:     fields: []
// CHECK-SAME:     instances:
// CHECK-SAME:       - name: vector
// CHECK-SAME:         dimensions: [ 1 ]
// CHECK-SAME:         interface:
// CHECK-SAME:           name: Bundle2
// CHECK-SAME:           fields:
// CHECK-SAME:             - name: foo
// CHECK-SAME:               dimensions: [ ]
// CHECK-SAME:               width: 1
// CHECK-SAME:             - name: bar
// CHECK-SAME:               dimensions: [ ]
// CHECK-SAME:               width: 2
// CHECK-SAME:           instances: []
// CHECK-SAME:   - name: VectorOfVectorView
// CHECK-SAME:     fields:
// CHECK-SAME:       - name: vector
// CHECK-SAME:         dimensions: [ 3, 2 ]
// CHECK-SAME:         width: 2
// CHECK-SAME:     instances: []
// CHECK-SAME:   - name: ConstantView
// CHECK-SAME:     fields:
// CHECK-SAME:       - name: foo
// CHECK-SAME:         dimensions: [ ]
// CHECK-SAME:         width: 1
// CHECK-SAME:       - name: bar
// CHECK-SAME:         dimensions: [ ]
// CHECK-SAME:         width: 2
// CHECK-SAME:     instances: []
// CHECK-SAME:   - name: ZeroWidthView
// CHECK-SAME:     fields:
// CHECK-SAME:       - name: zerowidth
// CHECK-SAME:         dimensions: [ ]
// CHECK-SAME:         width: 0
// CHECK-SAME:     instances: []

// The shared companion contains all instantiated interfaces.
// AugmentedGroundType annotations are removed.  Interface is driven via XMRs
// directly from ref resolve ops.
//
// CHECK:          firrtl.module @Companion
//
// CHECK:            %[[foo_ref:[a-zA-Z0-9_]+]] = firrtl.ref.resolve {{.+}} : !firrtl.probe<uint<1>>
// CHECK-NOT:        sifive.enterprise.grandcentral.AugmentedGroundType
// CHECK:            %[[bar_ref:[a-zA-Z0-9_]+]] = firrtl.ref.resolve {{.+}} : !firrtl.probe<uint<2>>
// CHECK-NOT:        sifive.enterprise.grandcentral.AugmentedGroundType

// CHECK:            %GroundView = sv.interface.instance sym @[[groundSym:[a-zA-Z0-9_]+]] : !sv.interface<@GroundView>
//
// CHECK{LITERAL}:   sv.verbatim "assign {{1}}.foo = {{0}};"
// CHECK-SAME:         (%foo) : !firrtl.uint<1>
// CHECK-SAME:         {symbols = [#hw.innerNameRef<@Companion::@[[groundSym]]>]}
// CHECK{LITERAL}:   sv.verbatim "assign {{1}}.bar = {{0}};"
// CHECK-SAME:         (%bar) : !firrtl.uint<2>
// CHECK-SAME:         {symbols = [#hw.innerNameRef<@Companion::@[[groundSym]]>]}

// CHECK-NEXT:       %VectorView = sv.interface.instance sym @[[vectorSym:[a-zA-Z0-9_]+]] : !sv.interface<@VectorView>
// CHECK{LITERAL}:   sv.verbatim "assign {{1}}.vector[0] = {{0}};"
// CHECK-SAME:         (%foo) : !firrtl.uint<1>
// CHECK-SAME:         {symbols = [#hw.innerNameRef<@Companion::@[[vectorSym]]>]}
// CHECK{LITERAL}:   sv.verbatim "assign {{1}}.vector[1] = {{0}};"
// CHECK-SAME:         (%foo) : !firrtl.uint<1>
// CHECK-SAME:         {symbols = [#hw.innerNameRef<@Companion::@[[vectorSym]]>]}

// CHECK-NEXT:       %BundleView = sv.interface.instance sym @[[bundleSym:[a-zA-Z0-9_]+]] : !sv.interface<@BundleView>
// CHECK{LITERAL}:   sv.verbatim "assign {{1}}.bundle.foo = {{0}};"
// CHECK-SAME:         (%foo) : !firrtl.uint<1>
// CHECK-SAME:         {symbols = [#hw.innerNameRef<@Companion::@[[bundleSym]]>]}
// CHECK{LITERAL}:   sv.verbatim "assign {{1}}.bundle.bar = {{0}};"
// CHECK-SAME:         (%bar) : !firrtl.uint<2>
// CHECK-SAME:         {symbols = [#hw.innerNameRef<@Companion::@[[bundleSym]]>]}

// CHECK-NEXT:       %VectorOfBundleView = sv.interface.instance sym @[[vectorOfBundleSym:[a-zA-Z0-9_]+]] : !sv.interface<@VectorOfBundleView>
// CHECK{LITERAL}:   sv.verbatim "assign {{1}}.vector[0].foo = {{0}};"
// CHECK-SAME:         (%foo) : !firrtl.uint<1>
// CHECK-SAME:         {symbols = [#hw.innerNameRef<@Companion::@[[vectorOfBundleSym]]>]}
// CHECK{LITERAL}:   sv.verbatim "assign {{1}}.vector[0].bar = {{0}};"
// CHECK-SAME:         (%bar) : !firrtl.uint<2>
// CHECK-SAME:         {symbols = [#hw.innerNameRef<@Companion::@[[vectorOfBundleSym]]>]}

// CHECK-NEXT:       %VectorOfVectorView = sv.interface.instance sym @[[vectorOfVectorSym:[a-zA-Z0-9_]+]] : !sv.interface<@VectorOfVectorView>
// CHECK{LITERAL}:   sv.verbatim "assign {{1}}.vector[0][0] = {{0}};"
// CHECK-SAME:         (%bar) : !firrtl.uint<2>
// CHECK-SAME:         {symbols = [#hw.innerNameRef<@Companion::@[[vectorOfVectorSym]]>]}
// CHECK{LITERAL}:   sv.verbatim "assign {{1}}.vector[0][1] = {{0}};"
// CHECK-SAME:         (%bar) : !firrtl.uint<2>
// CHECK-SAME:         {symbols = [#hw.innerNameRef<@Companion::@[[vectorOfVectorSym]]>]}
// CHECK{LITERAL}:   sv.verbatim "assign {{1}}.vector[0][2] = {{0}};"
// CHECK-SAME:         (%bar) : !firrtl.uint<2>
// CHECK-SAME:         {symbols = [#hw.innerNameRef<@Companion::@[[vectorOfVectorSym]]>]}
// CHECK{LITERAL}:   sv.verbatim "assign {{1}}.vector[1][0] = {{0}};"
// CHECK-SAME:         (%bar) : !firrtl.uint<2>
// CHECK-SAME:         {symbols = [#hw.innerNameRef<@Companion::@[[vectorOfVectorSym]]>]}
// CHECK{LITERAL}:   sv.verbatim "assign {{1}}.vector[1][1] = {{0}};"
// CHECK-SAME:         (%bar) : !firrtl.uint<2>
// CHECK-SAME:         {symbols = [#hw.innerNameRef<@Companion::@[[vectorOfVectorSym]]>]}
// CHECK{LITERAL}:   sv.verbatim "assign {{1}}.vector[1][2] = {{0}};"
// CHECK-SAME:         (%bar) : !firrtl.uint<2>
// CHECK-SAME:         {symbols = [#hw.innerNameRef<@Companion::@[[vectorOfVectorSym]]>]}

// CHECK-NEXT:       %ConstantView = sv.interface.instance sym @[[constantSym:[a-zA-Z0-9_]+]] : !sv.interface<@ConstantView>
// CHECK{LITERAL}:   sv.verbatim "assign {{1}}.foo = {{0}};"
// CHECK-SAME:         (%c0_ui1) : !firrtl.uint<1>
// CHECK-SAME:         {symbols = [#hw.innerNameRef<@Companion::@[[constantSym]]>]}
// CHECK{LITERAL}:   sv.verbatim "assign {{1}}.bar = {{0}};"
// CHECK-SAME:         (%c-1_si2) : !firrtl.sint<2>
// CHECK-SAME:         {symbols = [#hw.innerNameRef<@Companion::@[[constantSym]]>]}

// There are no more verbatim assigns after this.
// Zero-width views are not given XMR's.
//
// CHECK-NEXT:       %ZeroWidthView = sv.interface.instance sym @[[zeroWidthSym:[a-zA-Z0-9_]+]] : !sv.interface<@ZeroWidthView>
// CHECK-NOT:        sv.verbatim "assign

// These views do not have a notion of "companion" module.
//
// CHECK:          firrtl.module @InterfaceGroundType()
// CHECK:            firrtl.instance companion
// CHECK-NOT:         lowerToBind
// CHECK-NOT:         output_file = #hw.output_file<"bindings.sv", excludeFromFileList>}

// The body of all interfaces are populated with the correct signals, names,
// comments, and types.
//
// CHECK:      sv.interface @GroundView
// CHECK-SAME:   comment = "VCS coverage exclude_file"

// CHECK-NOT:    output_file = #hw.output_file<"gct-dir{{/|\\\\}}"

// CHECK-NEXT:   sv.verbatim "// description of foo"
// CHECK-NEXT:   sv.interface.signal @foo : i1
// CHECK-NEXT:   sv.verbatim "// multi\0A// line\0A// description\0A// of\0A// bar"
// CHECK-NEXT:   sv.interface.signal @bar : i2
//
// CHECK:      sv.interface @VectorView
// CHECK-NEXT:   sv.interface.signal @vector : !hw.uarray<2xi1>
//
// CHECK:      sv.interface @BundleView
// CHECK-NEXT:   sv.verbatim "Bundle bundle();"
//
// CHECK:      sv.interface @Bundle
// CHECK-NEXT:   sv.interface.signal @foo : i1
// CHECK-NEXT:   sv.interface.signal @bar : i2
//
// CHECK:      sv.interface @VectorOfBundleView
// CHECK-NEXT:   sv.verbatim "Bundle2 vector[1]();"
//
// CHECK:      sv.interface @Bundle2
// CHECK-NEXT:   sv.interface.signal @foo : i1
// CHECK-NEXT:   sv.interface.signal @bar : i2
//
// CHECK:      sv.interface @VectorOfVectorView
// CHECK-NEXT:   sv.interface.signal @vector : !hw.uarray<2xuarray<3xi2>>
//
// CHECK:      sv.interface @ConstantView
// CHECK-NEXT:   sv.interface.signal @foo : i1
// CHECK-NEXT:   sv.interface.signal @bar : i2
//
// CHECK:      sv.interface @ZeroWidthView
// CHECK-NEXT:   sv.interface.signal @zerowidth : i0

// -----

firrtl.circuit "DirectoryBehaviorWithDUT" attributes {
  annotations = [
    {class = "sifive.enterprise.grandcentral.AugmentedBundleType",
     defName = "Foo",
     elements = [],
     id = 0 : i64,
     name = "View"},
    {class = "sifive.enterprise.grandcentral.ExtractGrandCentralAnnotation",
     directory = "gct-dir",
     filename = "bindings.sv"}]} {

  // Each of these modules is instantiated in a different location.  A leading
  // "E" indicates that this is an external module.  A leading "M" indicates
  // that this is a module.  The instantiation location is indicated by three
  // binary bits with an "_" indicating the absence of instantiation:
  //   1) "T" indicates this is instantiated in the "Top" (above the DUT)
  //   2) "D" indicates this is instantiated in the "DUT"
  //   3) "C" indicates this is instantiated in the "Companion"
  // E.g., "ET_C" is an external module instantiated above the DUT and in the
  // Companion.
  firrtl.module @MT__() {}
  firrtl.module @M_D_() {}
  firrtl.module @M__C() {}
  firrtl.module @MTD_() {}
  firrtl.module @M_DC() {}
  firrtl.module @MT_C() {}
  firrtl.module @MTDC() {}
  firrtl.extmodule @ET__() attributes {annotations = [
    {class = "firrtl.transforms.BlackBoxInlineAnno", name = "ET__.v", text = ""}
  ]}
  firrtl.extmodule @E_D_() attributes {annotations = [
    {class = "firrtl.transforms.BlackBoxInlineAnno", name = "E_D_.v", text = ""}
  ]}
  firrtl.extmodule @E__C() attributes {annotations = [
    {class = "firrtl.transforms.BlackBoxInlineAnno", name = "E__C.v", text = ""}
  ]}
  firrtl.extmodule @ETD_() attributes {annotations = [
    {class = "firrtl.transforms.BlackBoxInlineAnno", name = "ETD_.v", text = ""}
  ]}
  firrtl.extmodule @E_DC() attributes {annotations = [
    {class = "firrtl.transforms.BlackBoxInlineAnno", name = "E_DC.v", text = ""}
  ]}
  firrtl.extmodule @ET_C() attributes {annotations = [
    {class = "firrtl.transforms.BlackBoxInlineAnno", name = "ET_C.v", text = ""}
  ]}
  firrtl.extmodule @ETDC() attributes {annotations = [
    {class = "firrtl.transforms.BlackBoxInlineAnno", name = "ETDC.v", text = ""}
  ]}

  // The Grand Central Companion module.
  firrtl.module private @Companion() attributes {
    annotations = [
      {class = "sifive.enterprise.grandcentral.ViewAnnotation.companion",
       defName = "Foo",
       id = 0 : i64,
       name = "View"}]} {

    firrtl.instance m__c @M__C()
    firrtl.instance m_dc @M_DC()
    firrtl.instance mt_c @MT_C()
    firrtl.instance mtdc @MTDC()

    firrtl.instance e__c @E__C()
    firrtl.instance e_dc @E_DC()
    firrtl.instance et_c @ET_C()
    firrtl.instance etdc @ETDC()
  }

  // The Design-under-test as indicated by the MarkDUTAnnotation
  firrtl.module private @DUT() attributes {
    annotations = [
      {class = "sifive.enterprise.firrtl.MarkDUTAnnotation"}
    ]
  } {
    firrtl.instance companion @Companion()

    firrtl.instance m_d_ @M_D_()
    firrtl.instance mtd_ @MTD_()
    firrtl.instance m_dc @M_DC()
    firrtl.instance mtdc @MTDC()

    firrtl.instance e_d_ @E_D_()
    firrtl.instance etd_ @ETD_()
    firrtl.instance e_dc @E_DC()
    firrtl.instance etdc @ETDC()
  }

  // The Top module that instantiates the DUT
  firrtl.module @DirectoryBehaviorWithDUT() {
    firrtl.instance dut @DUT()

    firrtl.instance mt__ @MT__()
    firrtl.instance mtd_ @MTD_()
    firrtl.instance mt_c @MT_C()
    firrtl.instance mtdc @MTDC()

    firrtl.instance et__ @ET__()
    firrtl.instance etd_ @ETD_()
    firrtl.instance et_c @ET_C()
    firrtl.instance etdc @ETDC()
  }
}

// Any module instantiated by the Companion, but not instantiated by the DUT is
// moved to the same directory as the Companion.  I.e., only "*__C" and "*T_C"
// modules should be moved into the "gct-dir".
//
// CHECK-LABEL: "DirectoryBehaviorWithDUT"
//
// CHECK-NOT:    output_file
// CHECK:      firrtl.module @M__C
// CHECK-SAME:   output_file = #hw.output_file<"gct-dir{{/|\\\\}}"
// CHECK-NOT:    output_file
// CHECK:      firrtl.module @MT_C
// CHECK-SAME:   output_file = #hw.output_file<"gct-dir{{/|\\\\}}"
//
// CHECK-NOT:    output_file
// CHECK:      firrtl.extmodule @E__C
// CHECK-SAME:   output_file = #hw.output_file<"gct-dir{{/|\\\\}}">
// CHECK-NOT:    output_file
// CHECK:      firrtl.extmodule @ET_C
// CHECK-SAME:   output_file = #hw.output_file<"gct-dir{{/|\\\\}}">
// CHECK-NOT:    output_file
//
// CHECK:      firrtl.module

// -----

firrtl.circuit "DirectoryBehaviorWithoutDUT" attributes {
  annotations = [
    {class = "sifive.enterprise.grandcentral.AugmentedBundleType",
     defName = "Foo",
     elements = [],
     id = 0 : i64,
     name = "View"},
    {class = "sifive.enterprise.grandcentral.ExtractGrandCentralAnnotation",
     directory = "gct-dir",
     filename = "bindings.sv"}]} {

  // Each of these modules is instantiated in a different location.  A leading
  // "E" indicates that this is an external module.  A leading "M" indicates
  // that this is a module.  The instantiation location is indicated by three
  // binary bits with an "_" indicating the absence of instantiation:
  //   1) "T" indicates this is instantiated in the "Top"
  //   2) "C" indicates this is instantiated in the "Companion"
  // E.g., "E_C" is an external module instantiated only in the Companion.
  firrtl.module @MT_() {}
  firrtl.module @M_C() {}
  firrtl.module @MTC() {}
  firrtl.extmodule @ET_() attributes {annotations = [
    {class = "firrtl.transforms.BlackBoxInlineAnno", name = "ET_.v", text = ""}
  ]}
  firrtl.extmodule @E_C() attributes {annotations = [
    {class = "firrtl.transforms.BlackBoxInlineAnno", name = "E_C.v", text = ""}
  ]}
  firrtl.extmodule @ETC() attributes {annotations = [
    {class = "firrtl.transforms.BlackBoxInlineAnno", name = "ETC.v", text = ""}
  ]}

  // The Grand Central Companion module.
  firrtl.module private @Companion() attributes {
    annotations = [
      {class = "sifive.enterprise.grandcentral.ViewAnnotation.companion",
       defName = "Foo",
       id = 0 : i64,
       name = "View"}]} {

    firrtl.instance m__c @M_C()
    firrtl.instance m_dc @MTC()

    firrtl.instance e__c @E_C()
    firrtl.instance e_dc @ETC()
  }

  // This is the DUT in the previous example, but is no longer marked as the
  // DUT.
  firrtl.module @DirectoryBehaviorWithoutDUT() {
    firrtl.instance companion @Companion()

    firrtl.instance m_d_ @MT_()
    firrtl.instance m_dc @MTC()

    firrtl.instance e_d_ @ET_()
    firrtl.instance e_dc @ETC()
  }

}

// Any module instantiated by the Companion, but not instantiated by the DUT is
// moved to the same directory as the Companion.  I.e., only "*_C" modules
// should be moved into the "gct-dir".
//
// CHECK-LABEL: "DirectoryBehaviorWithoutDUT"
//
// CHECK-NOT:    output_file
// CHECK:      firrtl.module @M_C
// CHECK-SAME:   output_file = #hw.output_file<"gct-dir{{/|\\\\}}"
// CHECK-NOT:    output_file
//
// CHECK:      firrtl.extmodule @E_C
// CHECK-SAME:   output_file = #hw.output_file<"gct-dir{{/|\\\\}}">
// CHECK-NOT:    output_file
//
// CHECK:      firrtl.module

// -----

firrtl.circuit "Top" attributes {
  annotations = [
    {
      class = "sifive.enterprise.grandcentral.ExtractGrandCentralAnnotation",
      directory = ".",
      filename = "bindings.sv"
    }
  ]
} {
  firrtl.module private @Companion_w1(in %_gen_uint: !firrtl.uint<1>) {
    %view_uintrefPort = firrtl.node %_gen_uint : !firrtl.uint<1>
    firrtl.view "View_w1", <{
      class = "sifive.enterprise.grandcentral.AugmentedBundleType",
      defName = "MyInterface_w1",
      elements = [
        {
          class = "sifive.enterprise.grandcentral.AugmentedBundleType",
          defName = "SameName",
          elements = [
            {
              class = "sifive.enterprise.grandcentral.AugmentedGroundType",
              name = "uint"
            }
          ],
          name = "SameName"
        }
      ]
    }>, %view_uintrefPort : !firrtl.uint<1>
  }
  firrtl.module private @Companion_w2(in %_gen_uint: !firrtl.uint<2>) {
    %view_uintrefPort = firrtl.node %_gen_uint : !firrtl.uint<2>
    firrtl.view "View_w2", <{
      class = "sifive.enterprise.grandcentral.AugmentedBundleType",
      defName = "MyInterface_w2",
      elements = [
        {
          class = "sifive.enterprise.grandcentral.AugmentedBundleType",
          defName = "SameName",
          elements = [
            {
              class = "sifive.enterprise.grandcentral.AugmentedGroundType",
              name = "uint"
            }
          ],
          name = "SameName"
        }
      ]
    }>, %view_uintrefPort : !firrtl.uint<2>
  }
  firrtl.module private @DUT() {
    %c0_ui2 = firrtl.constant 0 : !firrtl.uint<2>
    %c0_ui1 = firrtl.constant 0 : !firrtl.uint<1>
    %a_w1 = firrtl.wire   {annotations = [{class = "firrtl.transforms.DontTouchAnnotation"}]} : !firrtl.uint<1>
    firrtl.matchingconnect %a_w1, %c0_ui1 : !firrtl.uint<1>
    %a_w2 = firrtl.wire   {annotations = [{class = "firrtl.transforms.DontTouchAnnotation"}]} : !firrtl.uint<2>
    firrtl.matchingconnect %a_w2, %c0_ui2 : !firrtl.uint<2>
    %companion_w1__gen_uint = firrtl.instance companion_w1  @Companion_w1(in _gen_uint: !firrtl.uint<1>)
    %companion_w2__gen_uint = firrtl.instance companion_w2  @Companion_w2(in _gen_uint: !firrtl.uint<2>)
    firrtl.matchingconnect %companion_w1__gen_uint, %a_w1 : !firrtl.uint<1>
    firrtl.matchingconnect %companion_w2__gen_uint, %a_w2 : !firrtl.uint<2>
  }
  firrtl.module @Top() {
    firrtl.instance dut  @DUT()
  }
}

// Check that the correct subinterface name is used when aliasing is possible.
// Here, SameName is used twice as a sub-interface name and we need to make sure
// that MyInterface_w2 uses the uniqued name of SameName.
//
// See: https://github.com/llvm/circt/issues/4234

// CHECK-LABEL:  sv.interface @MyInterface_w1 {{.+}} {
// CHECK-NEXT:     sv.verbatim "SameName SameName();"
// CHECK-NEXT:   }
// CHECK-LABEL:  sv.interface @MyInterface_w2 {{.+}} {
// CHECK-NEXT:     sv.verbatim "SameName_0 SameName();"
// CHECK-NEXT:   }

// -----

firrtl.circuit "NoInterfaces" attributes {
  annotations = [
    {class = "sifive.enterprise.grandcentral.GrandCentralHierarchyFileAnnotation",
     filename = "gct.yaml"}]} {
  firrtl.module @NoInterfaces() {}
}

// CHECK-LABEL: module {
// CHECK:         sv.verbatim
// CHECK-SAME:      []
