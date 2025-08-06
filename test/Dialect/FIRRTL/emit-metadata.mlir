// RUN: circt-opt --firrtl-emit-metadata="repl-seq-mem=true repl-seq-mem-file=mems.conf" -split-input-file %s | FileCheck %s

//===----------------------------------------------------------------------===//
// RetimeModules
//===----------------------------------------------------------------------===//

// Test that both flavors of retiming metadata are generated:
//
//   1. Class-based metadata
//   2. JSON file-based metadata
//
// This retiming information should only be generated for things in the
// "design".  This test works by instantiating modules that are wholly in the
// design, wholly _not_ in the design, or in a mixture of both.  It then expects
// that only modules which have at least one instance in the design will get
// metadata.

firrtl.circuit "TestHarness" attributes {
  annotations = [
    {
      class = "sifive.enterprise.firrtl.RetimeModulesAnnotation",
      filename = "retime_modules.json"
    }
  ]
} {
  firrtl.layer @A bind {}
  firrtl.module @Foo() attributes {
    annotations = [
      {
        class = "freechips.rocketchip.util.RetimeModuleAnnotation"
      }
    ]
  } {}
  firrtl.module @Bar() attributes {
    annotations = [
      {
        class = "freechips.rocketchip.util.RetimeModuleAnnotation"
      }
    ]
  } {}
  firrtl.module @Baz() attributes {
    annotations = [
      {
        class = "freechips.rocketchip.util.RetimeModuleAnnotation"
      }
    ]
  } {}
  firrtl.module @Qux() attributes {
    annotations = [
      {
        class = "freechips.rocketchip.util.RetimeModuleAnnotation"
      }
    ]
  } {}
  firrtl.module @Quz() attributes {
    annotations = [
      {
        class = "freechips.rocketchip.util.RetimeModuleAnnotation"
      }
    ]
  } {}
  firrtl.module @DUT() attributes {
    annotations = [
      {
        class = "sifive.enterprise.firrtl.MarkDUTAnnotation"
      }
    ]
  } {
    firrtl.instance baz @Foo()
    firrtl.instance bar @Bar()
    firrtl.instance quz1 @Quz()
    firrtl.layerblock @A {
      firrtl.instance qux @Qux()
      firrtl.instance quz2 @Quz()
    }
  }
  firrtl.module @TestHarness() {
    firrtl.instance dut @DUT()
    firrtl.instance bar @Bar()
    firrtl.instance baz @Baz()
  }
}

// (1) Class-based metadata ----------------------------------------------------
//
// CHECK:               firrtl.class @RetimeModulesSchema(
// CHECK-SAME:            in %[[moduleName_in:.+]]: !firrtl.string,
// CHECK-SAME:            out %moduleName: !firrtl.string
// CHECK-SAME:          ) {
// CHECK-NEXT:            firrtl.propassign %moduleName, %[[moduleName_in]]
// CHECK-NEXT:          }
//
// CHECK-LABEL:         firrtl.class @RetimeModulesMetadata(
// CHECK-SAME:            out %Foo_field: !firrtl.class<@RetimeModulesSchema
// CHECK-SAME:            out %Bar_field: !firrtl.class<@RetimeModulesSchema
// CHECK-SAME:            out %Quz_field: !firrtl.class<@RetimeModulesSchema
// CHECK-SAME:          ) {{.*}} {
//
// CHECK-NEXT:            %[[#name:]] = firrtl.string "Foo"
// CHECK-NEXT:            %[[schema:.+]] = firrtl.object @RetimeModulesSchema(
// CHECK-NEXT:            %[[#moduleName:]] = firrtl.object.subfield %[[schema]][moduleName_in]
// CHECK-NEXT:            firrtl.propassign %[[#moduleName]], %[[#name]]
// CHECK-NEXT:            firrtl.propassign %Foo_field, %[[schema]]
//
// CHECK-NEXT:            %[[#name:]] = firrtl.string "Bar"
// CHECK-NEXT:            %[[schema:.+]] = firrtl.object @RetimeModulesSchema(
// CHECK-NEXT:            %[[#moduleName:]] = firrtl.object.subfield %[[schema]][moduleName_in]
// CHECK-NEXT:            firrtl.propassign %[[#moduleName]], %[[#name]]
// CHECK-NEXT:            firrtl.propassign %Bar_field, %[[schema]]
//
// CHECK-NEXT:            %[[#name:]] = firrtl.string "Quz"
// CHECK-NEXT:            %[[schema:.+]] = firrtl.object @RetimeModulesSchema(
// CHECK-NEXT:            %[[#moduleName:]] = firrtl.object.subfield %[[schema]][moduleName_in]
// CHECK-NEXT:            firrtl.propassign %[[#moduleName]], %[[#name]]
// CHECK-NEXT:            firrtl.propassign %Quz_field, %[[schema]]
//
// CHECK-NEXT:          }

// (2) JSON file-based metadata ------------------------------------------------
//
// CHECK-LABEL:         emit.file "retime_modules.json"
// CHECK-NEXT:            sv.verbatim "[
// CHECK-SAME{LITERAL}:     \22{{0}}\22,\0A
// CHECK-SAME{LITERAL}:     \22{{1}}\22,\0A
// CHECK-SAME{LITERAL}:     \22{{2}}\22
// CHECK-SAME:            ]"
// CHECK-SAME:            symbols = [@Foo, @Bar, @Quz]

// -----

// Test that retime information is always emited if there is no
// design-under-test (DUT) specified.

firrtl.circuit "Foo" attributes {
  annotations = [
    {
      class = "sifive.enterprise.firrtl.RetimeModulesAnnotation",
      filename = "retime_modules.json"
    }
  ]
} {
  firrtl.module @Bar() attributes {
    annotations = [
      {
        class = "freechips.rocketchip.util.RetimeModuleAnnotation"
      }
    ]
  } {}
  firrtl.module @Foo() {
  }
}

// CHECK-LABEL:         firrtl.class @RetimeModulesMetadata(
// CHECK-SAME:            out %Bar_field: !firrtl.class<@RetimeModulesSchema

// CHECK-LABEL:         emit.file "retime_modules.json"
// CHECK-NEXT:            sv.verbatim "[
// CHECK-SAME{LITERAL}:     \22{{0}}\22
// CHECK-SAME:            ]"
// CHECK-SAME:            symbols = [@Bar]

// -----

//===----------------------------------------------------------------------===//
// SitestBlackbox
//===----------------------------------------------------------------------===//

// CHECK-LABEL: firrtl.circuit "DUTBlackboxes" {
firrtl.circuit "DUTBlackboxes" attributes {
  annotations = [
    {
      class = "sifive.enterprise.firrtl.SitestBlackBoxAnnotation",
      filename = "dut_blackboxes.json"
    }
  ]
} {
  firrtl.module @DUTBlackboxes() attributes {
    annotations = [
      {
        class = "sifive.enterprise.firrtl.MarkDUTAnnotation"
      }
    ]
  } {}
// CHECK-NOT:  emit.file ""
// CHECK:      emit.file "dut_blackboxes.json" {
// CHECK-NEXT:   emit.verbatim "[]"
// CHECK-NEXT: }
// CHECK-NOT:  emit.file ""
}

// -----

// CHECK-LABEL: firrtl.circuit "TestBlackboxes"  {
firrtl.circuit "TestBlackboxes" attributes {
  annotations = [
    {
      class = "sifive.enterprise.firrtl.SitestTestHarnessBlackBoxAnnotation",
      filename = "test_blackboxes.json"
    }
  ]
} {
  firrtl.module @TestBlackboxes() attributes {
    annotations = [
      {
        class = "sifive.enterprise.firrtl.MarkDUTAnnotation"
      }
    ]
  } {}
// CHECK-NOT:  emit.file ""
// CHECK:      emit.file "test_blackboxes.json" {
// CHECK-NEXT:   emit.verbatim "[]"
// CHECK-NEXT: }
// CHECK-NOT:  emit.file ""
}

// -----

firrtl.circuit "BasicBlackboxes" attributes {
  annotations = [
    {
      class = "sifive.enterprise.firrtl.SitestBlackBoxAnnotation",
      filename = "dut_blackboxes.json"
    },
    {
      class = "sifive.enterprise.firrtl.SitestTestHarnessBlackBoxAnnotation",
      filename = "test_blackboxes.json"
    }
  ]
} {
  firrtl.layer @A bind {}
  firrtl.module @BasicBlackboxes() attributes {
    annotations = [
      {
        class = "sifive.enterprise.firrtl.MarkDUTAnnotation"
      }
    ]
  } {
    firrtl.instance test @DUTBlackbox_0()
    firrtl.instance test @DUTBlackbox_1()
    firrtl.instance test @DUTBlackbox_2()
    firrtl.instance layerBlackboxInDesign1 @LayerBlackboxInDesign()
    firrtl.instance blacklistedWithLibsDut @InlineBlackboxWithLibs()
    firrtl.layerblock @A {
      firrtl.instance layerBlackboxInDesign2 @LayerBlackboxInDesign()
      firrtl.instance layerBlackbox @LayerBlackbox()
    }
  }

  // These should all be ignored.
  firrtl.extmodule @ignored0() attributes {
    annotations = [
      {
        class = "firrtl.transforms.BlackBoxInlineAnno"
      }
    ],
    defname = "ignored0"
  }
  firrtl.extmodule @ignored1() attributes {
    annotations = [
      {
        class = "firrtl.transforms.BlackBoxPathAnno"
      }
    ],
    defname = "ignored1"
  }
  firrtl.extmodule @ignored2() attributes {
    annotations = [
      {
        class = "sifive.enterprise.grandcentral.DataTapsAnnotation.blackbox"
      }
    ],
    defname = "ignored2"
  }
  firrtl.extmodule @ignored3() attributes {
    annotations = [
      {
        class = "sifive.enterprise.grandcentral.MemTapAnnotation.blackbox", id = 4 : i64
      }
    ],
    defname = "ignored3"
  }
  firrtl.extmodule @ignored4() attributes {
    annotations = [
      {
        class = "firrtl.transforms.BlackBox"
      }
    ],
    defname = "ignored4"
  }

  // Gracefully handle missing defnames.
  firrtl.extmodule @NoDefName()

  firrtl.extmodule @TestBlackbox() attributes {defname = "TestBlackbox"}

  // Should be de-duplicated and sorted.
  firrtl.extmodule @DUTBlackbox_0() attributes {defname = "DUTBlackbox2"}
  firrtl.extmodule @DUTBlackbox_1() attributes {defname = "DUTBlackbox1"}
  firrtl.extmodule @DUTBlackbox_2() attributes {defname = "DUTBlackbox1"}
  firrtl.extmodule @LayerBlackboxInDesign() attributes {defname = "LayerBlackboxInDesign"}
  firrtl.extmodule @LayerBlackbox() attributes {defname = "LayerBlackbox"}

  // Test blacklisted blackbox with additional libraries - should be included
  firrtl.extmodule @InlineBlackboxWithLibs() attributes {
    annotations = [
      {
        class = "firrtl.transforms.BlackBoxInlineAnno"
      },
      {
        class = "sifive.enterprise.firrtl.SitestBlackBoxLibrariesAnnotation",
        libraries = ["lib1", "lib2"]
      }
    ],
    defname = "InlineBlackboxWithLibs"
  }

  // Test non-blacklisted blackbox with additional libraries - should be included
  firrtl.extmodule @BlackboxWithLibs() attributes {
    annotations = [
      {
        class = "sifive.enterprise.firrtl.SitestBlackBoxLibrariesAnnotation",
        libraries = ["lib3", "lib4", "lib5"]
      }
    ],
    defname = "BlackboxWithLibs"
  }

  firrtl.module @TestHarness() {
    firrtl.instance inlineBlackboxWithLibs @InlineBlackboxWithLibs()
    firrtl.instance blackboxWithLibs @BlackboxWithLibs()
    firrtl.instance test @TestBlackbox()
  }
}

// (1) Class-based metadata ----------------------------------------------------
//
// CHECK:               firrtl.class @SitestBlackBoxModulesSchema(
// CHECK-SAME:            in %[[moduleName_in:[^:]+]]: !firrtl.string,
// CHECK-SAME:            out %moduleName: !firrtl.string,
// CHECK-SAME:            in %[[inDut_in:[^:]+]]: !firrtl.bool,
// CHECK-SAME:            out %inDut: !firrtl.bool,
// CHECK-SAME:            in %[[libraries_in:[^:]+]]: !firrtl.list<string>,
// CHECK-SAME:            out %libraries: !firrtl.list<string>
// CHECK-SAME:          ) {
// CHECK-NEXT:            firrtl.propassign %moduleName, %[[moduleName_in]]
// CHECK-NEXT:            firrtl.propassign %inDut, %[[inDut_in]]
// CHECK-NEXT:            firrtl.propassign %libraries, %[[libraries_in]]
// CHECK:               }
//
// CHECK:               firrtl.class @SitestBlackBoxMetadata(
// CHECK-SAME:            out %TestBlackbox_field: !firrtl.class<@SitestBlackBoxModulesSchema(
// CHECK-SAME:            out %DUTBlackbox_0_field: !firrtl.class<@SitestBlackBoxModulesSchema(
// CHECK-SAME:            out %DUTBlackbox_1_field: !firrtl.class<@SitestBlackBoxModulesSchema(
// CHECK-SAME:            out %LayerBlackboxInDesign_field: !firrtl.class<@SitestBlackBoxModulesSchema(
// CHECK-SAME:            out %LayerBlackbox_field: !firrtl.class<@SitestBlackBoxModulesSchema(
// CHECK-SAME:            out %InlineBlackboxWithLibs_field: !firrtl.class<@SitestBlackBoxModulesSchema(
// CHECK-SAME:            out %BlackboxWithLibs_field: !firrtl.class<@SitestBlackBoxModulesSchema(
// CHECK-NOT:             !firrtl.class<@SitestBlackBoxModulesSchema(
//
// CHECK-NEXT:            %[[#defname:]] = firrtl.string "TestBlackbox"
// CHECK-NEXT:            %[[#inDutVal:]] = firrtl.bool false
// CHECK-NEXT:            %[[#libsList:]] = firrtl.list.create %[[#defname]] : !firrtl.list<string>
// CHECK-NEXT:            %[[object:.+]] = firrtl.object @SitestBlackBoxModulesSchema
// CHECK-NEXT:            %[[#moduleName:]] = firrtl.object.subfield %[[object]][moduleName_in]
// CHECK-NEXT:            firrtl.propassign %[[#moduleName]], %[[#defname:]] : !firrtl.string
// CHECK-NEXT:            %[[#inDut:]] = firrtl.object.subfield %[[object]][inDut_in]
// CHECK-NEXT:            firrtl.propassign %[[#inDut]], %[[#inDutVal]] : !firrtl.bool
// CHECK-NEXT:            %[[#libraries:]] = firrtl.object.subfield %[[object]][libraries_in]
// CHECK-NEXT:            firrtl.propassign %[[#libraries]], %[[#libsList]] : !firrtl.list<string>
// CHECK-NEXT:            firrtl.propassign %TestBlackbox_field, %[[object]]
//
// CHECK-NEXT:            %[[#defname:]] = firrtl.string "DUTBlackbox2"
// CHECK-NEXT:            %[[#inDutVal:]] = firrtl.bool true
// CHECK-NEXT:            %[[#libsList:]] = firrtl.list.create %[[#defname]] : !firrtl.list<string>
// CHECK-NEXT:            %[[object:.+]] = firrtl.object @SitestBlackBoxModulesSchema
// CHECK-NEXT:            %[[#moduleName:]] = firrtl.object.subfield %[[object]][moduleName_in]
// CHECK-NEXT:            firrtl.propassign %[[#moduleName]], %[[#defname:]] : !firrtl.string
// CHECK-NEXT:            %[[#inDut:]] = firrtl.object.subfield %[[object]][inDut_in]
// CHECK-NEXT:            firrtl.propassign %[[#inDut]], %[[#inDutVal]] : !firrtl.bool
// CHECK-NEXT:            %[[#libraries:]] = firrtl.object.subfield %[[object]][libraries_in]
// CHECK-NEXT:            firrtl.propassign %[[#libraries]], %[[#libsList]] : !firrtl.list<string>
// CHECK-NEXT:            firrtl.propassign %DUTBlackbox_0_field, %[[object]]
//
// CHECK-NEXT:            %[[#defname:]] = firrtl.string "DUTBlackbox1"
// CHECK-NEXT:            %[[#inDutVal:]] = firrtl.bool true
// CHECK-NEXT:            %[[#libsList:]] = firrtl.list.create %[[#defname]] : !firrtl.list<string>
// CHECK-NEXT:            %[[object:.+]] = firrtl.object @SitestBlackBoxModulesSchema
// CHECK-NEXT:            %[[#moduleName:]] = firrtl.object.subfield %[[object]][moduleName_in]
// CHECK-NEXT:            firrtl.propassign %[[#moduleName]], %[[#defname:]] : !firrtl.string
// CHECK-NEXT:            %[[#inDut:]] = firrtl.object.subfield %[[object]][inDut_in]
// CHECK-NEXT:            firrtl.propassign %[[#inDut]], %[[#inDutVal]] : !firrtl.bool
// CHECK-NEXT:            %[[#libraries:]] = firrtl.object.subfield %[[object]][libraries_in]
// CHECK-NEXT:            firrtl.propassign %[[#libraries]], %[[#libsList]] : !firrtl.list<string>
// CHECK-NEXT:            firrtl.propassign %DUTBlackbox_1_field, %[[object]]
//
// CHECK-NEXT:            %[[#defname:]] = firrtl.string "LayerBlackboxInDesign"
// CHECK-NEXT:            %[[#inDutVal:]] = firrtl.bool true
// CHECK-NEXT:            %[[#libsList:]] = firrtl.list.create %[[#defname]] : !firrtl.list<string>
// CHECK-NEXT:            %[[object:.+]] = firrtl.object @SitestBlackBoxModulesSchema
// CHECK-NEXT:            %[[#moduleName:]] = firrtl.object.subfield %[[object]][moduleName_in]
// CHECK-NEXT:            firrtl.propassign %[[#moduleName]], %[[#defname:]] : !firrtl.string
// CHECK-NEXT:            %[[#inDut:]] = firrtl.object.subfield %[[object]][inDut_in]
// CHECK-NEXT:            firrtl.propassign %[[#inDut]], %[[#inDutVal]] : !firrtl.bool
// CHECK-NEXT:            %[[#libraries:]] = firrtl.object.subfield %[[object]][libraries_in]
// CHECK-NEXT:            firrtl.propassign %[[#libraries]], %[[#libsList]] : !firrtl.list<string>
// CHECK-NEXT:            firrtl.propassign %LayerBlackboxInDesign_field, %[[object]]
//
// CHECK-NEXT:            %[[#defname:]] = firrtl.string "LayerBlackbox"
// CHECK-NEXT:            %[[#inDutVal:]] = firrtl.bool false
// CHECK-NEXT:            %[[#libsList:]] = firrtl.list.create %[[#defname]] : !firrtl.list<string>
// CHECK-NEXT:            %[[object:.+]] = firrtl.object @SitestBlackBoxModulesSchema
// CHECK-NEXT:            %[[#moduleName:]] = firrtl.object.subfield %[[object]][moduleName_in]
// CHECK-NEXT:            firrtl.propassign %[[#moduleName]], %[[#defname:]] : !firrtl.string
// CHECK-NEXT:            %[[#inDut:]] = firrtl.object.subfield %[[object]][inDut_in]
// CHECK-NEXT:            firrtl.propassign %[[#inDut]], %[[#inDutVal]] : !firrtl.bool
// CHECK-NEXT:            %[[#libraries:]] = firrtl.object.subfield %[[object]][libraries_in]
// CHECK-NEXT:            firrtl.propassign %[[#libraries]], %[[#libsList]] : !firrtl.list<string>
// CHECK-NEXT:            firrtl.propassign %LayerBlackbox_field, %[[object]]
//
// CHECK-NEXT:            %[[#defname:]] = firrtl.string "InlineBlackboxWithLibs"
// CHECK-NEXT:            %[[#inDutVal:]] = firrtl.bool true
// CHECK-NEXT:            %[[#lib1:]] = firrtl.string "lib1"
// CHECK-NEXT:            %[[#lib2:]] = firrtl.string "lib2"
// CHECK-NEXT:            %[[#libsList:]] = firrtl.list.create %[[#lib1]], %[[#lib2]] : !firrtl.list<string>
// CHECK-NEXT:            %[[object:.+]] = firrtl.object @SitestBlackBoxModulesSchema
// CHECK-NEXT:            %[[#moduleName:]] = firrtl.object.subfield %[[object]][moduleName_in]
// CHECK-NEXT:            firrtl.propassign %[[#moduleName]], %[[#defname:]] : !firrtl.string
// CHECK-NEXT:            %[[#inDut:]] = firrtl.object.subfield %[[object]][inDut_in]
// CHECK-NEXT:            firrtl.propassign %[[#inDut]], %[[#inDutVal]] : !firrtl.bool
// CHECK-NEXT:            %[[#libraries:]] = firrtl.object.subfield %[[object]][libraries_in]
// CHECK-NEXT:            firrtl.propassign %[[#libraries]], %[[#libsList]] : !firrtl.list<string>
// CHECK-NEXT:            firrtl.propassign %InlineBlackboxWithLibs_field, %[[object]]
//
// CHECK-NEXT:            %[[#defname:]] = firrtl.string "BlackboxWithLibs"
// CHECK-NEXT:            %[[#inDutVal:]] = firrtl.bool false
// CHECK-NEXT:            %[[#lib3:]] = firrtl.string "lib3"
// CHECK-NEXT:            %[[#lib4:]] = firrtl.string "lib4"
// CHECK-NEXT:            %[[#lib5:]] = firrtl.string "lib5"
// CHECK-NEXT:            %[[#libsList:]] = firrtl.list.create %[[#defname]], %[[#lib3]], %[[#lib4]], %[[#lib5]] : !firrtl.list<string>
// CHECK-NEXT:            %[[object:.+]] = firrtl.object @SitestBlackBoxModulesSchema
// CHECK-NEXT:            %[[#moduleName:]] = firrtl.object.subfield %[[object]][moduleName_in]
// CHECK-NEXT:            firrtl.propassign %[[#moduleName]], %[[#defname:]] : !firrtl.string
// CHECK-NEXT:            %[[#inDut:]] = firrtl.object.subfield %[[object]][inDut_in]
// CHECK-NEXT:            firrtl.propassign %[[#inDut]], %[[#inDutVal]] : !firrtl.bool
// CHECK-NEXT:            %[[#libraries:]] = firrtl.object.subfield %[[object]][libraries_in]
// CHECK-NEXT:            firrtl.propassign %[[#libraries]], %[[#libsList]] : !firrtl.list<string>
// CHECK-NEXT:            firrtl.propassign %BlackboxWithLibs_field, %[[object]]
//
// CHECK-NOT:             firrtl.object

// (2) JSON file-based metadata ------------------------------------------------
//
// CHECK:               emit.file "test_blackboxes.json" {
// CHECK-NEXT{LITERAL}:   emit.verbatim "[\0A
// CHECK-SAME:              \22BlackboxWithLibs\22,\0A
// CHECK-SAME:              \22LayerBlackbox\22,\0A
// CHECK-SAME:              \22TestBlackbox\22,\0A
// CHECK-SAME:              \22lib3\22,\0A
// CHECK-SAME:              \22lib4\22,\0A
// CHECK-SAME:              \22lib5\22\0A
// CHECK-SAME:            ]"
// CHECK-NEXT:          }
//
// CHECK:               emit.file "dut_blackboxes.json" {
// CHECK-NEXT{LITERAL}:   emit.verbatim "[\0A
// CHECK-SAME:              \22DUTBlackbox1\22,\0A
// CHECK-SAME:              \22DUTBlackbox2\22,\0A
// CHECK-SAME:              \22LayerBlackboxInDesign\22,\0A
// CHECK-SAME:              \22lib1\22,\0A
// CHECK-SAME:              \22lib2\22\0A
// CHECK-SAME:            ]"
// CHECK-NEXT:          }

// -----

//===----------------------------------------------------------------------===//
// Design-under-test (DUT) Metadata
//===----------------------------------------------------------------------===//

firrtl.circuit "Foo" {
  firrtl.module @Bar() attributes {
    annotations = [
      {
        class = "sifive.enterprise.firrtl.MarkDUTAnnotation"
      }
    ]
  } {
  }
  firrtl.module @Foo() {
    firrtl.instance bar sym @bar @Bar()
  }
}

// CHECK-LABEL:         firrtl.circuit "Foo"
// CHECK:                 hw.hierpath private @[[dutPathSym:.+]] [@Foo::@bar]

// CHECK-LABEL:         firrtl.module @Foo(
// CHECK-NEXT:            firrtl.instance bar
// CHECK-SAME:              id = distinct[[[dutId:[0-9]+]]]<>

// CHECK-LABEL:         firrtl.class @SiFive_Metadata(
// CHECK-SAME:            out %[[dutModulePath:dutModulePath.*]]: !firrtl.list<path>
// CHECK:                 %[[#a:]] = firrtl.path instance distinct[[[dutId]]]<>
// CHECK-NEXT:            %[[#b:]] = firrtl.list.create %[[#a]] : !firrtl.list<path>
// CHECK-NEXT:            firrtl.propassign %[[dutModulePath]], %[[#b]] : !firrtl.list<path>

// -----

//===----------------------------------------------------------------------===//
// MemoryMetadata
//===----------------------------------------------------------------------===//

// Test behavior when no memories are present:
//
//   1. Empty JSON metadata is emitted
//   2. No OM classes are created

// CHECK-LABEL: firrtl.circuit "top"
firrtl.circuit "top"
{
  firrtl.module @top() { }
  // CHECK:      emit.file "metadata{{/|\\\\}}seq_mems.json" {
  // CHECK-NEXT:   sv.verbatim "[]"
  // CHECK-NEXT: }

  // CHECK:      emit.file "mems.conf" {
  // CHECK-NEXT:   sv.verbatim ""
  // CHECK-NEXT: }
}

// CHECK-NOT: om.class @MemorySchema

// -----

// Test that a single memory in the DUT is lowered corectly.  This tests all the
// various features of metadata emission for a single memory that is
// instantiated under the design-under-test (DUT):
//
//   1. The MemorySchema and MemoryMetadata classes are created and populated
//      with the correct information.
//   2. The memory JSON file is created with the same information as (1).
//   3. A configuration file that contains the shape of the memory.
//
// Checks are broken up to test each of these files individually.  Later tests,
// that need to check all three files use the same check structure.
//
// This does _not_ check anything related to the design-under-test (DUT) which,
// due to the fact that this test has a `MarkDUTAnnotation`, will also generate
// that metadata.  DUT metadata is checked with an earlier test.

firrtl.circuit "Foo" {
  firrtl.module private @m() {
    firrtl.instance m_ext @m_ext()
  }
  firrtl.memmodule private @m_ext() attributes {
    dataWidth = 8 : ui32,
    depth = 16 : ui64,
    extraPorts = [
      {
        direction = "input",
        name = "user_input",
        width = 5 : ui32
      }
    ],
    maskBits = 1 : ui32,
    numReadPorts = 2 : ui32,
    numWritePorts = 3 : ui32,
    numReadWritePorts = 4 : ui32,
    readLatency = 1 : ui32,
    writeLatency = 1 : ui32
  }
  firrtl.module @Baz() {
    firrtl.instance m sym @m @m()
  }
  firrtl.module @Bar() attributes {
    annotations = [
      {
        class = "sifive.enterprise.firrtl.MarkDUTAnnotation"
      }
    ]
  } {
    firrtl.instance baz sym @baz @Baz()
  }
  firrtl.module @Foo() {
    firrtl.instance bar sym @bar @Bar()
  }
}

// (1) OM Info -----------------------------------------------------------------
// CHECK-LABEL:         firrtl.circuit "Foo"
// CHECK:                 hw.hierpath private @[[memPathSym:.+]] [@Bar::@baz, @Baz::@m]

// CHECK-LABEL:         firrtl.module @Baz()
// CHECK-NEXT:            firrtl.instance m
// CHECK-SAME:              id = distinct[[[#memId:]]]<>

// CHECK-LABEL:         firrtl.class @MemorySchema(
// CHECK-NEXT:            firrtl.propassign %name, %name_in
// CHECK-NEXT:            firrtl.propassign %depth, %depth_in
// CHECK-NEXT:            firrtl.propassign %width, %width_in
// CHECK-NEXT:            firrtl.propassign %maskBits, %maskBits_in
// CHECK-NEXT:            firrtl.propassign %readPorts, %readPorts_in
// CHECK-NEXT:            firrtl.propassign %writePorts, %writePorts_in
// CHECK-NEXT:            firrtl.propassign %readwritePorts, %readwritePorts_in
// CHECK-NEXT:            firrtl.propassign %writeLatency, %writeLatency_in
// CHECK-NEXT:            firrtl.propassign %readLatency, %readLatency_in
// CHECK-NEXT:            firrtl.propassign %ruwBehavior, %ruwBehavior_in
// CHECK-NEXT:            firrtl.propassign %hierarchy, %hierarchy_in
// CHECK-NEXT:            firrtl.propassign %inDut, %inDut_in
// CHECK-NEXT:            firrtl.propassign %extraPorts, %extraPorts_in
// CHECK-NEXT:            firrtl.propassign %preExtInstName, %preExtInstName_in

// CHECK-LABEL:         firrtl.class @MemoryMetadata({{.*$}}
// CHECK-NEXT:            %0 = firrtl.string "m_ext"
// CHECK-NEXT:            %1 = firrtl.path instance distinct[[[#memId]]]<>
// CHECK-NEXT:            %2 = firrtl.list.create %0
// CHECK-NEXT:            %3 = firrtl.list.create %1
// CHECK-NEXT:            %[[memoryObject:.+]] = firrtl.object @MemorySchema
// CHECK-NEXT:            %4 = firrtl.string "user_input"
// CHECK-NEXT:            %5 = firrtl.string "input"
// CHECK-NEXT:            %6 = firrtl.integer 5
// CHECK-NEXT:            %[[extraPortsObject:.+]] = firrtl.object @ExtraPortsMemorySchema
// CHECK-NEXT:            %7 = firrtl.object.subfield %[[extraPortsObject]][name_in]
// CHECK-NEXT:            firrtl.propassign %7, %4
// CHECK-NEXT:            %8 = firrtl.object.subfield %[[extraPortsObject]][direction_in]
// CHECK-NEXT:            firrtl.propassign %8, %5
// CHECK-NEXT:            %9 = firrtl.object.subfield %[[extraPortsObject]][width_in]
// CHECK-NEXT:            firrtl.propassign %9, %6
// CHECK-NEXT:            %10 = firrtl.list.create %[[extraPortsObject]]
// CHECK-NEXT:            %11 = firrtl.string "m_ext"
// CHECK-NEXT:            %12 = firrtl.object.subfield %[[memoryObject]][name_in]
// CHECK-NEXT:            firrtl.propassign %12, %11
// CHECK-NEXT:            %13 = firrtl.integer 16
// CHECK-NEXT:            %14 = firrtl.object.subfield %[[memoryObject]][depth_in]
// CHECK-NEXT:            firrtl.propassign %14, %13
// CHECK-NEXT:            %15 = firrtl.integer 8
// CHECK-NEXT:            %16 = firrtl.object.subfield %[[memoryObject]][width_in]
// CHECK-NEXT:            firrtl.propassign %16, %15
// CHECK-NEXT:            %17 = firrtl.integer 1
// CHECK-NEXT:            %18 = firrtl.object.subfield %[[memoryObject]][maskBits_in]
// CHECK-NEXT:            firrtl.propassign %18, %17
// CHECK-NEXT:            %19 = firrtl.integer 2
// CHECK-NEXT:            %20 = firrtl.object.subfield %[[memoryObject]][readPorts_in]
// CHECK-NEXT:            firrtl.propassign %20, %19
// CHECK-NEXT:            %21 = firrtl.integer 3
// CHECK-NEXT:            %22 = firrtl.object.subfield %[[memoryObject]][writePorts_in]
// CHECK-NEXT:            firrtl.propassign %22, %21
// CHECK-NEXT:            %23 = firrtl.integer 4
// CHECK-NEXT:            %24 = firrtl.object.subfield %[[memoryObject]][readwritePorts_in]
// CHECK-NEXT:            firrtl.propassign %24, %23
// CHECK-NEXT:            %25 = firrtl.integer 1
// CHECK-NEXT:            %26 = firrtl.object.subfield %[[memoryObject]][writeLatency_in]
// CHECK-NEXT:            firrtl.propassign %26, %25
// CHECK-NEXT:            %27 = firrtl.integer 1
// CHECK-NEXT:            %28 = firrtl.object.subfield %[[memoryObject]][readLatency_in]
// CHECK-NEXT:            firrtl.propassign %28, %27
// CHECK-NEXT:            %29 = firrtl.string "Undefined"
// CHECK-NEXT:            %30 = firrtl.object.subfield %[[memoryObject]][ruwBehavior_in]
// CHECK-NEXT:            firrtl.propassign %30, %29
// CHECK-NEXT:            %31 = firrtl.object.subfield %[[memoryObject]][hierarchy_in]
// CHECK-NEXT:            firrtl.propassign %31, %3
// CHECK-NEXT:            %32 = firrtl.bool true
// CHECK-NEXT:            %33 = firrtl.object.subfield %[[memoryObject]][inDut_in]
// CHECK-NEXT:            firrtl.propassign %33, %32
// CHECK-NEXT:            %34 = firrtl.object.subfield %[[memoryObject]][extraPorts_in]
// CHECK-NEXT:            firrtl.propassign %34, %10
// CHECK-NEXT:            %35 = firrtl.object.subfield %[[memoryObject]][preExtInstName_in]
// CHECK-NEXT:            firrtl.propassign %35, %2
// CHECK-NEXT:            firrtl.propassign %[[memoryObject]]_field, %[[memoryObject]]

// (2) Memory JSON -------------------------------------------------------------

// CHECK-LABEL:         emit.file "metadata{{/|\\\\}}seq_mems.json"
// CHECK-NEXT:            sv.verbatim
// CHECK-SAME{LITERAL}:     \22module_name\22: \22{{0}}\22
// CHECK-SAME:              \22depth\22: 16
// CHECK-SAME:              \22width\22: 8
// CHECK-SAME:              \22masked\22: false
// CHECK-SAME:              \22read\22: 2
// CHECK-SAME:              \22write\22: 3
// CHECK-SAME:              \22readwrite\22: 4
// CHECK-SAME:              \22ruw_behavior\22: \22Undefined\22
// CHECK-SAME:              \22extra_ports\22: [
// CHECK-SAME:                {
// CHECK-SAME:                  \22name\22: \22user_input\22
// CHECK-SAME:                  \22direction\22: \22input\22
// CHECK-SAME:                  \22width\22: 5
// CHECK-SAME:                }
// CHECK-SAME:              ]
// CHECK-SAME:              \22hierarchy\22: [
// CHECK-SAME{LITERAL}:       \22{{3}}.{{4}}.{{5}}.m_ext\22
// CHECK-SAME:              ]
// CHECK-SAME:              symbols = [@m_ext, @Foo, #hw.innerNameRef<@Foo::@bar>, @Bar, #hw.innerNameRef<@Bar::@baz>, #hw.innerNameRef<@Baz::@m>]

// (3) Configuration File ------------------------------------------------------

// CHECK-LABEL:         emit.file "mems.conf"
// CHECK-NEXT{LITERAL}:   sv.verbatim "name {{0}} depth 16 width 8 ports write,write,write,read,read,rw,rw,rw,rw\0A"
// CHECK-SAME:            symbols = [@m_ext]

// -----

// Test that zero read, write, and read-write ports produce sane output.

firrtl.circuit "Foo" {
  firrtl.module private @m() {
    firrtl.instance m_ext @m_ext()
  }
  firrtl.memmodule private @m_ext() attributes {
    dataWidth = 8 : ui32,
    depth = 16 : ui64,
    extraPorts = [],
    maskBits = 1 : ui32,
    numReadPorts = 0 : ui32,
    numWritePorts = 0 : ui32,
    numReadWritePorts = 0 : ui32,
    readLatency = 1 : ui32,
    writeLatency = 1 : ui32
  }
  firrtl.module @Baz() {
    firrtl.instance m sym @m @m()
  }
  firrtl.module @Bar() attributes {
    annotations = [
      {
        class = "sifive.enterprise.firrtl.MarkDUTAnnotation"
      }
    ]
  } {
    firrtl.instance baz sym @baz @Baz()
  }
  firrtl.module @Foo() {
    firrtl.instance bar sym @bar @Bar()
  }
}

// (1) OM Info -----------------------------------------------------------------
// CHECK-LABEL:         firrtl.class @MemoryMetadata({{.*$}}
// CHECK:                 %[[memoryObject:.+]] = firrtl.object @MemorySchema
// CHECK:                 %[[#zero:]] = firrtl.integer 0
// CHECK-NEXT:            %[[#r:]] = firrtl.object.subfield %[[memoryObject]][readPorts_in]
// CHECK-NEXT:            firrtl.propassign %[[#r]], %[[#zero]]
// CHECK-NEXT:            %[[#zero:]] = firrtl.integer 0
// CHECK-NEXT:            %[[#w:]] = firrtl.object.subfield %[[memoryObject]][writePorts_in]
// CHECK-NEXT:            firrtl.propassign %[[#w]], %[[#zero]]
// CHECK-NEXT:            %[[#zero:]] = firrtl.integer 0
// CHECK-NEXT:            %[[#rw:]] = firrtl.object.subfield %[[memoryObject]][readwritePorts_in]
// CHECK-NEXT:            firrtl.propassign %[[#rw]], %[[#zero]]

// (2) Memory JSON -------------------------------------------------------------
// CHECK-LABEL:         emit.file "metadata{{/|\\\\}}seq_mems.json"
// CHECK-NEXT:            sv.verbatim
// CHECK-SAME:              \22read\22: 0
// CHECK-SAME:              \22write\22: 0
// CHECK-SAME:              \22readwrite\22: 0

// (3) Configuration File ------------------------------------------------------
// CHECK-LABEL:         emit.file "mems.conf"
// CHECK-NEXT{LITERAL}:   sv.verbatim "name {{0}} depth 16 width 8 ports \0A"

// -----

// Test that a read-only memory produces metadata.

firrtl.circuit "Foo" {
  firrtl.module private @m() {
    firrtl.instance m_ext @m_ext()
  }
  firrtl.memmodule private @m_ext() attributes {
    dataWidth = 8 : ui32,
    depth = 16 : ui64,
    extraPorts = [],
    maskBits = 1 : ui32,
    numReadPorts = 1 : ui32,
    numWritePorts = 0 : ui32,
    numReadWritePorts = 0 : ui32,
    readLatency = 1 : ui32,
    writeLatency = 1 : ui32
  }
  firrtl.module @Baz() {
    firrtl.instance m sym @m @m()
  }
  firrtl.module @Bar() attributes {
    annotations = [
      {
        class = "sifive.enterprise.firrtl.MarkDUTAnnotation"
      }
    ]
  } {
    firrtl.instance baz sym @baz @Baz()
  }
  firrtl.module @Foo() {
    firrtl.instance bar sym @bar @Bar()
  }
}

// (1) OM Info -----------------------------------------------------------------
// CHECK-LABEL:         firrtl.class @MemoryMetadata({{.*$}}
// CHECK:                 %[[memoryObject:.+]] = firrtl.object @MemorySchema
// CHECK:                 firrtl.object.subfield %[[memoryObject]][maskBits_in]
// CHECK:                 %[[#one:]] = firrtl.integer 1
// CHECK-NEXT:            %[[#r:]] = firrtl.object.subfield %[[memoryObject]][readPorts_in]
// CHECK-NEXT:            firrtl.propassign %[[#r]], %[[#one]]
// CHECK-NEXT:            %[[#zero:]] = firrtl.integer 0
// CHECK-NEXT:            %[[#w:]] = firrtl.object.subfield %[[memoryObject]][writePorts_in]
// CHECK-NEXT:            firrtl.propassign %[[#w]], %[[#zero]]
// CHECK-NEXT:            %[[#zero:]] = firrtl.integer 0
// CHECK-NEXT:            %[[#rw:]] = firrtl.object.subfield %[[memoryObject]][readwritePorts_in]
// CHECK-NEXT:            firrtl.propassign %[[#rw]], %[[#zero]]

// (2) Memory JSON -------------------------------------------------------------
// CHECK-LABEL:         emit.file "metadata{{/|\\\\}}seq_mems.json"
// CHECK-NEXT:            sv.verbatim
// CHECK-SAME:              \22read\22: 1
// CHECK-SAME:              \22write\22: 0
// CHECK-SAME:              \22readwrite\22: 0

// (3) Configuration File ------------------------------------------------------
// CHECK-LABEL:         emit.file "mems.conf"
// CHECK-NEXT{LITERAL}:   sv.verbatim "name {{0}} depth 16 width 8 ports read\0A"

// -----

// Test that a memory that is not readLatency=1 and writeLatency=1 produces OM
// metadata, but not Memory JSON or Configuration File metadata.

firrtl.circuit "Foo" {
  firrtl.module private @m() {
    firrtl.instance m_ext @m_ext()
  }
  firrtl.memmodule private @m_ext() attributes {
    dataWidth = 8 : ui32,
    depth = 16 : ui64,
    extraPorts = [],
    maskBits = 1 : ui32,
    numReadPorts = 1 : ui32,
    numWritePorts = 0 : ui32,
    numReadWritePorts = 0 : ui32,
    readLatency = 42 : ui32,
    writeLatency = 9001 : ui32
  }
  firrtl.module @Baz() {
    firrtl.instance m sym @m @m()
  }
  firrtl.module @Bar() attributes {
    annotations = [
      {
        class = "sifive.enterprise.firrtl.MarkDUTAnnotation"
      }
    ]
  } {
    firrtl.instance baz sym @baz @Baz()
  }
  firrtl.module @Foo() {
    firrtl.instance bar sym @bar @Bar()
  }
}

// CHECK-LABEL:         firrtl.module @Baz()
// CHECK-NEXT:            firrtl.instance m
// CHECK-SAME:              id = distinct[[[#memId:]]]<>

// (1) OM Info -----------------------------------------------------------------
// CHECK-LABEL:         firrtl.class @MemoryMetadata({{.*$}}
// CHECK:                 %[[#memIdPath:]] = firrtl.path instance distinct[[[#memId]]]<>
// CHECK:                 firrtl.list.create %[[#memIdPath]]
// CHECK:                 %[[memoryObject:.+]] = firrtl.object @MemorySchema
// CHECK:                 %[[#a:]] = firrtl.integer 9001
// CHECK-NEXT:            %[[#writeLatency:]] = firrtl.object.subfield %[[memoryObject]][writeLatency_in]
// CHECK-NEXT:            firrtl.propassign %[[#writeLatency]], %[[#a]]
// CHECK-NEXT:            %[[#b:]] = firrtl.integer 42
// CHECK-NEXT:            %[[#readLatency:]] = firrtl.object.subfield %[[memoryObject]][readLatency_in]
// CHECK-NEXT:            firrtl.propassign %[[#readLatency]], %[[#b]]

// (2) Memory JSON -------------------------------------------------------------
// CHECK-LABEL:         emit.file "metadata{{/|\\\\}}seq_mems.json"
// CHECK-NEXT:            sv.verbatim "[]"

// (3) Configuration File ------------------------------------------------------
// CHECK-LABEL:         emit.file "mems.conf"
// CHECK-NEXT{LITERAL}:   sv.verbatim ""

// -----

// Memories that are outside the design should have empty OM hierarchies, empty
// JSON files, and populated configuration files.
//
// This test is checking three "outside the design" situations:
//
//   1. @m1 is instantiated in the test-harness
//   2. @m2 is instantiated under a layer in the test harness
//   3. @m3 is instantiated under a layer in the design

firrtl.circuit "Foo" {
  firrtl.layer @A bind {}
  firrtl.memmodule private @m1() attributes {
    dataWidth = 8 : ui32,
    depth = 16 : ui64,
    extraPorts = [],
    maskBits = 1 : ui32,
    numReadPorts = 1 : ui32,
    numWritePorts = 0 : ui32,
    numReadWritePorts = 0 : ui32,
    readLatency = 1 : ui32,
    writeLatency = 1 : ui32
  }
  firrtl.memmodule private @m2() attributes {
    dataWidth = 8 : ui32,
    depth = 32 : ui64,
    extraPorts = [],
    maskBits = 1 : ui32,
    numReadPorts = 1 : ui32,
    numWritePorts = 0 : ui32,
    numReadWritePorts = 0 : ui32,
    readLatency = 1 : ui32,
    writeLatency = 1 : ui32
  }
  firrtl.memmodule private @m3() attributes {
    dataWidth = 8 : ui32,
    depth = 64 : ui64,
    extraPorts = [],
    maskBits = 1 : ui32,
    numReadPorts = 1 : ui32,
    numWritePorts = 0 : ui32,
    numReadWritePorts = 0 : ui32,
    readLatency = 1 : ui32,
    writeLatency = 1 : ui32
  }
  firrtl.module @Foo() {
    firrtl.instance m1 @m1()
    firrtl.layerblock @A {
      firrtl.instance m2 @m2()
    }
    firrtl.instance bar sym @bar @Bar()
  }
  firrtl.module @Bar() attributes {
    annotations = [
      {
        class = "sifive.enterprise.firrtl.MarkDUTAnnotation"
      }
    ]
  } {
    firrtl.layerblock @A {
      firrtl.instance m3 @m3()
    }
  }
}

// (1) OM Info -----------------------------------------------------------------
// No distinct annotations are added to memory instances.
//
// CHECK-LABEL:         firrtl.module @Foo(
// CHECK-NEXT:            firrtl.instance m1
// CHECK-NOT:               id = distinct
// CHECK-SAME:              @m1()
// CHECK-NEXT:            firrtl.layerblock @A {
// CHECK-NEXT:              firrtl.instance m2
// CHECK-NOT:                 id = distinct
// CHECK-SAME:                @m2()
//
// CHECK-LABEL:         firrtl.module @Bar()
// CHECK-NEXT:            firrtl.layerblock @A {
// CHECK-NEXT:              firrtl.instance m3
// CHECK-NOT:                 id = distinct
// CHECK-SAME:                @m3()
//
// Use empty paths for the memories.
//
// CHECK-LABEL:         firrtl.class @MemoryMetadata({{.*$}}
// CHECK:                 %[[#pathList:]] = firrtl.list.create : !firrtl.list<path>
// CHECK:                 %[[memoryObject:.+]] = firrtl.object @MemorySchema(
// CHECK:                 %[[#memPaths:]] = firrtl.object.subfield %[[memoryObject]][hierarchy_in]
// CHECK-NEXT:            firrtl.propassign %[[#memPaths]], %[[#pathList]]
//
// CHECK:                 %[[#pathList:]] = firrtl.list.create : !firrtl.list<path>
// CHECK:                 %[[memoryObject:.+]] = firrtl.object @MemorySchema(
// CHECK:                 %[[#memPaths:]] = firrtl.object.subfield %[[memoryObject]][hierarchy_in]
// CHECK-NEXT:            firrtl.propassign %[[#memPaths]], %[[#pathList]]
//
// CHECK:                 %[[#pathList:]] = firrtl.list.create : !firrtl.list<path>
// CHECK:                 %[[memoryObject:.+]] = firrtl.object @MemorySchema(
// CHECK:                 %[[#memPaths:]] = firrtl.object.subfield %[[memoryObject]][hierarchy_in]
// CHECK-NEXT:            firrtl.propassign %[[#memPaths]], %[[#pathList]]

// (2) Memory JSON -------------------------------------------------------------
// CHECK-LABEL:         emit.file "metadata{{/|\\\\}}seq_mems.json"
// CHECK-NEXT:            sv.verbatim "[]"

// (3) Configuration File ------------------------------------------------------
// CHECK-LABEL:         emit.file "mems.conf"
// CHECK-NEXT:            sv.verbatim
// CHECK-SAME{LITERAL}:    name {{0}} depth 16 width 8 ports read\0A
// CHECK-SAME{LITERAL}:    name {{1}} depth 32 width 8 ports read\0A
// CHECK-SAME{LITERAL}:    name {{2}} depth 64 width 8 ports read\0A
// CHECK-SAME:             symbols = [@m1, @m2, @m3]

// -----

// Test that a memory that is instantiated both in the design and not in the
// design produces the correct metadata.  Test the following combiations of in
// and out of the design:
//
//   1. @m1 is testharness and design
//   2. @m2 is layer and design

firrtl.circuit "Foo" {
  firrtl.layer @A bind {}
  firrtl.memmodule private @m1() attributes {
    dataWidth = 8 : ui32,
    depth = 16 : ui64,
    extraPorts = [],
    maskBits = 1 : ui32,
    numReadPorts = 1 : ui32,
    numWritePorts = 0 : ui32,
    numReadWritePorts = 0 : ui32,
    readLatency = 1 : ui32,
    writeLatency = 1 : ui32
  }
  firrtl.module private @m1_ext() {
    firrtl.instance m @m1()
  }
  firrtl.memmodule private @m2() attributes {
    dataWidth = 8 : ui32,
    depth = 32 : ui64,
    extraPorts = [],
    maskBits = 1 : ui32,
    numReadPorts = 1 : ui32,
    numWritePorts = 0 : ui32,
    numReadWritePorts = 0 : ui32,
    readLatency = 1 : ui32,
    writeLatency = 1 : ui32
  }
  firrtl.module private @m2_ext() {
    firrtl.instance m @m2()
  }
  firrtl.module @Foo() {
    firrtl.instance m1 @m1_ext()
    firrtl.instance bar sym @bar @Bar()
  }
  firrtl.module @Bar() attributes {
    annotations = [
      {
        class = "sifive.enterprise.firrtl.MarkDUTAnnotation"
      }
    ]
  } {
    firrtl.instance m1 sym @m1 @m1_ext()
    firrtl.instance m2_1 sym @m2_1 @m2_ext()
    firrtl.layerblock @A {
      firrtl.instance m2_2 sym @m2_2 @m2_ext()
    }
  }
}

// (1) OM Info -----------------------------------------------------------------
//
// Hierarchical paths are added for the two paths in the design.
//
// CHECK-LABEL:         firrtl.circuit "Foo"
// CHECK-DAG:             hw.hierpath private @[[hierPath_m1:.+]] [@Bar::@m1]
// CHECK-DAG:             hw.hierpath private @[[hierPath_m2_1:.+]] [@Bar::@m2_1]
//
// Trackers are added only for instances in the design.
//
// CHECK-LABEL:         firrtl.module @Foo(
// CHECK-NEXT:            firrtl.instance m1
// CHECK-NOT:               id = distinct
// CHECK-SAME:              @m1
//
// CHECK-LABEL:         firrtl.module @Bar()
// CHECK-NEXT:            firrtl.instance m1
// CHECK-SAME:              {circt.nonlocal = @[[hierPath_m1]], class = "circt.tracker", id = distinct[[[#m1Id:]]]<>}
// CHECK-SAME:              @m1_ext()
// CHECK-NEXT:            firrtl.instance m2_1
// CHECK-SAME:              {circt.nonlocal = @[[hierPath_m2_1]], class = "circt.tracker", id = distinct[[[#m2_1Id:]]]<>}
// CHECK-SAME:              @m2_ext()
// CHECK-NEXT:            firrtl.layerblock @A {
// CHECK-NEXT:              firrtl.instance m2_2
// CHECK-NOT:                 id = distinct
// CHECK-SAME:                @m2_ext()
//
// This uses an unresolvable path for the memory that is not in the design.  An
// unresolvable path is one which references a tracker with an ID that does not
// exist.  This relies on later passes to interpret this as "optimized away" and
// not emit it.  This is admittedly janky and should be cleaned up---there's no
// point in generating this and putting it in the path list if the path is
// definitely unresolvable.
//
// CHECK-LABEL:         firrtl.class @MemoryMetadata({{.*$}}
//
// CHECK:                 %[[#memIdPath:]] = firrtl.path instance distinct[[[#m1Id]]]<>
// CHECK:                 %[[#unresolvablePath:]] = firrtl.path reference distinct[[[#]]]<>
// CHECK:                 %[[#pathList:]] = firrtl.list.create %[[#memIdPath]], %[[#unresolvablePath]] : !firrtl.list<path>
// CHECK:                 %[[memoryObject:.+]] = firrtl.object @MemorySchema(
// CHECK:                 %[[#memPaths:]] = firrtl.object.subfield %[[memoryObject]][hierarchy_in]
// CHECK-NEXT:            firrtl.propassign %[[#memPaths]], %[[#pathList]]
//
// CHECK:                 %[[#unresolvablePath:]] = firrtl.path reference distinct[[[#]]]<>
// CHECK:                 %[[#memIdPath:]] = firrtl.path instance distinct[[[#m2_1Id]]]<>
// CHECK:                 %[[#pathList:]] = firrtl.list.create %[[#unresolvablePath]], %[[#memIdPath]] : !firrtl.list<path>
// CHECK:                 %[[memoryObject:.+]] = firrtl.object @MemorySchema(
// CHECK:                 %[[#memPaths:]] = firrtl.object.subfield %[[memoryObject]][hierarchy_in]
// CHECK-NEXT:            firrtl.propassign %[[#memPaths]], %[[#pathList]]

// (2) Memory JSON -------------------------------------------------------------
// CHECK-LABEL:         emit.file "metadata{{/|\\\\}}seq_mems.json"
// CHECK-NEXT:            sv.verbatim "[
// CHECK-SAME:              \22hierarchy\22: [
// CHECK-SAME{LITERAL}:       \22{{3}}.{{4}}.m\22
// CHECK-SAME{LITERAL}:       \22{{3}}.{{6}}.m\22
// CHECK-SAME:              ]
// The regex `{{(@|\#)[^,]+,}}` is matching a symbol or inner name ref.
// CHECK-SAME:              symbols = [{{(@|\#)[^,]+,}} {{(@|\#)[^,]+,}} {{(@|\#)[^,]+,}} @Bar, #hw.innerNameRef<@Bar::@m1>, {{(@|\#)[^,]+,}} #hw.innerNameRef<@Bar::@m2_1>

// (3) Configuration File ------------------------------------------------------
// CHECK-LABEL:         emit.file "mems.conf"
// CHECK-NEXT:            sv.verbatim
// CHECK-SAME{LITERAL}:     name {{0}} depth 16 width 8 ports read\0
// CHECK-SAME{LITERAL}:     name {{1}} depth 32 width 8 ports read\0
// CHECK-SAME:              symbols = [@m1, @m2]

// -----

// Test behavior of multiple memories.

firrtl.circuit "Foo" {
  firrtl.memmodule private @m2() attributes {
    dataWidth = 8 : ui32,
    depth = 32 : ui64,
    extraPorts = [],
    maskBits = 1 : ui32,
    numReadPorts = 1 : ui32,
    numWritePorts = 0 : ui32,
    numReadWritePorts = 0 : ui32,
    readLatency = 1 : ui32,
    writeLatency = 1 : ui32
  }
  firrtl.memmodule private @m1() attributes {
    dataWidth = 8 : ui32,
    depth = 16 : ui64,
    extraPorts = [],
    maskBits = 1 : ui32,
    numReadPorts = 0 : ui32,
    numWritePorts = 1 : ui32,
    numReadWritePorts = 0 : ui32,
    readLatency = 1 : ui32,
    writeLatency = 1 : ui32
  }
  firrtl.module @Bar() attributes {
    annotations = [
      {
        class = "sifive.enterprise.firrtl.MarkDUTAnnotation"
      }
    ]
  } {
    firrtl.instance m sym @m1 @m1()
    firrtl.instance m sym @m2 @m2()
  }
  firrtl.module @Foo() {
    firrtl.instance bar sym @bar @Bar()
  }
}

//------------------------------------------------------------------ (1) OM Info
// CHECK-LABEL:         firrtl.class @MemoryMetadata({{.*$}}
// CHECK-COUNT-2:         %{{.+}} = firrtl.object @MemorySchema(
// CHECK-NOT:             %{{.+}} = firrtl.object @MemorySchema(

//-------------------------------------------------------------- (2) Memory JSON
// CHECK-LABEL:         emit.file "metadata{{/|\\\\}}seq_mems.json"
// CHECK-NEXT:            sv.verbatim "[
// CHECK-COUNT-2:           \22hierarchy\22: [
// CHECK-NOT:               \22hierarchy\22: [

//------------------------------------------------------- (3) Configuration File
// CHECK-LABEL:         emit.file "mems.conf"
// CHECK-NEXT:            sv.verbatim
// CHECK-SAME{LITERAL}:     name {{0}} depth 32 width 8 ports read\0A
// CHECK-SAME{LITERAL}:     name {{1}} depth 16 width 8 ports write\0A
// CHECK-SAME:              symbols = [@m2, @m1]


// -----

// Test defined read-under-write behavior.

firrtl.circuit "RUWOld" {
  firrtl.memmodule private @mOld() attributes {
    dataWidth = 8 : ui32,
    depth = 32 : ui64,
    extraPorts = [],
    maskBits = 1 : ui32,
    numReadPorts = 1 : ui32,
    numWritePorts = 1 : ui32,
    numReadWritePorts = 0 : ui32,
    readLatency = 1 : ui32,
    writeLatency = 1 : ui32,
    ruw = #firrtl<ruwbehavior Old>
  }
  firrtl.module @RUWOld() attributes {
    annotations = [
      {
        class = "sifive.enterprise.firrtl.MarkDUTAnnotation"
      }
    ]
  } {
    firrtl.instance m sym @mOld @mOld()
  }
}

//------------------------------------------------------------------ (1) OM Info
// CHECK-LABEL:         firrtl.class @MemoryMetadata({{.*$}}
// CHECK:                 %[[memoryObject:.+]] = firrtl.object @MemorySchema
// CHECK:                 %[[#a:]] = firrtl.string "Old"
// CHECK-NEXT:            %[[#ruw:]] = firrtl.object.subfield %[[memoryObject]][ruwBehavior_in]
// CHECK-NEXT:            firrtl.propassign %[[#ruw]], %[[#a]]

//-------------------------------------------------------------- (2) Memory JSON
// CHECK-LABEL:         emit.file "metadata{{/|\\\\}}seq_mems.json"
// CHECK-NEXT:            sv.verbatim "[
// CHECK:                  \22ruw_behavior\22: \22Old\22

//------------------------------------------------------- (3) Configuration File
// CHECK-LABEL:         emit.file "mems.conf"
// CHECK-NEXT:            sv.verbatim
// CHECK-SAME:            ruw Old

// -----

firrtl.circuit "RUWNew" {
  firrtl.memmodule private @mNew() attributes {
    dataWidth = 8 : ui32,
    depth = 32 : ui64,
    extraPorts = [],
    maskBits = 1 : ui32,
    numReadPorts = 1 : ui32,
    numWritePorts = 1 : ui32,
    numReadWritePorts = 0 : ui32,
    readLatency = 1 : ui32,
    writeLatency = 1 : ui32,
    ruw = #firrtl<ruwbehavior New>
  }
  firrtl.module @RUWNew() attributes {
    annotations = [
      {
        class = "sifive.enterprise.firrtl.MarkDUTAnnotation"
      }
    ]
  } {
    firrtl.instance m sym @mNew @mNew()
  }
}

//------------------------------------------------------------------ (1) OM Info
// CHECK-LABEL:         firrtl.class @MemoryMetadata({{.*$}}
// CHECK:                 %[[memoryObject:.+]] = firrtl.object @MemorySchema
// CHECK:                 %[[#a:]] = firrtl.string "New"
// CHECK-NEXT:            %[[#ruw:]] = firrtl.object.subfield %[[memoryObject]][ruwBehavior_in]
// CHECK-NEXT:            firrtl.propassign %[[#ruw]], %[[#a]]

//-------------------------------------------------------------- (2) Memory JSON
// CHECK-LABEL:         emit.file "metadata{{/|\\\\}}seq_mems.json"
// CHECK-NEXT:            sv.verbatim "[
// CHECK:                  \22ruw_behavior\22: \22New\22

//------------------------------------------------------- (3) Configuration File
// CHECK-LABEL:         emit.file "mems.conf"
// CHECK-NEXT:            sv.verbatim
// CHECK-SAME:            ruw New
