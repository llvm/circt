; RUN: firtool -split-input-file %s | FileCheck %s

; This test is checking that FIRRTL external modules which have BlackBoxInline
; annotations want to write to the same file write to the proper output
; directory.  All external modules _have different parameters_ so that
; deduplication is blocked.
;
; The test enumerates all possible combinations for a simple design that
; consists of a TestHarness top module, a DUT instantiated by the TestHarness,
; and a Grand Central companion module inside the DUT.  Additionally, both the
; DUT and the TestHarness have layerblocks of the Verification color.  This is
; summarized by the table below. A one ("1") in the "T" column indicates it is
; instantiated by the TestHarness, a one in the "D" column indicates it is
; instantiated by the DUT, and a one in the "G" column indicates it is
; instantiated by the Grand Central companion.  A zero ("0") indicates it is not
; instantiated.  An "L" indicates that it is instantiated in a layer in that
; module.
;
;            T D G Output
;     --------------------
;     Foo    0 0 1 verification/gct/Foo.sv
;     Bar    0 1 0 Bar.sv
;     Baz    0 1 1 Baz.sv
;     Qux    1 0 0 verification/testbench/Qux.sv
;     Quz    1 0 1 verification/Quz.sv
;     Corge  1 1 0 Corge.sv
;     Grault 1 1 1 Grault
;     Bazola 0 L 0 Bazola.sv
;     Ztesch L 0 0 verification/testbench/Ztesch.sv
;     Thud   L L 0 Thud.sv
;
; CHECK-LABEL: module DUT
;
; CHECK: FILE "verification{{[/\]}}gct{{[/\]}}Foo.sv"
; CHECK: FILE ".{{[/\]}}Bar.sv"
; CHECK: FILE "Baz.sv"
; CHECK: FILE "verification{{[/\]}}testbench{{[/\]}}Qux.sv"
; CHECK: FILE "verification{{[/\]}}Quz.sv"
; CHECK: FILE "Corge.sv"
; CHECK: FILE "Grault.sv"
; CHECK: FILE "verification{{[/\]}}assert{{[/\]}}Bazola.sv"
; CHECK: FILE "verification{{[/\]}}assert{{[/\]}}Ztesch.sv"
; CHECK: FILE "verification{{[/\]}}assert{{[/\]}}Thud.sv"

FIRRTL version 4.0.0
circuit TestHarness: %[[
  {
    "class": "sifive.enterprise.grandcentral.ViewAnnotation",
    "name": "MyView",
    "companion": "~TestHarness|GrandCentral",
    "parent": "~TestHarness|TestHarness",
    "view": {
      "class": "sifive.enterprise.grandcentral.AugmentedBundleType",
      "defName": "Interface",
      "elements": [
        {
          "name": "uint",
          "description": "a wire called 'uint'",
          "tpe": {
            "class": "sifive.enterprise.grandcentral.AugmentedGroundType",
            "ref": {
              "module": "DUT",
              "path": [],
              "ref": "a",
              "component": []
            },
            "tpe": {
              "class": "sifive.enterprise.grandcentral.GrandCentralView$UnknownGroundType$"
            }
          }
        }
      ]
    }
  },
  {
    "class": "sifive.enterprise.firrtl.MarkDUTAnnotation",
    "target": "~TestHarness|DUT"
  },
  {
    "class": "sifive.enterprise.grandcentral.ExtractGrandCentralAnnotation",
    "directory": "verification/gct",
    "filename": "bindings.sv"
  },
  {
    "class": "sifive.enterprise.firrtl.TestBenchDirAnnotation",
    "dirname": "verification/testbench"
  },
  {
    "class": "firrtl.transforms.BlackBoxInlineAnno",
    "target": "~TestHarness|BlackBox_Foo_GCT",
    "name": "Foo.sv",
    "text": "module Foo #(parameter X=hello)(output a);\nendmodule"
  },
  {
    "class": "firrtl.transforms.BlackBoxInlineAnno",
    "target": "~TestHarness|BlackBox_Bar_DUT",
    "name": "Bar.sv",
    "text": "module Bar #(parameter X=hello)(output a);\nendmodule"
  },
  {
    "class": "firrtl.transforms.BlackBoxInlineAnno",
    "target": "~TestHarness|BlackBox_Baz_DUT",
    "name": "Baz.sv",
    "text": "module Bar #(parameter X=hello)(output a);\nendmodule"
  },
  {
    "class": "firrtl.transforms.BlackBoxInlineAnno",
    "target": "~TestHarness|BlackBox_Baz_GCT",
    "name": "Baz.sv",
    "text": "module Baz #(parameter X=hello)(output a);\nendmodule"
  },
  {
    "class": "firrtl.transforms.BlackBoxInlineAnno",
    "target": "~TestHarness|BlackBox_Qux_TestHarness",
    "name": "Qux.sv",
    "text": "module Baz #(parameter X=hello)(output a);\nendmodule"
  },
  {
    "class": "firrtl.transforms.BlackBoxInlineAnno",
    "target": "~TestHarness|BlackBox_Quz_TestHarness",
    "name": "Quz.sv",
    "text": "module Qux #(parameter X=hello)(output a);\nendmodule"
  },
  {
    "class": "firrtl.transforms.BlackBoxInlineAnno",
    "target": "~TestHarness|BlackBox_Quz_GCT",
    "name": "Quz.sv",
    "text": "module Quz #(parameter X=hello)(output a);\nendmodule"
  },
  {
    "class": "firrtl.transforms.BlackBoxInlineAnno",
    "target": "~TestHarness|BlackBox_Corge_TestHarness",
    "name": "Corge.sv",
    "text": "module Quz #(parameter X=hello)(output a);\nendmodule"
  },
  {
    "class": "firrtl.transforms.BlackBoxInlineAnno",
    "target": "~TestHarness|BlackBox_Corge_DUT",
    "name": "Corge.sv",
    "text": "module Corge #(parameter X=hello)(output a);\nendmodule"
  },
  {
    "class": "firrtl.transforms.BlackBoxInlineAnno",
    "target": "~TestHarness|BlackBox_Grault_TestHarness",
    "name": "Grault.sv",
    "text": "module Corge #(parameter X=hello)(output a);\nendmodule"
  },
  {
    "class": "firrtl.transforms.BlackBoxInlineAnno",
    "target": "~TestHarness|BlackBox_Grault_DUT",
    "name": "Grault.sv",
    "text": "module Grault #(parameter X=hello)(output a);\nendmodule"
  },
  {
    "class": "firrtl.transforms.BlackBoxInlineAnno",
    "target": "~TestHarness|BlackBox_Grault_GCT",
    "name": "Grault.sv",
    "text": "module Grault #(parameter X=hello)(output a);\nendmodule"
  },
  {
    "class": "firrtl.transforms.BlackBoxInlineAnno",
    "target": "~TestHarness|BlackBox_Bazola_DUT",
    "name": "Bazola.sv",
    "text": "module Bazola #(parameter X=hello)(output a);\nendmodule"
  },
  {
    "class": "firrtl.transforms.BlackBoxInlineAnno",
    "target": "~TestHarness|BlackBox_Ztesch_TestHarness",
    "name": "Ztesch.sv",
    "text": "module Ztesch #(parameter X=hello)(output a);\nendmodule"
  },
  {
    "class": "firrtl.transforms.BlackBoxInlineAnno",
    "target": "~TestHarness|BlackBox_Thud_TestHarness",
    "name": "Thud.sv",
    "text": "module Thud #(parameter X=hello)(output a);\nendmodule"
  },
  {
    "class": "firrtl.transforms.BlackBoxInlineAnno",
    "target": "~TestHarness|BlackBox_Thud_DUT",
    "name": "Thud.sv",
    "text": "module Thud #(parameter X=hello)(output a);\nendmodule"
  }
]]
  layer Assert, bind, "verification/assert":

  extmodule BlackBox_Foo_GCT:
    output a: UInt<1>
    defname = Foo
    parameter X = "Foo_GCT"

  extmodule BlackBox_Bar_DUT:
    output a: UInt<1>
    defname = Bar
    parameter X = "Bar_DUT"

  extmodule BlackBox_Baz_DUT:
    output a: UInt<1>
    defname = Baz
    parameter X = "Baz_DUT"
  extmodule BlackBox_Baz_GCT:
    output a: UInt<1>
    defname = Baz
    parameter X = "Baz_GCT"

  extmodule BlackBox_Qux_TestHarness:
    output a: UInt<1>
    defname = Qux
    parameter X = "Qux_TestHarness"

  extmodule BlackBox_Quz_TestHarness:
    output a: UInt<1>
    defname = Quz
    parameter X = "Quz_TestHarness"
  extmodule BlackBox_Quz_GCT:
    output a: UInt<1>
    defname = Quz
    parameter X = "Quz_GCT"

  extmodule BlackBox_Corge_TestHarness:
    output a: UInt<1>
    defname = Corge
    parameter X = "Corge_TestHarness"
  extmodule BlackBox_Corge_DUT:
    output a: UInt<1>
    defname = Corge
    parameter X = "Corge_DUT"

  extmodule BlackBox_Grault_TestHarness:
    output a: UInt<1>
    defname = Grault
    parameter X = "Grault_TestHarness"
  extmodule BlackBox_Grault_DUT:
    output a: UInt<1>
    defname = Grault
    parameter X = "Grault_DUT"
  extmodule BlackBox_Grault_GCT:
    output a: UInt<1>
    defname = Grault
    parameter X = "Grault_GCT"

  extmodule BlackBox_Bazola_DUT:
    output a: UInt<1>
    defname = Bazola
    parameter X = "Bazola_TestHarness"

  extmodule BlackBox_Ztesch_TestHarness:
    output a: UInt<1>
    defname = Ztesch
    parameter X = "Ztesch_TestHarness"

  extmodule BlackBox_Thud_TestHarness:
    output a: UInt<1>
    defname = Thud
    parameter X = "Thud_TestHarness"
  extmodule BlackBox_Thud_DUT:
    output a: UInt<1>
    defname = Thud
    parameter X = "Thud_DUT"

  public module TestHarness:

    inst dut of DUT

    inst blackBox_Qux_TestHarness of BlackBox_Qux_TestHarness
    inst blackBox_Quz_TestHarness of BlackBox_Quz_TestHarness
    inst blackBox_Corge_TestHarness of BlackBox_Corge_TestHarness
    inst blackBox_Grault_TestHarness of BlackBox_Grault_TestHarness

    layerblock Assert:
      inst blackBox_Ztesch_TestHarness of BlackBox_Ztesch_TestHarness
      inst blackBox_Thud_TestHarness of BlackBox_Thud_TestHarness

  module DUT:

    wire a: UInt<1>
    invalidate a

    inst grandCentral of GrandCentral

    inst blackBox_Bar_DUT of BlackBox_Bar_DUT
    inst blackBox_Baz_DUT of BlackBox_Baz_DUT
    inst blackBox_Corge_DUT of BlackBox_Corge_DUT
    inst blackBox_Grault_DUT of BlackBox_Grault_DUT

    layerblock Assert:
      inst blackBox_Bazola_DUT of BlackBox_Bazola_DUT
      inst blackBox_Thud_DUT of BlackBox_Thud_DUT

  module GrandCentral:

    inst blackBox_Foo_GCT of BlackBox_Foo_GCT
    inst blackBox_Baz_GCT of BlackBox_Baz_GCT
    inst blackBox_Quz_GCT of BlackBox_Quz_GCT
    inst blackBox_Grault_GCT of BlackBox_Grault_GCT

; // -----

; This test is checking the behavior of how blackboxes which _cannot_
; deduplicate with each other because they have differing parameters, but must
; be "deduplicated" on disk.  The only thing marking them as needing to
; deduplicate is that the have the same "name" field in their annotations.  This
; should write files to the LCA of the two directories that the external modules
; are supposed to be written to.  Specifically, this is checking for correct
; interactions between `AssignOutputDirs` and `BlackBoxReader`.
;
; CHECK-LABEL: module BlackBoxLCA
;
; CHECK: FILE "Bar.sv"

FIRRTL version 4.0.0
circuit BlackBoxLCA: %[[
  {
    "class": "firrtl.transforms.BlackBoxInlineAnno",
    "target": "~BlackBoxLCA|Foo",
    "name": "Bar.sv",
    "text": "Bar definition that takes parameters"
  },
  {
    "class": "firrtl.transforms.BlackBoxInlineAnno",
    "target": "~BlackBoxLCA|Bar",
    "name": "Bar.sv",
    "text": "Bar definition that takes parameters"
  }
]]
  layer A, bind, "A":
  layer B, bind, "B":

  extmodule Bar:
    input a: UInt<1>
    defname = Bar
    parameter x = 1

  extmodule Foo:
    input a: UInt<1>
    defname = Bar
    parameter x = 0


  public module BlackBoxLCA:
    input a: UInt<1>

    layerblock A:
      inst foo of Foo
      connect foo.a, a

    layerblock B:
      inst bar of Bar
      connect bar.a, a

; // -----

; When a black box module is instantiated under both the design and the
; testbench, the black box must be placed in the design (the testbench can refer
; to modules in the design, but the design must not depend on the testbench).

FIRRTL version 4.0.0

circuit TestHarness: %[[
  {
    "class": "firrtl.transforms.BlackBoxInlineAnno",
    "target": "~TestHarness|BlackBox",
    "name": "BlackBox.sv",
    "text": "module BlackBox();\nendmodule"
  },
  {
    "class": "sifive.enterprise.firrtl.MarkDUTAnnotation",
    "target": "~TestHarness|Component"
  },
  {
    "class": "sifive.enterprise.firrtl.TestBenchDirAnnotation",
    "dirname": "testbench"
  }
]]
  ;; CHECK: FILE ".{{/|\\}}BlackBox.sv"
  extmodule BlackBox:
    defname = BlackBox

  public module Component:
    inst black_box of BlackBox
  
  public module TestHarness:
    inst component of Component
    inst black_box of BlackBox
