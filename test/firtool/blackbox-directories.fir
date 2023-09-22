; RUN: firtool %s | FileCheck %s

; This test is checking that FIRRTL external modules which have BlackBoxInline
; annotations want to write to the same file write to the proper output
; directory.  All external modules _have different parameters_ so that
; deduplication is blocked.
;
; The test enumerates all possible combinations for a simple design that
; consists of a TestHarness top module, a DUT instantiated by the TestHarness,
; and a Grand Central companion module inside the DUT.  This is summarized by
; the table below. A one in the "T" column indicates it is instantiated by the
; TestHarness, a one in the "D" column indicates it is instantiated by the DUT,
; and a one in the "G" column indicates it is instantiated by the Grand Central
; companion.
;
;            T D G Output
;     --------------------
;     Foo    0 0 1 gct/Foo.sv
;     Bar    0 1 0 Bar.sv
;     Baz    0 1 1 Baz.sv
;     Qux    1 0 0 testbench/Qux.sv
;     Quz    1 0 1 gct/Quz.sv
;     Corge  1 1 0 Corge.sv
;     Grault 1 1 1 Grault
;
; CHECK-LABEL: module DUT
;
; CHECK: FILE "gct{{[/\]}}Foo.sv"
; CHECK: FILE ".{{[/\]}}Bar.sv"
; CHECK: FILE ".{{[/\]}}Baz.sv"
; CHECK: FILE "testbench{{[/\]}}Qux.sv"
; CHECK: FILE "gct{{[/\]}}Quz.sv"
; CHECK: FILE ".{{[/\]}}Corge.sv"
; CHECK: FILE ".{{[/\]}}Grault.sv"

FIRRTL version 3.0.0
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
              "circuit": "TestHarness",
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
    "directory": "gct",
    "filename": "bindings.sv"
  },
  {
    "class": "sifive.enterprise.firrtl.TestBenchDirAnnotation",
    "dirname": "testbench"
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
  }
]]
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

  module TestHarness:

    inst dut of DUT

    inst blackBox_Qux_TestHarness of BlackBox_Qux_TestHarness
    inst blackBox_Quz_TestHarness of BlackBox_Quz_TestHarness
    inst blackBox_Corge_TestHarness of BlackBox_Corge_TestHarness
    inst blackBox_Grault_TestHarness of BlackBox_Grault_TestHarness

  module DUT:

    wire a: UInt<1>
    invalidate a

    inst grandCentral of GrandCentral

    inst blackBox_Bar_DUT of BlackBox_Bar_DUT
    inst blackBox_Baz_DUT of BlackBox_Baz_DUT
    inst blackBox_Corge_DUT of BlackBox_Corge_DUT
    inst blackBox_Grault_DUT of BlackBox_Grault_DUT

  module GrandCentral:

    inst blackBox_Foo_GCT of BlackBox_Foo_GCT
    inst blackBox_Baz_GCT of BlackBox_Baz_GCT
    inst blackBox_Quz_GCT of BlackBox_Quz_GCT
    inst blackBox_Grault_GCT of BlackBox_Grault_GCT
