; RUN: firtool %s | FileCheck %s --implicit-check-not FILE

; This test checks output directory behavior of circuits with layers with other
; features.  Generally, layer code is treated as "verification" code and should
; be kept out of the design area when `firtool` is run with options where the
; design area is explicitly specified.
;
; This test is expected to eventually be removed once the FIRRTL ABI switches to
; using filelists or manifests and a flat output directory structure.

FIRRTL version 4.0.0
circuit Testbench: %[[
  {
    "class": "sifive.enterprise.grandcentral.ViewAnnotation",
    "name": "MyView",
    "companion": "~Testbench|GrandCentral",
    "parent": "~Testbench|Testbench",
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
              "class": "sifive.enterprise.grandcentral.GrandCentralView_UnknownGroundType_"
            }
          }
        }
      ]
    }
  },
  {
    "class": "sifive.enterprise.grandcentral.ExtractGrandCentralAnnotation",
    "directory": "gct",
    "filename": "gct/bindings.sv"
  },
  {
    "class":"firrtl.transforms.DontTouchAnnotation",
    "target":"~Testbench|DUT>a"
  },
  {
    "class":"firrtl.transforms.DontTouchAnnotation",
    "target":"~Testbench|Foo>a"
  },
  {
    "class":"firrtl.transforms.DontTouchAnnotation",
    "target":"~Testbench|Bar>a"
  },
  {
    "class":"firrtl.transforms.DontTouchAnnotation",
    "target":"~Testbench|Baz>a"
  },
  {
    "class":"firrtl.transforms.DontTouchAnnotation",
    "target":"~Testbench|Qux>a"
  },
  {
    "class":"firrtl.transforms.DontTouchAnnotation",
    "target":"~Testbench|Quz>a"
  },
  {
    "class": "sifive.enterprise.firrtl.MarkDUTAnnotation",
    "target": "~Testbench|DUT"
  },
  {
    "class": "sifive.enterprise.firrtl.TestBenchDirAnnotation",
    "dirname": "verification/testbench"
  }
]]
  layer A, bind, "verification/testbench":

  module Foo:
    node a = UInt<1>(0)
  module Bar:
    node a = UInt<1>(1)
  module Baz:
    node a = UInt<2>(2)

  module Qux:
    node a = UInt<2>(3)
  module Quz:
    node a = UInt<3>(4)

  module GrandCentral:
    inst qux of Qux
    inst quz of Quz

  public module DUT:
    inst foo of Foo
    inst baz of Baz
    inst grandCentral of GrandCentral

    node a = UInt<3>(4)

    layerblock A:
      inst bar of Bar
      inst bazz of Baz
      inst quz of Quz

  public module Testbench:
    inst dut of DUT

; The only modules NOT in a file are Foo, Baz, and DUT.
;
; CHECK: module Foo();
; CHECK: module Baz();
; CHECK: module DUT();

; The following modules are all in the Testbench directory.
;
; CHECK-DAG: FILE "verification{{[/\]}}testbench{{[/\]}}layers-Testbench-A.sv"
; CHECK-DAG: FILE "verification{{[/\]}}testbench{{[/\]}}Bar.sv"
; CHECK-DAG: FILE "verification{{[/\]}}testbench{{[/\]}}DUT_A.sv"
; CHECK-DAG: FILE "verification{{[/\]}}testbench{{[/\]}}Testbench.sv"

; The following modules are all in the Grand Central directory.
;
; CHECK-DAG: FILE "gct{{[/\]}}GrandCentral.sv"
; CHECK-DAG: FILE "gct{{[/\]}}Interface.sv"
; CHECK-DAG: FILE "gct{{[/\]}}Qux.sv"
; CHECK-DAG: FILE "gct{{[/\]}}Quz.sv"
; CHECK-DAG: FILE "gct{{[/\]}}bindings.sv"
