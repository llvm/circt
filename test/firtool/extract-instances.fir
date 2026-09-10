; RUN: firtool -split-input-file -strip-debug-info %s | FileCheck %s

; End-to-end tests of instance extraction.  This tests combinations of the
; `moveDut` behavior along with whether or not a grouping module is specified.
; This only tests blackboxes (not clock gates or memories) as this keeps the
; test terser.  This assumes that this generalizes to memories and clock gates
; correctly.

FIRRTL version 5.1.0
circuit Top: %[[
  {
    "class": "sifive.enterprise.firrtl.TestBenchDirAnnotation",
    "dirname": "testbench"
  },
  {
    "class":"sifive.enterprise.firrtl.MarkDUTAnnotation",
    "target":"~|Foo"
  },
  {
    "class":"sifive.enterprise.firrtl.InjectDUTHierarchyAnnotation",
    "name":"Logic",
    "moveDut": false
  },
  {
    "class": "sifive.enterprise.firrtl.ExtractBlackBoxAnnotation",
    "target": "~|BlackBox",
    "filename": "BlackBoxes.txt",
    "prefix": "bb",
    "dest": "BlackBoxes"
  }
]]

  extmodule BlackBox:
    output a: UInt<1>

  module Bar:
    output a: UInt<1>

    inst blackbox of BlackBox
    connect a, blackbox.a

  module Foo:
    output a: UInt<1>

    inst bar of Bar
    connect a, bar.a

  public module Top:
    output a: UInt<1>

    inst foo of Foo
    connect a, foo.a

; CHECK-NOT:   FILE "testbench{{.*}}Logic.sv"
; CHECK-LABEL: module Logic(
; CHECK-NOT:   module
; CHECK:         Bar bar (

; CHECK-NOT:   FILE "testbench{{.*}}BlackBoxes.sv"
; CHECK-LABEL: module BlackBoxes(
; CHECK-NOT:   module
; CHECK:         BlackBox blackbox (

; CHECK-NOT:   FILE "testbench{{.*}}Foo.sv"
; CHECK-LABEL: module Foo(
; CHECK-NOT:   module
; CHECK:         BlackBoxes BlackBoxes (
; CHECK-NOT:   module
; CHECK:         Logic Logic

; // -----

FIRRTL version 5.1.0
circuit Top: %[[
  {
    "class": "sifive.enterprise.firrtl.TestBenchDirAnnotation",
    "dirname": "testbench"
  },
  {
    "class":"sifive.enterprise.firrtl.MarkDUTAnnotation",
    "target":"~|Foo"
  },
  {
    "class":"sifive.enterprise.firrtl.InjectDUTHierarchyAnnotation",
    "name":"Logic",
    "moveDut": true
    },
  {
    "class": "sifive.enterprise.firrtl.ExtractBlackBoxAnnotation",
    "target": "~|BlackBox",
    "filename": "BlackBoxes.txt",
    "prefix": "bb",
    "dest": "BlackBoxes"
  }
]]

  extmodule BlackBox:
    output a: UInt<1>

  module Bar:
    output a: UInt<1>

    inst blackbox of BlackBox
    connect a, blackbox.a

  module Foo:
    output a: UInt<1>

    inst bar of Bar
    connect a, bar.a

  public module Top:
    output a: UInt<1>

    inst foo of Foo
    connect a, foo.a

; CHECK-NOT:   FILE "testbench{{.*}}Foo.sv"
; CHECK-LABEL: module Foo(
; CHECK-NOT:   module
; CHECK:         Bar bar (

; CHECK:       FILE "testbench{{.*}}BlackBoxes.sv"
; CHECK-LABEL: module BlackBoxes(
; CHECK-NOT:   module
; CHECK:         BlackBox blackbox (

; CHECK:       FILE "testbench{{.*}}Logic.sv"
; CHECK-LABEL: module Logic(
; CHECK-NOT:   module
; CHECK:         BlackBoxes BlackBoxes (
; CHECK-NOT:   module
; CHECK:         Foo Foo

; // -----

FIRRTL version 5.1.0
circuit Top: %[[
  {
    "class": "sifive.enterprise.firrtl.TestBenchDirAnnotation",
    "dirname": "testbench"
  },
  {
    "class":"sifive.enterprise.firrtl.MarkDUTAnnotation",
    "target":"~|Foo"
  },
  {
    "class":"sifive.enterprise.firrtl.InjectDUTHierarchyAnnotation",
    "name":"Logic",
    "moveDut": false
    },
  {
    "class": "sifive.enterprise.firrtl.ExtractBlackBoxAnnotation",
    "target": "~|BlackBox",
    "filename": "BlackBoxes.txt",
    "prefix": "bb"
  }
]]

  extmodule BlackBox:
    output a: UInt<1>

  module Bar:
    output a: UInt<1>

    inst blackbox of BlackBox
    connect a, blackbox.a

  module Foo:
    output a: UInt<1>

    inst bar of Bar
    connect a, bar.a

  public module Top:
    output a: UInt<1>

    inst foo of Foo
    connect a, foo.a

; CHECK-NOT:   FILE "testbench{{.*}}Logic.sv"
; CHECK-LABEL: module Logic(
; CHECK-NOT:   module
; CHECK:         Bar bar (

; CHECK-NOT:   FILE "testbench{{.*}}Foo.sv"
; CHECK-LABEL: module Foo(
; CHECK-NOT:   module
; CHECK:         Logic Logic

; CHECK:       FILE "testbench{{.*}}Top.sv"
; CHECK-LABEL: module Top(
; CHECK-NOT:   module
; CHECK:         Foo foo (
; CHECK-NOT:   module
; CHECK:         BlackBox blackbox (

; // -----

FIRRTL version 5.1.0
circuit Top: %[[
  {
    "class": "sifive.enterprise.firrtl.TestBenchDirAnnotation",
    "dirname": "testbench"
  },
  {
    "class":"sifive.enterprise.firrtl.MarkDUTAnnotation",
    "target":"~|Foo"
  },
  {
    "class":"sifive.enterprise.firrtl.InjectDUTHierarchyAnnotation",
    "name":"Logic",
    "moveDut": true
    },
  {
    "class": "sifive.enterprise.firrtl.ExtractBlackBoxAnnotation",
    "target": "~|BlackBox",
    "filename": "BlackBoxes.txt",
    "prefix": "bb"
  }
]]

  extmodule BlackBox:
    output a: UInt<1>

  module Bar:
    output a: UInt<1>

    inst blackbox of BlackBox
    connect a, blackbox.a

  module Foo:
    output a: UInt<1>

    inst bar of Bar
    connect a, bar.a

  public module Top:
    output a: UInt<1>

    inst foo of Foo
    connect a, foo.a

; CHECK-NOT:   FILE "testbench{{.*}}Foo.sv"
; CHECK-LABEL: module Foo(
; CHECK-NOT:   module
; CHECK:         Bar bar (

; CHECK:       FILE "testbench{{.*}}Logic.sv"
; CHECK-LABEL: module Logic(
; CHECK-NOT:   module
; CHECK:         Foo Foo (
; CHECK-NOT:   module
; CHECK:         BlackBox blackbox (
