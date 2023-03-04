; RUN: firtool %s -annotation-file %s.sitestblackboxes.anno.json | FileCheck %s -check-prefixes=CHECK,SITEST_NODUT
; RUN: firtool %s -annotation-file %s.sitestblackboxes.anno.json -annotation-file %s.markdut.anno.json | FileCheck %s -check-prefixes=CHECK,SITEST_DUT
; RUN: firtool %s -repl-seq-mem -repl-seq-mem-file=mems.conf -disable-all-randomization -disable-opt | FileCheck %s -check-prefixes=CHECK,MEMS_NODUT
; RUN: firtool %s -repl-seq-mem -repl-seq-mem-file=mems.conf -disable-all-randomization -disable-opt -annotation-file %s.markdut.anno.json | FileCheck %s -check-prefixes=CHECK,MEMS_DUT

; CHECK: module Foo
circuit TestHarness:
  ; Foo* are instantiated only by the TestHarness
  module Foo:

    mem foo_m :
      data-type => UInt<8>
      depth => 1
      reader => r
      writer => w
      read-latency => 1
      write-latency => 1
      read-under-write => undefined

    foo_m is invalid

  extmodule Foo_BlackBox:
    defname = Foo_BlackBox

  ; Bar* are instantiated only by the DUT
  module Bar:

    mem bar_m :
      data-type => UInt<8>
      depth => 2
      reader => r
      writer => w
      read-latency => 1
      write-latency => 1
      read-under-write => undefined

    bar_m is invalid

  extmodule Bar_BlackBox:
    defname = Bar_BlackBox

  ; Baz* are instantiated by both the TestHarness and the DUT
  module Baz:

    mem baz_m :
      data-type => UInt<8>
      depth => 3
      reader => r
      writer => w
      read-latency => 1
      write-latency => 1
      read-under-write => undefined

    baz_m is invalid

  extmodule Baz_BlackBox:
    defname = Baz_BlackBox

  ; This is the design-under-test.  This is marked as such by a separate,
  ; optional annotation file.
  module DUT:

    inst bar of Bar
    inst bar_bbox of Bar_BlackBox
    inst baz of Baz
    inst baz_bbox of Baz_BlackBox

  ; This is the Test Harness, i.e., the top of the design.
  module TestHarness:

    inst foo of Foo
    inst foo_bbox of Foo_BlackBox
    inst dut of DUT
    inst baz of Baz
    inst baz_bbox of Baz_BlackBox


; SITEST_DUT:       FILE "testbench.sitest.json"
; SITEST-DUT-NOT:   FILE
; SITEST_DUT:         "Foo_BlackBox"

; SITEST_DUT:       FILE "design.sitest.json"
; SITEST_DUT-NOT:   FILE
; SITEST_DUT-DAG:     "Bar_BlackBox",
; SITEST_DUT-DAG:     "Baz_BlackBox"

; SITEST_NODUT:     FILE "testbench.sitest.json"
; SITEST-NODUT-NOT: FILE

; SITEST_NODUT:     FILE "design.sitest.json"
; SITEST_NODUT-NOT: FILE
; SITEST_NODUT-DAG:   "Foo_BlackBox"
; SITEST_NODUT-DAG:   "Bar_BlackBox"{{,?}}
; SITEST_NODUT-DAG:   "Baz_BlackBox"{{,?}}

; MEMS_DUT:         FILE "metadata{{[/\]}}seq_mems.json"
; MEMS_DUT-NOT:     FILE
; MEMS_DUT-DAG:       "DUT.bar.bar_m.bar_m_ext"
; MEMS_DUT-DAG:       "DUT.baz.baz_m.baz_m_ext"

; MEMS_NODUT:       FILE "metadata{{[/\]}}seq_mems.json"
; MEMS_NODUT-NOT:   FILE
; MEMS_NODUT-DAG:    "TestHarness.foo.foo_m.foo_m_ext"
; MEMS_NODUT-DAG:    "TestHarness.dut.bar.bar_m.bar_m_ext"
; MEMS_NODUT-DAG:    "TestHarness.dut.baz.baz_m.baz_m_ext"
; MEMS_NODUT-DAG:    "TestHarness.baz.baz_m.baz_m_ext"
