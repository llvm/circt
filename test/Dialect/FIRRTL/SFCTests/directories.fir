; RUN: firtool %s -disable-all-randomization -disable-opt -annotation-file %s.sitestblackboxes.anno.json | FileCheck %s -check-prefixes=CHECK,SITEST_NODUT
; RUN: firtool %s -disable-all-randomization -disable-opt -annotation-file %s.sitestblackboxes.anno.json -annotation-file %s.markdut.anno.json | FileCheck %s -check-prefixes=CHECK,SITEST_DUT
; RUN: firtool %s -repl-seq-mem -repl-seq-mem-file=mems.conf -disable-all-randomization -disable-opt | FileCheck %s -check-prefixes=CHECK,MEMS_NODUT
; RUN: firtool %s -repl-seq-mem -repl-seq-mem-file=mems.conf -disable-all-randomization -disable-opt -annotation-file %s.markdut.anno.json | FileCheck %s -check-prefixes=CHECK,MEMS_DUT
; RUN: firtool %s -disable-all-randomization -disable-opt -annotation-file %s.memtoregofvec.anno.json | FileCheck %s -check-prefixes=CHECK,MEMTOREG_NODUT
; RUN: firtool %s -disable-all-randomization -disable-opt -annotation-file %s.memtoregofvec.anno.json -annotation-file %s.markdut.anno.json | FileCheck %s -check-prefixes=CHECK,MEMTOREG_DUT
; RUN: firtool %s -disable-all-randomization -disable-opt -repl-seq-mem -repl-seq-mem-file=mems.conf -annotation-file %s.sitestblackboxes.anno.json --ir-verilog | FileCheck %s -check-prefix=MLIR_OUT

FIRRTL version 4.0.0
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

    mem foo_combmem :
      data-type => UInt<8>
      depth => 1
      reader => r
      writer => w
      read-latency => 0
      write-latency => 1
      read-under-write => undefined

    invalidate foo_m
    invalidate foo_combmem

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

    mem bar_combmem :
      data-type => UInt<8>
      depth => 2
      reader => r
      writer => w
      read-latency => 0
      write-latency => 1
      read-under-write => undefined

    invalidate bar_m
    invalidate bar_combmem

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

    mem baz_combmem :
      data-type => UInt<8>
      depth => 3
      reader => r
      writer => w
      read-latency => 0
      write-latency => 1
      read-under-write => undefined

    invalidate baz_m
    invalidate baz_combmem

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
  public module TestHarness:

    inst foo of Foo
    inst foo_bbox of Foo_BlackBox
    inst dut of DUT
    inst baz of Baz
    inst baz_bbox of Baz_BlackBox


; CHECK:            module Foo()
; MEMTOREG_DUT-NOT:   reg [7:0] foo_combmem
; MEMTOREG_NODUT:     reg [7:0] foo_combmem
; CHECK:            endmodule

; CHECK:            module Bar()
; MEMTOREG_DUT:       reg [7:0] bar_combmem
; MEMTOREG_NODUT:     reg [7:0] bar_combmem
; CHECK:            endmodule

; CHECK:            module Baz()
; MEMTOREG_DUT:       reg [7:0] baz_combmem
; MEMTOREG_NODUT:     reg [7:0] baz_combmem
; CHECK:            endmodule

; SITEST_DUT:       FILE "testbench.sitest.json"
; SITEST_DUT-NOT:   FILE
; SITEST_DUT:         "Foo_BlackBox"

; SITEST_DUT:       FILE "design.sitest.json"
; SITEST_DUT-NOT:   FILE
; SITEST_DUT-DAG:     "Bar_BlackBox",
; SITEST_DUT-DAG:     "Baz_BlackBox"

; SITEST_NODUT:     FILE "testbench.sitest.json"
; SITEST_NODUT-NOT: FILE

; MLIR_OUT:  om.class @SitestBlackBoxModulesSchema(%basepath: !om.basepath, %moduleName_in: !om.string, %inDut_in: i1) -> (moduleName: !om.string, inDut: i1) {
; MLIR_OUT:    om.class.fields %moduleName_in, %inDut_in : !om.string, i1
; MLIR_OUT:  }

; MLIR_OUT:  om.class @SitestBlackBoxMetadata(%basepath: !om.basepath) -> [[V1:.+]]: !om.class.type<@SitestBlackBoxModulesSchema>, [[V2:.+]]: !om.class.type<@SitestBlackBoxModulesSchema>, [[V3:.+]]: !om.class.type<@SitestBlackBoxModulesSchema>
; MLIR_OUT-DAG:    [[STR1:%.+]] = om.constant "Foo_BlackBox" : !om.string
; MLIR_OUT-DAG:    [[DUT1:%.+]] = om.constant true
; MLIR_OUT-DAG:    [[OBJ1:%.+]] = om.object @SitestBlackBoxModulesSchema(%basepath, [[STR1]], [[DUT1]]) : (!om.basepath, !om.string, i1) -> !om.class.type<@SitestBlackBoxModulesSchema>
; MLIR_OUT-DAG:    [[STR2:%.+]] = om.constant "Bar_BlackBox" : !om.string
; MLIR_OUT-DAG:    [[DUT2:%.+]] = om.constant true
; MLIR_OUT-DAG:    [[OBJ2:%.+]] = om.object @SitestBlackBoxModulesSchema(%basepath, [[STR2]], [[DUT2]]) : (!om.basepath, !om.string, i1) -> !om.class.type<@SitestBlackBoxModulesSchema>
; MLIR_OUT-DAG:    [[STR3:%.+]] = om.constant "Baz_BlackBox" : !om.string
; MLIR_OUT-DAG:    [[DUT3:%.+]] = om.constant true
; MLIR_OUT-DAG:    [[OBJ3:%.+]] = om.object @SitestBlackBoxModulesSchema(%basepath, [[STR3]], [[DUT3]]) : (!om.basepath, !om.string, i1) -> !om.class.type<@SitestBlackBoxModulesSchema>
; MLIR_OUT:    om.class.fields [[OBJ1]], [[OBJ2]], [[OBJ3]] : !om.class.type<@SitestBlackBoxModulesSchema>, !om.class.type<@SitestBlackBoxModulesSchema>, !om.class.type<@SitestBlackBoxModulesSchema>
; MLIR_OUT:  }

; MLIR_OUT:  om.class @MemorySchema(%basepath: !om.basepath, %name_in: !om.string, %depth_in: !om.integer, %width_in: !om.integer, %maskBits_in: !om.integer, %readPorts_in: !om.integer, %writePorts_in: !om.integer, %readwritePorts_in: !om.integer, %writeLatency_in: !om.integer, %readLatency_in: !om.integer, %hierarchy_in: !om.list<!om.path>, %inDut_in: i1, %extraPorts_in: !om.list<!om.class.type<@ExtraPortsMemorySchema>>, %preExtInstName_in: !om.list<!om.string>) -> (name: !om.string, depth: !om.integer, width: !om.integer, maskBits: !om.integer, readPorts: !om.integer, writePorts: !om.integer, readwritePorts: !om.integer, writeLatency: !om.integer, readLatency: !om.integer, hierarchy: !om.list<!om.path>, inDut: i1, extraPorts: !om.list<!om.class.type<@ExtraPortsMemorySchema>>, preExtInstName: !om.list<!om.string>)
; MLIR_OUT:    om.class.fields %name_in, %depth_in, %width_in, %maskBits_in, %readPorts_in, %writePorts_in, %readwritePorts_in, %writeLatency_in, %readLatency_in, %hierarchy_in, %inDut_in, %extraPorts_in, %preExtInstName_in : !om.string, !om.integer, !om.integer, !om.integer, !om.integer, !om.integer, !om.integer, !om.integer, !om.integer, !om.list<!om.path>, i1, !om.list<!om.class.type<@ExtraPortsMemorySchema>>, !om.list<!om.string>
; MLIR_OUT:  om.class @MemoryMetadata(%basepath: !om.basepath) -> (foo_m_ext_field: !om.class.type<@MemorySchema>, bar_m_ext_field: !om.class.type<@MemorySchema>, baz_m_ext_field: !om.class.type<@MemorySchema>)
; MLIR_OUT-DAG: om.path_create instance %basepath @memNLA
; MLIR_OUT-DAG: om.list_create
; MLIR_OUT-DAG: [[OBJ1:%.+]] = om.object @MemorySchema
; MLIR_OUT-DAG: om.constant "foo_m_ext" : !om.string
; MLIR_OUT-DAG: om.constant #om.integer<1 : ui64> : !om.integer
; MLIR_OUT-DAG: om.constant #om.integer<8 : ui32> : !om.integer
; MLIR_OUT-DAG: om.constant #om.integer<1 : ui32> : !om.integer
; MLIR_OUT-DAG: om.constant #om.integer<0 : ui32> : !om.integer
; MLIR_OUT-DAG: om.path_create instance %basepath @memNLA_0
; MLIR_OUT-DAG: om.list_create
; MLIR_OUT-DAG: [[OBJ2:%.+]] = om.object @MemorySchema
; MLIR_OUT-DAG: om.constant "bar_m_ext" : !om.string
; MLIR_OUT-DAG: om.constant #om.integer<2 : ui64> : !om.integer
; MLIR_OUT-DAG: om.path_create instance %basepath @memNLA_1
; MLIR_OUT-DAG: om.path_create instance %basepath @memNLA_2
; MLIR_OUT-DAG: om.list_create
; MLIR_OUT-DAG: [[OBJ3:%.+]] = om.object @MemorySchema
; MLIR_OUT-DAG: om.constant "baz_m_ext" : !om.string
; MLIR_OUT-DAG: om.constant #om.integer<3 : ui64> : !om.integer
; MLIR_OUT:     om.class.fields [[OBJ1]], [[OBJ2]], [[OBJ3]] : !om.class.type<@MemorySchema>, !om.class.type<@MemorySchema>, !om.class.type<@MemorySchema>

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
