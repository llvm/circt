// RUN: circt-opt --firrtl-emit-metadata="repl-seq-mem=true repl-seq-mem-file=mems.conf" -split-input-file %s | FileCheck %s

//===----------------------------------------------------------------------===//
// RetimeModules
//===----------------------------------------------------------------------===//

firrtl.circuit "retime0" attributes {
  annotations = [
    {
      class = "sifive.enterprise.firrtl.RetimeModulesAnnotation",
      filename = "retime_modules.json"
    }
  ]
} {
  firrtl.module @retime0() attributes {
    annotations = [
      {
        class = "freechips.rocketchip.util.RetimeModuleAnnotation"
      }
    ]
  } {}

  firrtl.module @retime1() { }

  firrtl.module @retime2() attributes {
    annotations = [
      {
        class = "freechips.rocketchip.util.RetimeModuleAnnotation"
      }
    ]
  } {}
}
// CHECK-LABEL: firrtl.circuit "retime0"   {
// CHECK:         firrtl.module @retime0(out %metadataObj: !firrtl.any
// CHECK:           [[SIFIVE_METADATA:%.+]] = firrtl.object @SiFive_Metadata
// CHECK:           [[METADATA_OBJ:%.+]] = firrtl.object.anyref_cast [[SIFIVE_METADATA]]
// CHECK:           propassign %metadataObj, [[METADATA_OBJ]]
// CHECK:         firrtl.module @retime1() {
// CHECK:         firrtl.module @retime2()

// CHECK:    firrtl.class @RetimeModulesSchema(in %[[moduleName_in:.+]]: !firrtl.string, out %moduleName: !firrtl.string) {
// CHECK:    firrtl.propassign %moduleName, %[[moduleName_in]]

// CHECK:  firrtl.class @RetimeModulesMetadata
// CHECK-SAME: (out %[[retime0_field:[a-zA-Z][a-zA-Z0-9_]*]]: !firrtl.class<@RetimeModulesSchema
// CHECK:    %[[v0:.+]] = firrtl.string "retime0"
// CHECK:    %retime0 = firrtl.object @RetimeModulesSchema
// CHECK-NEXT:    %[[v1:.+]] = firrtl.object.subfield %retime0
// CHECK-NEXT:    firrtl.propassign %[[v1]], %[[v0]] : !firrtl.string
// CHECK-NEXT:    firrtl.propassign %[[retime0_field]], %retime0
// CHECK:    %2 = firrtl.string "retime2"
// CHECK:    %retime2 = firrtl.object @RetimeModulesSchema
// CHECK:    firrtl.object.subfield %retime2
// CHECK:    firrtl.propassign %[[retime2field:.+]], %retime2

// CHECK:               emit.file "retime_modules.json" {
// CHECK-NEXT{LITERAL}:   sv.verbatim "[\0A \22{{0}}\22,\0A \22{{1}}\22\0A]" {symbols = [@retime0, @retime2]}
// CHECK-NEXT:          }

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
  firrtl.module @DUTBlackboxes() attributes {annotations = [
      {class = "sifive.enterprise.firrtl.MarkDUTAnnotation"}]} {
  }
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
  firrtl.module @TestBlackboxes() attributes {annotations = [
      {class = "sifive.enterprise.firrtl.MarkDUTAnnotation"}]} {
  }
// CHECK-NOT:  emit.file ""
// CHECK:      emit.file "test_blackboxes.json" {
// CHECK-NEXT:   emit.verbatim "[]"
// CHECK-NEXT: }
// CHECK-NOT:  emit.file ""
}

// -----

// CHECK-LABEL: firrtl.circuit "BasicBlackboxes"   {
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

// CHECK:    firrtl.class @SitestBlackBoxModulesSchema(in %[[moduleName_in:.+]]: !firrtl.string, out %moduleName: !firrtl.string) {
// CHECK:      firrtl.propassign %moduleName, %[[moduleName_in]]
// CHECK:    }

// CHECK:    firrtl.class @SitestBlackBoxMetadata(out %TestBlackbox_field: !firrtl.class<@SitestBlackBoxModulesSchema(in moduleName_in: !firrtl.string, out moduleName: !firrtl.string)>, out %DUTBlackbox_0_field: !firrtl.class<@SitestBlackBoxModulesSchema(in moduleName_in: !firrtl.string, out moduleName: !firrtl.string)>, out %DUTBlackbox_1_field: !firrtl.class<@SitestBlackBoxModulesSchema(in moduleName_in: !firrtl.string, out moduleName: !firrtl.string)>) attributes {portAnnotations = []} {
// CHECK:      firrtl.string "TestBlackbox"
// CHECK:      %TestBlackbox = firrtl.object @SitestBlackBoxModulesSchema
// CHECK:      firrtl.propassign %TestBlackbox_field, %TestBlackbox
// CHECK:      firrtl.string "DUTBlackbox2"
// CHECK:      %DUTBlackbox_0 = firrtl.object @SitestBlackBoxModulesSchema
// CHECK:      firrtl.propassign %DUTBlackbox_0_field, %DUTBlackbox_0
// CHECK:      firrtl.string "DUTBlackbox1"
// CHECK:      %DUTBlackbox_1 = firrtl.object @SitestBlackBoxModulesSchema
// CHECK:      firrtl.propassign %DUTBlackbox_1_field, %DUTBlackbox_1
// CHECK:    }
  // Gracefully handle missing defnames.
  firrtl.extmodule @NoDefName()

  firrtl.extmodule @TestBlackbox() attributes {defname = "TestBlackbox"}

  // CHECK:               emit.file "test_blackboxes.json" {
  // CHECK-NEXT{LITERAL}:   emit.verbatim "[\0A \22TestBlackbox\22\0A]"
  // CHECK-NEXT:          }

  // Should be de-duplicated and sorted.
  firrtl.extmodule @DUTBlackbox_0() attributes {defname = "DUTBlackbox2"}
  firrtl.extmodule @DUTBlackbox_1() attributes {defname = "DUTBlackbox1"}
  firrtl.extmodule @DUTBlackbox_2() attributes {defname = "DUTBlackbox1"}
  // CHECK:               emit.file "dut_blackboxes.json" {
  // CHECK-NEXT{LITERAL}:   emit.verbatim "[\0A \22DUTBlackbox1\22,\0A \22DUTBlackbox2\22\0A]"
  // CHECK-NEXT:          }
}

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
// CHECK:                 hw.hierpath @[[dutPathSym:.+]] [@Foo::@bar]

// CHECK-LABEL:         firrtl.module @Foo(
// CHECK-NEXT:            firrtl.instance bar
// CHECK-SAME:              id = distinct[[[dutId:[0-9]+]]]<>

// CHECK-LABEL:         firrtl.class @SiFive_Metadata(
// CHECK-SAME:            out %[[dutModulePath:dutModulePath.*]]: !firrtl.list<path>
// CHECK:                 %[[a:.+]] = firrtl.path instance distinct[[[dutId]]]<>
// CHECK-NEXT:            %[[b:.+]] = firrtl.list.create %[[a]] : !firrtl.list<path>
// CHECK-NEXT:            firrtl.propassign %[[dutModulePath]], %[[b]] : !firrtl.list<path>

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
// CHECK:                 hw.hierpath @[[memPathSym:.+]] [@Bar::@baz, @Baz::@m]

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
// CHECK-NEXT:            %29 = firrtl.object.subfield %[[memoryObject]][hierarchy_in]
// CHECK-NEXT:            firrtl.propassign %29, %3
// CHECK-NEXT:            %30 = firrtl.bool true
// CHECK-NEXT:            %31 = firrtl.object.subfield %[[memoryObject]][inDut_in]
// CHECK-NEXT:            firrtl.propassign %31, %30
// CHECK-NEXT:            %32 = firrtl.object.subfield %[[memoryObject]][extraPorts_in]
// CHECK-NEXT:            firrtl.propassign %32, %10
// CHECK-NEXT:            %33 = firrtl.object.subfield %[[memoryObject]][preExtInstName_in]
// CHECK-NEXT:            firrtl.propassign %33, %2
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

// CHECK-LABEL: firrtl.circuit "DualReadsSMem"
firrtl.circuit "DualReadsSMem" {
  firrtl.module @DualReadsSMem() {
    %0:12 = firrtl.instance DualReads_ext @DualReads_ext(
      in R0_addr: !firrtl.uint<4>,
      in R0_en: !firrtl.uint<1>,
      in R0_clk: !firrtl.clock,
      in R0_data: !firrtl.uint<42>,
      in R1_addr: !firrtl.uint<4>,
      in R1_en: !firrtl.uint<1>,
      in R1_clk: !firrtl.clock,
      in R1_data: !firrtl.uint<42>,
      in W0_addr: !firrtl.uint<4>,
      in W0_en: !firrtl.uint<1>,
      in W0_clk: !firrtl.clock,
      in W0_data: !firrtl.uint<42>
    )
  }
  firrtl.memmodule @DualReads_ext(
    in R0_addr: !firrtl.uint<4>,
    in R0_en: !firrtl.uint<1>,
    in R0_clk: !firrtl.clock,
    in R0_data: !firrtl.uint<42>,
    in R1_addr: !firrtl.uint<4>,
    in R1_en: !firrtl.uint<1>,
    in R1_clk: !firrtl.clock,
    in R1_data: !firrtl.uint<42>,
    in W0_addr: !firrtl.uint<4>,
    in W0_en: !firrtl.uint<1>,
    in W0_clk: !firrtl.clock,
    in W0_data: !firrtl.uint<42>
  ) attributes {
    dataWidth = 42 : ui32,
    depth = 12 : ui64,
    extraPorts = [],
    maskBits = 1 : ui32,
    numReadPorts = 2 : ui32,
    numReadWritePorts = 0 : ui32,
    numWritePorts = 1 : ui32,
    readLatency = 1 : ui32,
    writeLatency = 1 : ui32
  }
  // CHECK{LITERAL}: sv.verbatim "[\0A {\0A \22module_name\22: \22{{0}}\22,\0A \22depth\22: 12,\0A \22width\22: 42,\0A \22masked\22: false,\0A \22read\22: 2,\0A \22write\22: 1,\0A \22readwrite\22: 0,\0A \22extra_ports\22: [],\0A \22hierarchy\22: [\0A \22{{1}}.DualReads_ext\22\0A ]\0A }\0A]"
  // CHECK: {symbols = [@DualReads_ext, @DualReadsSMem]}
  // CHECK{LITERAL}: sv.verbatim "name {{0}} depth 12 width 42 ports write,read,read\0A" {symbols = [@DualReads_ext]}
}

// -----

// CHECK-LABEL: firrtl.circuit "ReadOnlyMemory"
firrtl.circuit "ReadOnlyMemory" {
  firrtl.module @ReadOnlyMemory() {
    %0:4 = firrtl.instance rom_ext sym @rom_ext_0 @rom_ext(
      in R0_addr: !firrtl.uint<9>,
      in R0_en: !firrtl.uint<1>,
      in R0_clk: !firrtl.clock,
      out R0_data: !firrtl.uint<32>
    )
  }
  firrtl.memmodule @rom_ext(
    in R0_addr: !firrtl.uint<9>,
    in R0_en: !firrtl.uint<1>,
    in R0_clk: !firrtl.clock,
    out R0_data: !firrtl.uint<32>
  ) attributes {
    dataWidth = 32 : ui32,
    depth = 512 : ui64,
    extraPorts = [],
    maskBits = 0 : ui32,
    numReadPorts = 1 : ui32,
    numReadWritePorts = 0 : ui32,
    numWritePorts = 0 : ui32,
    readLatency = 1 : ui32,
    writeLatency = 1 : ui32
  }
  // CHECK{LITERAL}: sv.verbatim "[\0A  {\0A \22module_name\22: \22{{0}}\22,\0A \22depth\22: 512,\0A \22width\22: 32,\0A \22masked\22: false,\0A \22read\22: 1,\0A \22write\22: 0,\0A \22readwrite\22: 0,\0A \22extra_ports\22: [],\0A \22hierarchy\22: [\0A \22{{1}}.rom_ext\22\0A ]\0A }\0A]"
  // CHECK: symbols = [@rom_ext, @ReadOnlyMemory]}
  // CHECK{LITERAL}: sv.verbatim "name {{0}} depth 512 width 32 ports read\0A" {symbols = [@rom_ext]}
}

// -----

// CHECK-LABEL: firrtl.circuit "top"
firrtl.circuit "top" {
    // CHECK: hw.hierpath @[[DUTNLA:.+]] [@top::@sym]
    // CHECK-LABEL: firrtl.module @top
    firrtl.module @top()  {
      // CHECK: firrtl.instance dut sym @[[DUT_SYM:.+]] {annotations = [{circt.nonlocal = @dutNLA, class = "circt.tracker", id = distinct[0]<>}]} @DUT()
      firrtl.instance dut @DUT()
      firrtl.instance mem1 @Mem1()
      firrtl.instance mem2 @Mem2()
    }
    firrtl.module private @Mem1() {
      %0:4 = firrtl.instance head_ext @head_ext(
        in W0_addr: !firrtl.uint<5>,
        in W0_en: !firrtl.uint<1>,
        in W0_clk: !firrtl.clock,
        in W0_data: !firrtl.uint<5>
      )
    }
    firrtl.module private @Mem2() {
      %0:4 =  firrtl.instance head_0_ext @head_0_ext(
        in W0_addr: !firrtl.uint<5>,
        in W0_en: !firrtl.uint<1>,
        in W0_clk: !firrtl.clock,
        in W0_data: !firrtl.uint<5>
      )
    }
    // CHECK-LABEL: firrtl.module private @DUT(
    firrtl.module private @DUT() attributes {annotations = [
      {class = "sifive.enterprise.firrtl.MarkDUTAnnotation"}]} {
      // CHECK: firrtl.instance mem1 sym @[[MEM1_SYM:.+]] {annotations = [{
      // CHECK-SAME: circt.nonlocal = @memNLA, class = "circt.tracker", id = distinct
      firrtl.instance mem1 @Mem()
    }
    firrtl.module private @Mem() {
      %0:10 = firrtl.instance memory_ext @memory_ext(
        in R0_addr: !firrtl.uint<4>,
        in R0_en: !firrtl.uint<1>,
        in R0_clk: !firrtl.clock,
        out R0_data: !firrtl.uint<8>,
        in RW0_addr: !firrtl.uint<4>,
        in RW0_en: !firrtl.uint<1>,
        in RW0_clk: !firrtl.clock,
        in RW0_wmode: !firrtl.uint<1>,
        in RW0_wdata: !firrtl.uint<8>,
        out RW0_rdata: !firrtl.uint<8>
      )
      %1:8 = firrtl.instance dumm_ext @dumm_ext(
        in R0_addr: !firrtl.uint<5>,
        in R0_en: !firrtl.uint<1>,
        in R0_clk: !firrtl.clock,
        out R0_data: !firrtl.uint<5>,
        in W0_addr: !firrtl.uint<5>,
        in W0_en: !firrtl.uint<1>,
        in W0_clk: !firrtl.clock,
        in W0_data: !firrtl.uint<5>
      )
    }
    firrtl.memmodule private @head_ext(
      in W0_addr: !firrtl.uint<5>,
      in W0_en: !firrtl.uint<1>,
      in W0_clk: !firrtl.clock,
      in W0_data: !firrtl.uint<5>) attributes {dataWidth = 5 : ui32,
      depth = 20 : ui64,
      extraPorts = [],
      maskBits = 1 : ui32,
      numReadPorts = 0 : ui32,
      numReadWritePorts = 0 : ui32,
      numWritePorts = 1 : ui32,
      readLatency = 1 : ui32,
      writeLatency = 1 : ui32
    }
    firrtl.memmodule private @head_0_ext(
      in W0_addr: !firrtl.uint<5>,
      in W0_en: !firrtl.uint<1>,
      in W0_clk: !firrtl.clock,
      in W0_data: !firrtl.uint<5>) attributes {dataWidth = 5 : ui32,
      depth = 20 : ui64,
      extraPorts = [],
      maskBits = 1 : ui32,
      numReadPorts = 0 : ui32,
      numReadWritePorts = 0 : ui32,
      numWritePorts = 1 : ui32,
      readLatency = 1 : ui32,
      writeLatency = 1 : ui32
    }
    firrtl.memmodule private @memory_ext(
      in R0_addr: !firrtl.uint<4>,
      in R0_en: !firrtl.uint<1>,
      in R0_clk: !firrtl.clock,
      out R0_data: !firrtl.uint<8>,
      in RW0_addr: !firrtl.uint<4>,
      in RW0_en: !firrtl.uint<1>,
      in RW0_clk: !firrtl.clock,
      in RW0_wmode: !firrtl.uint<1>,
      in RW0_wdata: !firrtl.uint<8>,
      out RW0_rdata: !firrtl.uint<8>) attributes {dataWidth = 8 : ui32,
      depth = 16 : ui64,
      extraPorts = [],
      maskBits = 1 : ui32,
      numReadPorts = 1 : ui32,
      numReadWritePorts = 1 : ui32,
      numWritePorts = 0 : ui32,
      readLatency = 1 : ui32,
      writeLatency = 1 : ui32
    }
    firrtl.memmodule private @dumm_ext(
      in R0_addr: !firrtl.uint<5>,
      in R0_en: !firrtl.uint<1>,
      in R0_clk: !firrtl.clock,
      out R0_data: !firrtl.uint<5>,
      in W0_addr: !firrtl.uint<5>,
      in W0_en: !firrtl.uint<1>,
      in W0_clk: !firrtl.clock,
      in W0_data: !firrtl.uint<5>) attributes {dataWidth = 5 : ui32,
      depth = 20 : ui64,
      extraPorts = [],
      maskBits = 1 : ui32,
      numReadPorts = 1 : ui32,
      numReadWritePorts = 0 : ui32,
      numWritePorts = 1 : ui32,
      readLatency = 1 : ui32,
      writeLatency = 1 : ui32
    }

  // CHECK-LABEL:  firrtl.class @MemoryMetadata
  // CHECK:  %[[V2:.+]] = firrtl.string "memory_ext"
  // CHECK:  %[[V3:.+]] = firrtl.path instance distinct[1]<>
  // CHECK:  firrtl.list.create %[[V2]] : !firrtl.list<string>
  // CHECK:  firrtl.list.create %[[V3]] : !firrtl.list<path>

  // CHECK:               emit.file "metadata{{/|\\\\}}seq_mems.json" {
  // CHECK-NEXT{LITERAL}:   sv.verbatim "[\0A {\0A \22module_name\22: \22{{0}}\22,\0A \22depth\22: 16,\0A \22width\22: 8,\0A \22masked\22: false,\0A \22read\22: 1,\0A \22write\22: 0,\0A \22readwrite\22: 1,\0A \22extra_ports\22: [],\0A \22hierarchy\22: [\0A \22{{3}}.{{4}}.memory_ext\22\0A ]\0A },\0A {\0A \22module_name\22: \22{{5}}\22,\0A \22depth\22: 20,\0A \22width\22: 5,\0A \22masked\22: false,\0A \22read\22: 1,\0A \22write\22: 1,\0A \22readwrite\22: 0,\0A \22extra_ports\22: [],\0A \22hierarchy\22: [\0A \22{{3}}.{{4}}.dumm_ext\22\0A ]\0A }\0A]"
  // CHECK-SAME:              {symbols = [@memory_ext, @top, #hw.innerNameRef<@top::@[[DUT_SYM]]>, @DUT, #hw.innerNameRef<@DUT::@[[MEM1_SYM]]>, @dumm_ext]}
  // CHECK-NEXT:          }

  // CHECK:               emit.file "mems.conf" {
  // CHECK-NEXT{LITERAL}:   sv.verbatim "name {{0}} depth 20 width 5 ports write\0Aname {{1}} depth 20 width 5 ports write\0Aname {{2}} depth 16 width 8 ports read,rw\0Aname {{3}} depth 20 width 5 ports write,read\0A"
  // CHECK-SAME:              {symbols = [@head_ext, @head_0_ext, @memory_ext, @dumm_ext]}
  // CHECK-NEXT:          }

  // CHECK-LABEL:  firrtl.class @SiFive_Metadata
  // CHECK:    %[[V0:.+]] = firrtl.path instance distinct[0]<>
  // CHECK-NEXT:    %[[V1:.+]] = firrtl.list.create %[[V0]] : !firrtl.list<path>
  // CHECK-NEXT:    firrtl.propassign %dutModulePath_field_1, %[[V1]] : !firrtl.list<path>
}
