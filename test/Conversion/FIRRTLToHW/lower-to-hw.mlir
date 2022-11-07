// RUN: circt-opt -lower-firrtl-to-hw -verify-diagnostics %s | FileCheck %s
// RUN: circt-opt -pass-pipeline="lower-firrtl-to-hw{disable-mem-randomization}" -verify-diagnostics %s | FileCheck %s --check-prefix DISABLE_RANDOM --implicit-check-not RANDOMIZE_MEM
// RUN: circt-opt -pass-pipeline="lower-firrtl-to-hw{disable-reg-randomization}" -verify-diagnostics %s | FileCheck %s --check-prefix DISABLE_RANDOM --implicit-check-not RANDOMIZE_REG
// RUN: circt-opt -pass-pipeline="lower-firrtl-to-hw{disable-mem-randomization disable-reg-randomization}" -verify-diagnostics %s | FileCheck %s --check-prefix DISABLE_RANDOM --implicit-check-not RANDOMIZE_MEM --implicit-check-not RANDOMIZE_REG

// DISABLE_RANDOM-LABEL: module @Simple
firrtl.circuit "Simple"   attributes {annotations = [{class =
"sifive.enterprise.firrtl.ExtractAssumptionsAnnotation", directory = "dir1",  filename = "./dir1/filename1" }, {class =
"sifive.enterprise.firrtl.ExtractCoverageAnnotation", directory = "dir2",  filename = "./dir2/filename2" }, {class =
"sifive.enterprise.firrtl.ExtractAssertionsAnnotation", directory = "dir3",  filename = "./dir3/filename3" }]}
{

  //These come from MemSimple, IncompleteRead, and MemDepth1
  // CHECK-LABEL: hw.generator.schema @FIRRTLMem, "FIRRTL_Memory", ["depth", "numReadPorts", "numWritePorts", "numReadWritePorts", "readLatency", "writeLatency", "width", "maskGran", "readUnderWrite", "writeUnderWrite", "writeClockIDs"]
  // CHECK: hw.module.generated @aa_combMem, @FIRRTLMem(%W0_addr: i4, %W0_en: i1, %W0_clk: i1, %W0_data: i8, %W1_addr: i4, %W1_en: i1, %W1_clk: i1, %W1_data: i8) attributes {depth = 16 : i64, maskGran = 8 : ui32, numReadPorts = 0 : ui32, numReadWritePorts = 0 : ui32, numWritePorts = 2 : ui32, readLatency = 1 : ui32, readUnderWrite = 0 : ui32, width = 8 : ui32, writeClockIDs = [0 : i32, 0 : i32], writeLatency = 1 : ui32, writeUnderWrite = 1 : i32}
  // CHECK: hw.module.generated @ab_combMem, @FIRRTLMem(%W0_addr: i4, %W0_en: i1, %W0_clk: i1, %W0_data: i8, %W1_addr: i4, %W1_en: i1, %W1_clk: i1, %W1_data: i8) attributes {depth = 16 : i64, maskGran = 8 : ui32, numReadPorts = 0 : ui32, numReadWritePorts = 0 : ui32, numWritePorts = 2 : ui32, readLatency = 1 : ui32, readUnderWrite = 0 : ui32, width = 8 : ui32, writeClockIDs = [0 : i32, 1 : i32], writeLatency = 1 : ui32, writeUnderWrite = 1 : i32}
  // CHECK: hw.module.generated @mem0_combMem, @FIRRTLMem(%R0_addr: i1, %R0_en: i1, %R0_clk: i1) -> (R0_data: i32) attributes {depth = 1 : i64, maskGran = 32 : ui32, numReadPorts = 1 : ui32, numReadWritePorts = 0 : ui32, numWritePorts = 0 : ui32, readLatency = 0 : ui32, readUnderWrite = 1 : ui32, width = 32 : ui32, writeClockIDs = [], writeLatency = 1 : ui32, writeUnderWrite = 1 : i32}
  // CHECK: hw.module.generated @_M_combMem, @FIRRTLMem(%R0_addr: i4, %R0_en: i1, %R0_clk: i1) -> (R0_data: i42) attributes {depth = 12 : i64, maskGran = 42 : ui32, numReadPorts = 1 : ui32, numReadWritePorts = 0 : ui32, numWritePorts = 0 : ui32, readLatency = 0 : ui32, readUnderWrite = 0 : ui32, width = 42 : ui32, writeClockIDs = [], writeLatency = 1 : ui32, writeUnderWrite = 1 : i32}
  // CHECK: hw.module.generated @tbMemoryKind1_combMem, @FIRRTLMem(%R0_addr: i4, %R0_en: i1, %R0_clk: i1, %W0_addr: i4, %W0_en: i1, %W0_clk: i1, %W0_data: i8) -> (R0_data: i8) attributes {depth = 16 : i64, maskGran = 8 : ui32, numReadPorts = 1 : ui32, numReadWritePorts = 0 : ui32, numWritePorts = 1 : ui32, readLatency = 1 : ui32, readUnderWrite = 0 : ui32, width = 8 : ui32, writeClockIDs = [0 : i32], writeLatency = 1 : ui32, writeUnderWrite = 1 : i32}
  // CHECK: hw.module.generated @_M_mask_combMem, @FIRRTLMem(%R0_addr: i10, %R0_en: i1, %R0_clk: i1, %RW0_addr: i10, %RW0_en: i1, %RW0_clk: i1, %RW0_wmode: i1, %RW0_wdata: i40, %RW0_wmask: i4, %W0_addr: i10, %W0_en: i1, %W0_clk: i1, %W0_data: i40, %W0_mask: i4) -> (R0_data: i40, RW0_rdata: i40) attributes {depth = 1022 : i64, maskGran = 10 : ui32, numReadPorts = 1 : ui32, numReadWritePorts = 1 : ui32, numWritePorts = 1 : ui32, readLatency = 1 : ui32, readUnderWrite = 0 : ui32, width = 40 : ui32, writeClockIDs = [0 : i32], writeLatency = 1 : ui32, writeUnderWrite = 1 : i32}
  // CHECK: hw.module.generated @_M_combMem_0, @FIRRTLMem(%R0_addr: i4, %R0_en: i1, %R0_clk: i1, %RW0_addr: i4, %RW0_en: i1, %RW0_clk: i1, %RW0_wmode: i1, %RW0_wdata: i42, %W0_addr: i4, %W0_en: i1, %W0_clk: i1, %W0_data: i42) -> (R0_data: i42, RW0_rdata: i42) attributes {depth = 12 : i64, maskGran = 42 : ui32, numReadPorts = 1 : ui32, numReadWritePorts = 1 : ui32, numWritePorts = 1 : ui32, readLatency = 0 : ui32, readUnderWrite = 0 : ui32, width = 42 : ui32, writeClockIDs = [0 : i32], writeLatency = 1 : ui32, writeUnderWrite = 1 : i32}

  // CHECK-LABEL: hw.module @Simple
  firrtl.module @Simple(in %in1: !firrtl.uint<4>,
                        in %in2: !firrtl.uint<2>,
                        in %in3: !firrtl.sint<8>,
                        in %in4: !firrtl.uint<0>,
                        in %in5: !firrtl.sint<0>,
                        out %out1: !firrtl.sint<1>,
                        out %out2: !firrtl.sint<1>  ) {
    // Issue #364: https://github.com/llvm/circt/issues/364
    // CHECK: = hw.constant -1175 : i12
    // CHECK-DAG: hw.constant -4 : i4
    %c12_ui4 = firrtl.constant 12 : !firrtl.uint<4>

    // CHECK-DAG: hw.constant 2 : i3
    %c2_si3 = firrtl.constant 2 : !firrtl.sint<3>


    // CHECK: %out4 = sv.wire sym @__Simple__out4 : !hw.inout<i4>
    // CHECK: %out5 = sv.wire sym @__Simple__out5 : !hw.inout<i4>
    %out4 = firrtl.wire {annotations = [{class = "firrtl.transforms.DontTouchAnnotation"}]} : !firrtl.uint<4>
    %out5 = firrtl.wire sym @__Simple__out5 : !firrtl.uint<4>
    // CHECK: sv.wire sym @__Simple{{.*}}
    // CHECK: sv.wire sym @__Simple{{.*}}
    %500 = firrtl.wire {annotations = [{class = "firrtl.transforms.DontTouchAnnotation"}]} : !firrtl.uint<4>
    %501 = firrtl.wire {annotations = [{class = "firrtl.transforms.DontTouchAnnotation"}]} : !firrtl.uint<5>

    // CHECK: sv.wire sym @__Simple__dntnode
    %dntnode = firrtl.node %in1 {annotations = [{class = "firrtl.transforms.DontTouchAnnotation"}]} : !firrtl.uint<4>

    // CHECK: %clockWire = sv.wire
    // CHECK: sv.assign %clockWire, %false : i1
    %c0_clock = firrtl.specialconstant 0 : !firrtl.clock
    %clockWire = firrtl.wire : !firrtl.clock
    firrtl.connect %clockWire, %c0_clock : !firrtl.clock, !firrtl.clock

    // CHECK: sv.assign %out5, %c0_i4 : i4
    %tmp1 = firrtl.invalidvalue : !firrtl.uint<4>
    firrtl.connect %out5, %tmp1 : !firrtl.uint<4>, !firrtl.uint<4>

    // CHECK: [[ZEXT:%.+]] = comb.concat %false, %in1 : i1, i4
    // CHECK: [[ADD:%.+]] = comb.add bin %c12_i5, [[ZEXT]] : i5
    %0 = firrtl.add %c12_ui4, %in1 : (!firrtl.uint<4>, !firrtl.uint<4>) -> !firrtl.uint<5>

    %1 = firrtl.asUInt %in1 : (!firrtl.uint<4>) -> !firrtl.uint<4>

    // CHECK: [[ZEXT1:%.+]] = comb.concat %false, [[ADD]] : i1, i5
    // CHECK: [[ZEXT2:%.+]] = comb.concat %c0_i2, %in1 : i2, i4
    // CHECK-NEXT: [[SUB:%.+]] = comb.sub bin [[ZEXT1]], [[ZEXT2]] : i6
    %2 = firrtl.sub %0, %1 : (!firrtl.uint<5>, !firrtl.uint<4>) -> !firrtl.uint<6>

    %in2s = firrtl.asSInt %in2 : (!firrtl.uint<2>) -> !firrtl.sint<2>

    // CHECK: [[PADRES_SIGN:%.+]] = comb.extract %in2 from 1 : (i2) -> i1
    // CHECK: [[PADRES:%.+]] = comb.concat [[PADRES_SIGN]], %in2 : i1, i2
    %3 = firrtl.pad %in2s, 3 : (!firrtl.sint<2>) -> !firrtl.sint<3>

    // CHECK: [[PADRES2:%.+]] = comb.concat %c0_i2, %in2 : i2, i2
    %4 = firrtl.pad %in2, 4 : (!firrtl.uint<2>) -> !firrtl.uint<4>

    // CHECK: [[IN2EXT:%.+]] = comb.concat %c0_i2, %in2 : i2, i2
    // CHECK: [[XOR:%.+]] = comb.xor bin [[IN2EXT]], [[PADRES2]] : i4
    %5 = firrtl.xor %in2, %4 : (!firrtl.uint<2>, !firrtl.uint<4>) -> !firrtl.uint<4>

    // CHECK: comb.and bin [[XOR]]
    %and = firrtl.and %5, %4 : (!firrtl.uint<4>, !firrtl.uint<4>) -> !firrtl.uint<4>

    // CHECK: comb.or bin [[XOR]]
    %or = firrtl.or %5, %4 : (!firrtl.uint<4>, !firrtl.uint<4>) -> !firrtl.uint<4>

    // CHECK: [[CONCAT1:%.+]] = comb.concat [[PADRES2]], [[XOR]] : i4, i4
    %6 = firrtl.cat %4, %5 : (!firrtl.uint<4>, !firrtl.uint<4>) -> !firrtl.uint<8>

    // CHECK: comb.concat %in1, %in2
    %7 = firrtl.cat %in1, %in2 : (!firrtl.uint<4>, !firrtl.uint<2>) -> !firrtl.uint<6>

    // CHECK-NEXT: sv.assign %out5, [[PADRES2]] : i4
    firrtl.connect %out5, %4 : !firrtl.uint<4>, !firrtl.uint<4>

    // CHECK-NEXT: sv.assign %out4, [[XOR]] : i4
    firrtl.connect %out4, %5 : !firrtl.uint<4>, !firrtl.uint<4>

    // CHECK-NEXT: [[ZEXT:%.+]] = comb.concat %c0_i2, %in2 : i2, i2
    // CHECK-NEXT: sv.assign %out4, [[ZEXT]] : i4
    firrtl.connect %out4, %in2 : !firrtl.uint<4>, !firrtl.uint<2>

    // CHECK-NEXT: %test-name = sv.wire sym @"__Simple__test-name" : !hw.inout<i4>
    firrtl.wire {name = "test-name", annotations = [{class = "firrtl.transforms.DontTouchAnnotation"}]} : !firrtl.uint<4>

    // CHECK-NEXT: = sv.wire : !hw.inout<i2>
    %_t_1 = firrtl.wire droppable_name : !firrtl.uint<2>

    // CHECK-NEXT: = sv.wire  : !hw.inout<array<13xi1>>
    %_t_2 = firrtl.wire droppable_name : !firrtl.vector<uint<1>, 13>

    // CHECK-NEXT: = sv.wire  : !hw.inout<array<13xi2>>
    %_t_3 = firrtl.wire droppable_name : !firrtl.vector<uint<2>, 13>

    // CHECK-NEXT: = comb.extract [[CONCAT1]] from 3 : (i8) -> i5
    %8 = firrtl.bits %6 7 to 3 : (!firrtl.uint<8>) -> !firrtl.uint<5>

    // CHECK-NEXT: = comb.extract [[CONCAT1]] from 5 : (i8) -> i3
    %9 = firrtl.head %6, 3 : (!firrtl.uint<8>) -> !firrtl.uint<3>

    // CHECK-NEXT: = comb.extract [[CONCAT1]] from 0 : (i8) -> i5
    %10 = firrtl.tail %6, 3 : (!firrtl.uint<8>) -> !firrtl.uint<5>

    // CHECK-NEXT: = comb.extract [[CONCAT1]] from 3 : (i8) -> i5
    %11 = firrtl.shr %6, 3 : (!firrtl.uint<8>) -> !firrtl.uint<5>

    %12 = firrtl.shr %6, 8 : (!firrtl.uint<8>) -> !firrtl.uint<1>

    // CHECK-NEXT: = comb.extract %in3 from 7 : (i8) -> i1
    %13 = firrtl.shr %in3, 8 : (!firrtl.sint<8>) -> !firrtl.sint<1>

    // CHECK-NEXT: = comb.concat [[CONCAT1]], %c0_i3 : i8, i3
    %14 = firrtl.shl %6, 3 : (!firrtl.uint<8>) -> !firrtl.uint<11>

    // CHECK-NEXT: = comb.parity [[CONCAT1]] : i8
    %15 = firrtl.xorr %6 : (!firrtl.uint<8>) -> !firrtl.uint<1>

    // CHECK-NEXT: = comb.icmp bin eq  {{.*}}, %c-1_i8 : i8
    %16 = firrtl.andr %6 : (!firrtl.uint<8>) -> !firrtl.uint<1>

    // CHECK-NEXT: = comb.icmp bin ne {{.*}}, %c0_i8 : i8
    %17 = firrtl.orr %6 : (!firrtl.uint<8>) -> !firrtl.uint<1>

    // CHECK-NEXT: [[ZEXTC1:%.+]] = comb.concat %c0_i6, [[CONCAT1]] : i6, i8
    // CHECK-NEXT: [[ZEXT2:%.+]] = comb.concat %c0_i8, [[SUB]] : i8, i6
    // CHECK-NEXT: [[VAL18:%.+]] = comb.mul bin [[ZEXTC1]], [[ZEXT2]] : i14
    %18 = firrtl.mul %6, %2 : (!firrtl.uint<8>, !firrtl.uint<6>) -> !firrtl.uint<14>

    // CHECK: [[IN3SEXT:%.+]] = comb.concat {{.*}}, %in3 : i1, i8
    // CHECK: [[PADRESSEXT:%.+]] = comb.concat {{.*}}, [[PADRES]] : i6, i3
    // CHECK-NEXT: = comb.divs bin [[IN3SEXT]], [[PADRESSEXT]] : i9
    %19 = firrtl.div %in3, %3 : (!firrtl.sint<8>, !firrtl.sint<3>) -> !firrtl.sint<9>

    // CHECK: [[IN3EX:%.+]] = comb.concat {{.*}}, [[PADRES]] : i5, i3
    // CHECK-NEXT: [[MOD1:%.+]] = comb.mods bin %in3, [[IN3EX]] : i8
    // CHECK-NEXT: = comb.extract [[MOD1]] from 0 : (i8) -> i3
    %20 = firrtl.rem %in3, %3 : (!firrtl.sint<8>, !firrtl.sint<3>) -> !firrtl.sint<3>

    // CHECK: [[IN4EX:%.+]] = comb.concat {{.*}}, [[PADRES]] : i5, i3
    // CHECK-NEXT: [[MOD2:%.+]] = comb.mods bin [[IN4EX]], %in3 : i8
    // CHECK-NEXT: = comb.extract [[MOD2]] from 0 : (i8) -> i3
    %21 = firrtl.rem %3, %in3 : (!firrtl.sint<3>, !firrtl.sint<8>) -> !firrtl.sint<3>

    // Nodes with names become wires.
    // CHECK-NEXT: %n1 = sv.wire
    // CHECK-NEXT: sv.assign %n1, %in2
    // CHECK-NEXT: sv.read_inout %n1
    %n1 = firrtl.node interesting_name %in2 {name = "n1"} : !firrtl.uint<2>
    
    // CHECK-NEXT: [[WIRE:%n2]] = sv.wire sym @__Simple__n2 : !hw.inout<i2>
    // CHECK-NEXT: sv.assign [[WIRE]], %in2 : i2
    // CHECK-NEXT: sv.read_inout %n2
    %n2 = firrtl.node interesting_name %in2  {name = "n2", annotations = [{class = "firrtl.transforms.DontTouchAnnotation"}]} : !firrtl.uint<2>

    // Nodes with no names are just dropped.
    %22 = firrtl.node droppable_name %in2 {name = ""} : !firrtl.uint<2>

    // CHECK-NEXT: [[WIRE:%n3]] = sv.wire sym @nodeSym : !hw.inout<i2>
    // CHECK-NEXT: sv.assign [[WIRE]], %in2 : i2
    // CHECK-NEXT: sv.read_inout [[WIRE]]
    %n3 = firrtl.node sym @nodeSym %in2 : !firrtl.uint<2>

    // CHECK-NEXT: [[CVT:%.+]] = comb.concat %false, %in2 : i1, i2
    %23 = firrtl.cvt %22 : (!firrtl.uint<2>) -> !firrtl.sint<3>

    // Will be dropped, here because this triggered a crash
    %s23 = firrtl.cvt %in3 : (!firrtl.sint<8>) -> !firrtl.sint<8>

    // CHECK-NEXT: [[XOR:%.+]] = comb.xor bin [[CVT]], %c-1_i3 : i3
    %24 = firrtl.not %23 : (!firrtl.sint<3>) -> !firrtl.uint<3>

    %s24 = firrtl.asSInt %24 : (!firrtl.uint<3>) -> !firrtl.sint<3>

    // CHECK: [[SEXT:%.+]] = comb.concat {{.*}}, [[XOR]] : i1, i3
    // CHECK-NEXT: [[SUB:%.+]] = comb.sub bin %c0_i4, [[SEXT]] : i4
    %25 = firrtl.neg %s24 : (!firrtl.sint<3>) -> !firrtl.sint<4>

    // CHECK: [[CVT4:%.+]] = comb.concat {{.*}}, [[CVT]] : i1, i3
    // CHECK-NEXT: comb.mux bin {{.*}}, [[CVT4]], [[SUB]] : i4
    %26 = firrtl.mux(%17, %23, %25) : (!firrtl.uint<1>, !firrtl.sint<3>, !firrtl.sint<4>) -> !firrtl.sint<4>

    // CHECK-NEXT: = comb.icmp bin eq {{.*}}, %c-1_i14 : i14
    %28 = firrtl.andr %18 : (!firrtl.uint<14>) -> !firrtl.uint<1>

    // CHECK-NEXT: = comb.icmp bin ceq {{.*}}, %x_i1
    %x28 = firrtl.verif_isX %28 : !firrtl.uint<1>

    // CHECK-NEXT: [[XOREXT:%.+]] = comb.concat %c0_i11, [[XOR]]
    // CHECK-NEXT: [[SHIFT:%.+]] = comb.shru bin [[XOREXT]], [[VAL18]] : i14
    // CHECK-NEXT: [[DSHR:%.+]] = comb.extract [[SHIFT]] from 0 : (i14) -> i3
    %29 = firrtl.dshr %24, %18 : (!firrtl.uint<3>, !firrtl.uint<14>) -> !firrtl.uint<3>

    // CHECK-NEXT: = comb.concat %c0_i5, {{.*}} : i5, i3
    // CHECK-NEXT: [[SHIFT:%.+]] = comb.shrs bin %in3, {{.*}} : i8
    %a29 = firrtl.dshr %in3, %9 : (!firrtl.sint<8>, !firrtl.uint<3>) -> !firrtl.sint<8>

    // CHECK: = comb.concat {{.*}}, %in3 : i7, i8
    // CHECK-NEXT: = comb.concat %c0_i12, [[DSHR]]
    // CHECK-NEXT: [[SHIFT:%.+]] = comb.shl bin {{.*}}, {{.*}} : i15
    %30 = firrtl.dshl %in3, %29 : (!firrtl.sint<8>, !firrtl.uint<3>) -> !firrtl.sint<15>

    // CHECK-NEXT: = comb.shl bin [[DSHR]], [[DSHR]] : i3
    %dshlw = firrtl.dshlw %29, %29 : (!firrtl.uint<3>, !firrtl.uint<3>) -> !firrtl.uint<3>

    // Issue #367: https://github.com/llvm/circt/issues/367
    // CHECK: = comb.concat {{.*}} : i10, i4
    // CHECK-NEXT: [[SHIFT:%.+]] = comb.shrs bin {{.*}}, {{.*}} : i14
    // CHECK-NEXT: = comb.extract [[SHIFT]] from 0 : (i14) -> i4
    %31 = firrtl.dshr %25, %18 : (!firrtl.sint<4>, !firrtl.uint<14>) -> !firrtl.sint<4>

    // CHECK-NEXT: comb.icmp bin ule {{.*}}, {{.*}} : i4
    %41 = firrtl.leq %in1, %4 : (!firrtl.uint<4>, !firrtl.uint<4>) -> !firrtl.uint<1>
    // CHECK-NEXT: comb.icmp bin ult {{.*}}, {{.*}} : i4
    %42 = firrtl.lt %in1, %4 : (!firrtl.uint<4>, !firrtl.uint<4>) -> !firrtl.uint<1>
    // CHECK-NEXT: comb.icmp bin uge {{.*}}, {{.*}} : i4
    %43 = firrtl.geq %in1, %4 : (!firrtl.uint<4>, !firrtl.uint<4>) -> !firrtl.uint<1>
    // CHECK-NEXT: comb.icmp bin ugt {{.*}}, {{.*}} : i4
    %44 = firrtl.gt %in1, %4 : (!firrtl.uint<4>, !firrtl.uint<4>) -> !firrtl.uint<1>
    // CHECK-NEXT: comb.icmp bin eq {{.*}}, {{.*}} : i4
    %45 = firrtl.eq %in1, %4 : (!firrtl.uint<4>, !firrtl.uint<4>) -> !firrtl.uint<1>
    // CHECK-NEXT: comb.icmp bin ne {{.*}}, {{.*}} : i4
    %46 = firrtl.neq %in1, %4 : (!firrtl.uint<4>, !firrtl.uint<4>) -> !firrtl.uint<1>

    // Noop
    %47 = firrtl.asClock %44 : (!firrtl.uint<1>) -> !firrtl.clock
    %48 = firrtl.asAsyncReset %44 : (!firrtl.uint<1>) -> !firrtl.asyncreset

    // CHECK: [[VERB1:%.+]] = sv.verbatim.expr "MAGIC_CONSTANT" : () -> i42
    // CHECK: [[VERB2:%.+]] = sv.verbatim.expr "$bits({{[{][{]0[}][}]}}, {{[{][{]1[}][}]}})"([[VERB1]]) : (i42) -> i32 {symbols = [@Simple]}
    // CHECK: [[VERB3:%.+]] = sv.verbatim.expr.se "$size({{[{][{]0[}][}]}}, {{[{][{]1[}][}]}})"([[VERB1]]) : (i42) -> !hw.inout<i32> {symbols = [@Simple]}
    // CHECK: [[VERB3READ:%.+]] = sv.read_inout [[VERB3]]
    // CHECK: [[VERB1EXT:%.+]] = comb.concat {{%.+}}, [[VERB1]] : i1, i42
    // CHECK: [[VERB2EXT:%.+]] = comb.concat {{%.+}}, [[VERB2]] : i11, i32
    // CHECK: [[ADD:%.+]] = comb.add bin [[VERB1EXT]], [[VERB2EXT]] : i43
    // CHECK: [[VERB3EXT:%.+]] = comb.concat {{%.+}}, [[VERB3READ]] : i12, i32
    // CHECK: [[ADDEXT:%.+]] = comb.concat {{%.+}}, [[ADD]] : i1, i43
    // CHECK: = comb.add bin [[VERB3EXT]], [[ADDEXT]] : i44
    %56 = firrtl.verbatim.expr "MAGIC_CONSTANT" : () -> !firrtl.uint<42>
    %57 = firrtl.verbatim.expr "$bits({{0}}, {{1}})"(%56) : (!firrtl.uint<42>) -> !firrtl.uint<32> {symbols = [@Simple]}
    %58 = firrtl.verbatim.wire "$size({{0}}, {{1}})"(%56) : (!firrtl.uint<42>) -> !firrtl.uint<32> {symbols = [@Simple]}
    %59 = firrtl.add %56, %57 : (!firrtl.uint<42>, !firrtl.uint<32>) -> !firrtl.uint<43>
    %60 = firrtl.add %58, %59 : (!firrtl.uint<32>, !firrtl.uint<43>) -> !firrtl.uint<44>

    // Issue #353
    // CHECK: [[PADRES_EXT:%.+]] = comb.concat {{.*}}, [[PADRES]] : i5, i3
    // CHECK: = comb.and bin %in3, [[PADRES_EXT]] : i8
    %49 = firrtl.and %in3, %3 : (!firrtl.sint<8>, !firrtl.sint<3>) -> !firrtl.uint<8>

    // Issue #355: https://github.com/llvm/circt/issues/355
    // CHECK: [[IN1:%.+]] = comb.concat %c0_i6, %in1 : i6, i4
    // CHECK: [[DIV:%.+]] = comb.divu bin [[IN1]], %c306_i10 : i10
    // CHECK: = comb.extract [[DIV]] from 0 : (i10) -> i4
    %c306_ui10 = firrtl.constant 306 : !firrtl.uint<10>
    %50 = firrtl.div %in1, %c306_ui10 : (!firrtl.uint<4>, !firrtl.uint<10>) -> !firrtl.uint<4>

    %c1175_ui11 = firrtl.constant 1175 : !firrtl.uint<11>
    %51 = firrtl.neg %c1175_ui11 : (!firrtl.uint<11>) -> !firrtl.sint<12>
    // https://github.com/llvm/circt/issues/821
    // CHECK: [[CONCAT:%.+]] = comb.concat %false, %in1 : i1, i4
    // CHECK:  = comb.sub bin %c0_i5, [[CONCAT]] : i5
    %52 = firrtl.neg %in1 : (!firrtl.uint<4>) -> !firrtl.sint<5>
    %53 = firrtl.neg %in4 : (!firrtl.uint<0>) -> !firrtl.sint<1>
    // CHECK: [[SEXT:%.+]] = comb.concat {{.*}}, %in3 : i1, i8
    // CHECK: = comb.sub bin %c0_i9, [[SEXT]] : i9
    %54 = firrtl.neg %in3 : (!firrtl.sint<8>) -> !firrtl.sint<9>
    firrtl.connect %out1, %53 : !firrtl.sint<1>, !firrtl.sint<1>
    %55 = firrtl.neg %in5 : (!firrtl.sint<0>) -> !firrtl.sint<1>

    %61 = firrtl.multibit_mux %17, %55, %55, %55 : !firrtl.uint<1>, !firrtl.sint<1>
    // CHECK:      %[[ZEXT_INDEX:.+]] = comb.concat %false, {{.*}} : i1, i1
    // CHECK-NEXT: %[[ARRAY:.+]] = hw.array_create %false, %false, %false
    // CHECK-NEXT: %[[GET0:.+]] = hw.array_get %[[ARRAY]][%c0_i2]
    // CHECK-NEXT: %[[FILLER:.+]] = hw.array_create %[[GET0]] : i1
    // CHECK-NEXT: %[[EXT:.+]] = hw.array_concat %[[FILLER]], %[[ARRAY]]
    // CHECK-NEXT: %[[ARRAY_GET:.+]] = hw.array_get %[[EXT]][%[[ZEXT_INDEX]]]
    // CHECK-NEXT: %[[WIRE:.+]] = sv.wire
    // CHECK-NEXT: sv.assign %[[WIRE]], %[[ARRAY_GET]]
    // CHECK-NEXT: %[[READ_WIRE:.+]] = sv.read_inout %[[WIRE]] : !hw.inout<i1>
    // CHECK: hw.output %false, %[[READ_WIRE]] : i1, i1
    firrtl.connect %out2, %61 : !firrtl.sint<1>, !firrtl.sint<1>
  }

//   module Print :
//    input clock: Clock
//    input reset: UInt<1>
//    input a: UInt<4>
//    input b: UInt<4>
//    printf(clock, reset, "No operands!\n")
//    printf(clock, reset, "Hi %x %x\n", add(a, a), b)

  // CHECK-LABEL: hw.module private @Print
  firrtl.module private @Print(in %clock: !firrtl.clock, in %reset: !firrtl.uint<1>,
                       in %a: !firrtl.uint<4>, in %b: !firrtl.uint<4>) {
    // CHECK: [[ADD:%.+]] = comb.add

    // CHECK:      sv.ifdef "SYNTHESIS" {
    // CHECK-NEXT: } else  {
    // CHECK-NEXT:   sv.always posedge %clock {
    // CHECK-NEXT:     %PRINTF_COND_ = sv.macro.ref< "PRINTF_COND_"> : i1
    // CHECK-NEXT:     [[AND:%.+]] = comb.and bin %PRINTF_COND_, %reset
    // CHECK-NEXT:     sv.if [[AND]] {
    // CHECK-NEXT:       [[FD:%.+]] = hw.constant -2147483646 : i32
    // CHECK-NEXT:       sv.fwrite [[FD]], "No operands!\0A"
    // CHECK-NEXT:     }
    // CHECK-NEXT:     %PRINTF_COND__0 = sv.macro.ref< "PRINTF_COND_"> : i1
    // CHECK-NEXT:     [[AND:%.+]] = comb.and bin %PRINTF_COND__0, %reset : i1
    // CHECK-NEXT:     sv.if [[AND]] {
    // CHECK-NEXT:       [[FD:%.+]] = hw.constant -2147483646 : i32
    // CHECK-NEXT:       sv.fwrite [[FD]], "Hi %x %x\0A"(%2, %b) : i5, i4
    // CHECK-NEXT:     }
    // CHECK-NEXT:   }
    // CHECK-NEXT: }
   firrtl.printf %clock, %reset, "No operands!\0A"

    %0 = firrtl.add %a, %a : (!firrtl.uint<4>, !firrtl.uint<4>) -> !firrtl.uint<5>

    firrtl.printf %clock, %reset, "Hi %x %x\0A"(%0, %b) : !firrtl.uint<5>, !firrtl.uint<4>

    firrtl.skip

    // CHECK: hw.output
   }



// module Stop3 :
//    input clock1: Clock
//    input clock2: Clock
//    input reset: UInt<1>
//    stop(clock1, reset, 42)
//    stop(clock2, reset, 0)

  // CHECK-LABEL: hw.module private @Stop
  firrtl.module private @Stop(in %clock1: !firrtl.clock, in %clock2: !firrtl.clock, in %reset: !firrtl.uint<1>) {

    // CHECK-NEXT: sv.ifdef "SYNTHESIS" {
    // CHECK-NEXT: } else {
    // CHECK-NEXT:   sv.always posedge %clock1 {
    // CHECK-NEXT:     %STOP_COND_ = sv.macro.ref< "STOP_COND_"> : i1
    // CHECK-NEXT:     %0 = comb.and bin %STOP_COND_, %reset : i1
    // CHECK-NEXT:     sv.if %0 {
    // CHECK-NEXT:       sv.fatal
    // CHECK-NEXT:     }
    // CHECK-NEXT:   }
    firrtl.stop %clock1, %reset, 42

    // CHECK-NEXT:   sv.always posedge %clock2 {
    // CHECK-NEXT:     %STOP_COND_ = sv.macro.ref< "STOP_COND_"> : i1
    // CHECK-NEXT:     %0 = comb.and bin %STOP_COND_, %reset : i1
    // CHECK-NEXT:     sv.if %0 {
    // CHECK-NEXT:       sv.finish
    // CHECK-NEXT:     }
    // CHECK-NEXT:   }
    // CHECK-NEXT: }
    firrtl.stop %clock2, %reset, 0
  }

  // circuit Verification:
  //   module Verification:
  //     input clock: Clock
  //     input aCond: UInt<8>
  //     input aEn: UInt<8>
  //     input bCond: UInt<1>
  //     input bEn: UInt<1>
  //     input cCond: UInt<1>
  //     input cEn: UInt<1>
  //     assert(clock, bCond, bEn, "assert0")
  //     assert(clock, bCond, bEn, "assert0") : assert_0
  //     assume(clock, aCond, aEn, "assume0")
  //     assume(clock, aCond, aEn, "assume0") : assume_0
  //     cover(clock,  cCond, cEn, "cover0)"
  //     cover(clock,  cCond, cEn, "cover0)" : cover_0

  // CHECK-LABEL: hw.module private @Verification
  firrtl.module private @Verification(in %clock: !firrtl.clock, in %aCond: !firrtl.uint<1>,
    in %aEn: !firrtl.uint<1>, in %bCond: !firrtl.uint<1>, in %bEn: !firrtl.uint<1>,
    in %cCond: !firrtl.uint<1>, in %cEn: !firrtl.uint<1>, in %value: !firrtl.uint<42>) {

    firrtl.assert %clock, %aCond, %aEn, "assert0" {isConcurrent = true}
    firrtl.assert %clock, %aCond, %aEn, "assert0" {isConcurrent = true, name = "assert_0"}
    firrtl.assert %clock, %aCond, %aEn, "assert0"(%value) : !firrtl.uint<42> {isConcurrent = true}
    // CHECK-NEXT: [[TRUE:%.+]] = hw.constant true
    // CHECK-NEXT: [[TMP1:%.+]] = comb.xor %aEn, [[TRUE]]
    // CHECK-NEXT: [[TMP2:%.+]] = comb.or bin [[TMP1]], %aCond
    // CHECK-NEXT: sv.assert.concurrent posedge %clock, [[TMP2]] message "assert0"
    // CHECK-NEXT: [[TRUE:%.+]] = hw.constant true
    // CHECK-NEXT: [[TMP3:%.+]] = comb.xor %aEn, [[TRUE]]
    // CHECK-NEXT: [[TMP4:%.+]] = comb.or bin [[TMP3]], %aCond
    // CHECK-NEXT: sv.assert.concurrent posedge %clock, [[TMP4]] label "assert__assert_0" message "assert0"
    // CHECK-NEXT: [[SAMPLED:%.+]] =  sv.system.sampled %value : i42
    // CHECK-NEXT: [[TRUE:%.+]] = hw.constant true
    // CHECK-NEXT: [[TMP5:%.+]] = comb.xor %aEn, [[TRUE]]
    // CHECK-NEXT: [[TMP6:%.+]] = comb.or bin [[TMP5]], %aCond
    // CHECK-NEXT: sv.assert.concurrent posedge %clock, [[TMP6]] message "assert0"([[SAMPLED]]) : i42
    // CHECK-NEXT: sv.ifdef "USE_PROPERTY_AS_CONSTRAINT" {
    // CHECK-NEXT:   sv.assume.concurrent posedge %clock, [[TMP2]]
    // CHECK-NEXT:   sv.assume.concurrent posedge %clock, [[TMP4]] label "assume__assert_0"
    // CHECK-NEXT:   sv.assume.concurrent posedge %clock, [[TMP6]]
    // CHECK-NEXT: }
    firrtl.assume %clock, %bCond, %bEn, "assume0" {isConcurrent = true}
    firrtl.assume %clock, %bCond, %bEn, "assume0" {isConcurrent = true, name = "assume_0"}
    firrtl.assume %clock, %bCond, %bEn, "assume0"(%value) : !firrtl.uint<42> {isConcurrent = true}
    // CHECK-NEXT: [[TRUE:%.+]] = hw.constant true
    // CHECK-NEXT: [[TMP1:%.+]] = comb.xor %bEn, [[TRUE]]
    // CHECK-NEXT: [[TMP2:%.+]] = comb.or bin [[TMP1]], %bCond
    // CHECK-NEXT: sv.assume.concurrent posedge %clock, [[TMP2]] message "assume0"
    // CHECK-NEXT: [[TRUE:%.+]] = hw.constant true
    // CHECK-NEXT: [[TMP1:%.+]] = comb.xor %bEn, [[TRUE]]
    // CHECK-NEXT: [[TMP2:%.+]] = comb.or bin [[TMP1]], %bCond
    // CHECK-NEXT: sv.assume.concurrent posedge %clock, [[TMP2]] label "assume__assume_0" message "assume0"
    // CHECK-NEXT: [[SAMPLED:%.+]] = sv.system.sampled %value
    // CHECK-NEXT: [[TRUE:%.+]] = hw.constant true
    // CHECK-NEXT: [[TMP1:%.+]] = comb.xor %bEn, [[TRUE]]
    // CHECK-NEXT: [[TMP2:%.+]] = comb.or bin [[TMP1]], %bCond
    // CHECK-NEXT: sv.assume.concurrent posedge %clock, [[TMP2]] message "assume0"([[SAMPLED]]) : i42
    firrtl.cover %clock, %cCond, %cEn, "cover0" {isConcurrent = true}
    firrtl.cover %clock, %cCond, %cEn, "cover0" {isConcurrent = true, name = "cover_0"}
    firrtl.cover %clock, %cCond, %cEn, "cover0"(%value) : !firrtl.uint<42> {isConcurrent = true}
    // CHECK-NEXT: [[TMP:%.+]] = comb.and bin %cEn, %cCond
    // CHECK-NEXT: sv.cover.concurrent posedge %clock, [[TMP]]
    // CHECK-NEXT: [[TMP:%.+]] = comb.and bin %cEn, %cCond
    // CHECK-NEXT: sv.cover.concurrent posedge %clock, [[TMP]] label "cover__cover_0"
    // CHECK-NEXT: [[TMP:%.+]] = comb.and bin %cEn, %cCond
    // CHECK-NEXT: sv.cover.concurrent posedge %clock, [[TMP]]
    firrtl.cover %clock, %cCond, %cEn, "cover1" {eventControl = 1 : i32, isConcurrent = true, name = "cover_1"}
    firrtl.cover %clock, %cCond, %cEn, "cover2" {eventControl = 2 : i32, isConcurrent = true, name = "cover_2"}
    // CHECK: sv.cover.concurrent negedge %clock, {{%.+}} label "cover__cover_1"
    // CHECK: sv.cover.concurrent edge %clock, {{%.+}} label "cover__cover_2"

    // CHECK-NEXT: sv.always posedge %clock {
    // CHECK-NEXT:   sv.if %aEn {
    // CHECK-NEXT:     sv.assert %aCond, immediate message "assert0"
    // CHECK-NEXT:     sv.assert %aCond, immediate label "assert__assert_0" message "assert0"
    // CHECK-NEXT:     sv.assert %aCond, immediate message "assert0"(%value) : i42
    // CHECK-NEXT:   }
    // CHECK-NEXT:   sv.if %bEn {
    // CHECK-NEXT:     sv.assume %bCond, immediate message "assume0"
    // CHECK-NEXT:     sv.assume %bCond, immediate label "assume__assume_0" message "assume0"
    // CHECK-NEXT:     sv.assume %bCond, immediate message "assume0"(%value) : i42
    // CHECK-NEXT:   }
    // CHECK-NEXT:   sv.if %cEn {
    // CHECK-NEXT:     sv.cover %cCond, immediate
    // CHECK-NOT:        label
    // CHECK-NEXT:     sv.cover %cCond, immediate label "cover__cover_0"
    // CHECK-NEXT:     sv.cover %cCond, immediate
    // CHECK-NOT:        label
    // CHECK-NEXT:   }
    // CHECK-NEXT: }
    firrtl.assert %clock, %aCond, %aEn, "assert0"
    firrtl.assert %clock, %aCond, %aEn, "assert0" {name = "assert_0"}
    firrtl.assert %clock, %aCond, %aEn, "assert0"(%value) : !firrtl.uint<42>
    firrtl.assume %clock, %bCond, %bEn, "assume0"
    firrtl.assume %clock, %bCond, %bEn, "assume0" {name = "assume_0"}
    firrtl.assume %clock, %bCond, %bEn, "assume0"(%value) : !firrtl.uint<42>
    firrtl.cover %clock, %cCond, %cEn, "cover0"
    firrtl.cover %clock, %cCond, %cEn, "cover0" {name = "cover_0"}
    firrtl.cover %clock, %cCond, %cEn, "cover0"(%value) : !firrtl.uint<42>
    // CHECK-NEXT: hw.output
  }

  // CHECK-LABEL: hw.module private @VerificationGuards
  firrtl.module private @VerificationGuards(
    in %clock: !firrtl.clock,
    in %cond: !firrtl.uint<1>,
    in %enable: !firrtl.uint<1>
  ) {
    firrtl.assert %clock, %cond, %enable, "assert0" {isConcurrent = true, guards = ["HELLO", "WORLD"]}
    firrtl.assume %clock, %cond, %enable, "assume0" {isConcurrent = true, guards = ["HELLO", "WORLD"]}
    firrtl.cover %clock, %cond, %enable, "cover0" {isConcurrent = true, guards = ["HELLO", "WORLD"]}

    // CHECK-NEXT: sv.ifdef "HELLO" {
    // CHECK-NEXT:   sv.ifdef "WORLD" {
    // CHECK-NEXT:     [[TRUE:%.+]] = hw.constant true
    // CHECK-NEXT:     [[TMP1:%.+]] = comb.xor %enable, [[TRUE]]
    // CHECK-NEXT:     [[TMP2:%.+]] = comb.or bin [[TMP1]], %cond
    // CHECK-NEXT:     sv.assert.concurrent posedge %clock, [[TMP2]] message "assert0"
    // CHECK-NEXT:     sv.ifdef "USE_PROPERTY_AS_CONSTRAINT" {
    // CHECK-NEXT:       sv.assume.concurrent posedge %clock, [[TMP2]]
    // CHECK-NEXT:     }
    // CHECK-NEXT:     [[TRUE:%.+]] = hw.constant true
    // CHECK-NEXT:     [[TMP1:%.+]] = comb.xor %enable, [[TRUE]]
    // CHECK-NEXT:     [[TMP2:%.+]] = comb.or bin [[TMP1]], %cond
    // CHECK-NEXT:     sv.assume.concurrent posedge %clock, [[TMP2]] message "assume0"
    // CHECK-NEXT:     [[TMP:%.+]] = comb.and bin %enable, %cond
    // CHECK-NEXT:     sv.cover.concurrent posedge %clock, [[TMP]]
    // CHECK-NOT:      label
    // CHECK-NEXT:   }
    // CHECK-NEXT: }
  }

  // CHECK-LABEL: hw.module private @VerificationAssertFormat
  firrtl.module private @VerificationAssertFormat(
    in %clock: !firrtl.clock,
    in %cond: !firrtl.uint<1>,
    in %enable: !firrtl.uint<1>,
    in %value: !firrtl.uint<42>
  ) {
    firrtl.assert %clock, %cond, %enable, "assert0" {isConcurrent = true, format = "sva"}
    // CHECK-NEXT: [[TRUE:%.+]] = hw.constant true
    // CHECK-NEXT: [[TMP1:%.+]] = comb.xor %enable, [[TRUE]]
    // CHECK-NEXT: [[TMP2:%.+]] = comb.or bin [[TMP1]], %cond
    // CHECK-NEXT: sv.assert.concurrent posedge %clock, [[TMP2]] message "assert0"
    // CHECK-NEXT: sv.ifdef "USE_PROPERTY_AS_CONSTRAINT" {
    // CHECK-NEXT:   sv.assume.concurrent posedge %clock, [[TMP2]]
    // CHECK-NEXT: }
    firrtl.assert %clock, %cond, %enable, "assert1"(%value) : !firrtl.uint<42> {isConcurrent = true, format = "ifElseFatal"}
    // CHECK-NEXT: [[TRUE:%.+]] = hw.constant true
    // CHECK-NEXT: [[TMP1:%.+]] = comb.xor %cond, [[TRUE]]
    // CHECK-NEXT: [[TMP2:%.+]] = comb.and bin %enable, [[TMP1]]
    // CHECK-NEXT: sv.ifdef "SYNTHESIS" {
    // CHECK-NEXT: } else {
    // CHECK-NEXT:   sv.always posedge %clock {
    // CHECK-NEXT:     sv.if [[TMP2]] {
    // CHECK-NEXT:       [[ASSERT_VERBOSE_COND:%.+]] = sv.macro.ref< "ASSERT_VERBOSE_COND_"> : i1
    // CHECK-NEXT:       sv.if [[ASSERT_VERBOSE_COND]] {
    // CHECK-NEXT:         sv.error "assert1"(%value) : i42
    // CHECK-NEXT:       }
    // CHECK-NEXT:       [[STOP_COND:%.+]] = sv.macro.ref< "STOP_COND_"> : i1
    // CHECK-NEXT:       sv.if [[STOP_COND]] {
    // CHECK-NEXT:         sv.fatal
    // CHECK-NEXT:       }
    // CHECK-NEXT:     }
    // CHECK-NEXT:   }
    // CHECK-NEXT: }
  }

  firrtl.module private @bar(in %io_cpu_flush: !firrtl.uint<1>) {
    // CHECK: hw.probe @baz, %io_cpu_flush, %io_cpu_flush : i1, i1
    firrtl.probe @baz, %io_cpu_flush, %io_cpu_flush  : !firrtl.uint<1>, !firrtl.uint<1>
  }

  // CHECK-LABEL: hw.module private @foo
  firrtl.module private @foo() {
    // CHECK-NEXT:  %io_cpu_flush.wire = sv.wire sym @__foo__io_cpu_flush.wire : !hw.inout<i1>
    // CHECK-NEXT:  [[IO:%[0-9]+]] = sv.read_inout %io_cpu_flush.wire
    %io_cpu_flush.wire = firrtl.wire {annotations = [{class = "firrtl.transforms.DontTouchAnnotation"}]} : !firrtl.uint<1>
    // CHECK-NEXT: hw.instance "fetch" @bar(io_cpu_flush: [[IO]]: i1)
    %i = firrtl.instance fetch @bar(in io_cpu_flush: !firrtl.uint<1>)
    firrtl.connect %i, %io_cpu_flush.wire : !firrtl.uint<1>, !firrtl.uint<1>

    %hits_1_7 = firrtl.node %io_cpu_flush.wire {name = "hits_1_7", annotations = [{class = "firrtl.transforms.DontTouchAnnotation"}]} : !firrtl.uint<1>
    // CHECK-NEXT:  %hits_1_7 = sv.wire sym @__foo__hits_1_7
    // CHECK-NEXT:  sv.assign %hits_1_7, [[IO]] : i1
    %1455 = builtin.unrealized_conversion_cast %hits_1_7 : !firrtl.uint<1> to !firrtl.uint<1>
  }

  // CHECK: sv.bind <@bindTest::@[[bazSymbol:.+]]>
  // CHECK-NOT: output_file
  // CHECK-NEXT: sv.bind <@bindTest::@[[quxSymbol:.+]]> {
  // CHECK-SAME: output_file = #hw.output_file<"bindings.sv", excludeFromFileList>
  // CHECK-NEXT: hw.module private @bindTest()
  firrtl.module private @bindTest() {
    // CHECK: hw.instance "baz" sym @[[bazSymbol]] @bar
    %baz = firrtl.instance baz {lowerToBind} @bar(in io_cpu_flush: !firrtl.uint<1>)
    // CHECK: hw.instance "qux" sym @[[quxSymbol]] @bar
    %qux = firrtl.instance qux {lowerToBind, output_file = #hw.output_file<"bindings.sv", excludeFromFileList>} @bar(in io_cpu_flush: !firrtl.uint<1>)
  }


  // CHECK-LABEL: hw.module private @output_fileTest
  // CHECK-SAME: output_file = #hw.output_file<"output_fileTest.sv", excludeFromFileList>
  firrtl.module private @output_fileTest() attributes {
      output_file = #hw.output_file<"output_fileTest.sv", excludeFromFileList >} {
  }

  // https://github.com/llvm/circt/issues/314
  // CHECK-LABEL: hw.module private @issue314
  firrtl.module private @issue314(in %inp_2: !firrtl.uint<27>, in %inpi: !firrtl.uint<65>) {
    // CHECK: %c0_i38 = hw.constant 0 : i38
    // CHECK: %tmp48 = sv.wire
    %tmp48 = firrtl.wire : !firrtl.uint<27>

    // CHECK-NEXT: %0 = comb.concat %c0_i38, %inp_2 : i38, i27
    // CHECK-NEXT: %1 = comb.divu bin %0, %inpi : i65
    %0 = firrtl.div %inp_2, %inpi : (!firrtl.uint<27>, !firrtl.uint<65>) -> !firrtl.uint<27>
    // CHECK-NEXT: %2 = comb.extract %1 from 0 : (i65) -> i27
    // CHECK-NEXT: sv.assign %tmp48, %2 : i27
    firrtl.connect %tmp48, %0 : !firrtl.uint<27>, !firrtl.uint<27>
  }

  // https://github.com/llvm/circt/issues/318
  // CHECK-LABEL: hw.module private @test_rem
  // CHECK-NEXT:     %0 = comb.modu
  // CHECK-NEXT:     hw.output %0
  firrtl.module private @test_rem(in %tmp85: !firrtl.uint<1>, in %tmp79: !firrtl.uint<1>,
       out %out: !firrtl.uint<1>) {
    %2 = firrtl.rem %tmp79, %tmp85 : (!firrtl.uint<1>, !firrtl.uint<1>) -> !firrtl.uint<1>
    firrtl.connect %out, %2 : !firrtl.uint<1>, !firrtl.uint<1>
  }

  // CHECK-LABEL: hw.module private @Analog(%a1: !hw.inout<i1>, %b1: !hw.inout<i1>,
  // CHECK:                          %c1: !hw.inout<i1>) -> (outClock: i1) {
  // CHECK-NEXT:   %0 = sv.read_inout %c1 : !hw.inout<i1>
  // CHECK-NEXT:   %1 = sv.read_inout %b1 : !hw.inout<i1>
  // CHECK-NEXT:   %2 = sv.read_inout %a1 : !hw.inout<i1>
  // CHECK-NEXT:   sv.ifdef "SYNTHESIS"  {
  // CHECK-NEXT:     sv.assign %a1, %1 : i1
  // CHECK-NEXT:     sv.assign %a1, %0 : i1
  // CHECK-NEXT:     sv.assign %b1, %2 : i1
  // CHECK-NEXT:     sv.assign %b1, %0 : i1
  // CHECK-NEXT:     sv.assign %c1, %2 : i1
  // CHECK-NEXT:     sv.assign %c1, %1 : i1
  // CHECK-NEXT:    } else {
  // CHECK-NEXT:     sv.ifdef "verilator" {
  // CHECK-NEXT:       sv.verbatim "`error \22Verilator does not support alias and thus cannot arbitrarily connect bidirectional wires and ports\22"
  // CHECK-NEXT:     } else {
  // CHECK-NEXT:       sv.alias %a1, %b1, %c1 : !hw.inout<i1>
  // CHECK-NEXT:     }
  // CHECK-NEXT:    }
  // CHECK-NEXT:    hw.output %2 : i1
  firrtl.module private @Analog(in %a1: !firrtl.analog<1>, in %b1: !firrtl.analog<1>,
                        in %c1: !firrtl.analog<1>, out %outClock: !firrtl.clock) {
    firrtl.attach %a1, %b1, %c1 : !firrtl.analog<1>, !firrtl.analog<1>, !firrtl.analog<1>

    %1 = firrtl.asClock %a1 : (!firrtl.analog<1>) -> !firrtl.clock
    firrtl.connect %outClock, %1 : !firrtl.clock, !firrtl.clock
  }

  //  module MemSimple :
  //     input clock1  : Clock
  //     input clock2  : Clock
  //     input inpred  : UInt<1>
  //     input indata  : SInt<42>
  //     output result : SInt<42>
  //     output result2 : SInt<42>
  //
  //     mem _M : @[Decoupled.scala 209:27]
  //           data-type => SInt<42>
  //           depth => 12
  //           read-latency => 0
  //           write-latency => 1
  //           reader => read
  //           writer => write
  //           readwriter => rw
  //           read-under-write => undefined
  //
  //     result <= _M.read.data
  //     result2 <= _M.rw.rdata
  //
  //     _M.read.addr <= UInt<1>("h0")
  //     _M.read.en <= UInt<1>("h1")
  //     _M.read.clk <= clock1
  //     _M.rw.addr <= UInt<1>("h0")
  //     _M.rw.en <= UInt<1>("h1")
  //     _M.rw.clk <= clock1
  //     _M.rw.wmask <= UInt<1>("h1")
  //     _M.rw.wmode <= UInt<1>("h1")
  //     _M.write.addr <= validif(inpred, UInt<3>("h0"))
  //     _M.write.en <= mux(inpred, UInt<1>("h1"), UInt<1>("h0"))
  //     _M.write.clk <= clock2
  //     _M.write.data <= validif(inpred, indata)
  //     _M.write.mask <= validif(inpred, UInt<1>("h1"))

  // CHECK-LABEL: hw.module private @MemSimple(
  firrtl.module private @MemSimple(in %clock1: !firrtl.clock, in %clock2: !firrtl.clock,
                           in %inpred: !firrtl.uint<1>, in %indata: !firrtl.sint<42>,
                           out %result: !firrtl.sint<42>,
                           out %result2: !firrtl.sint<42>) {
    %c0_ui1 = firrtl.constant 0 : !firrtl.uint<1>
    %c1_ui1 = firrtl.constant 1 : !firrtl.uint<1>
    %c0_ui3 = firrtl.constant 0 : !firrtl.uint<3>
    %_M_read, %_M_rw, %_M_write = firrtl.mem Undefined {depth = 12 : i64, name = "_M", portNames = ["read", "rw", "write"], readLatency = 0 : i32, writeLatency = 1 : i32} : !firrtl.bundle<addr: uint<4>, en: uint<1>, clk: clock, data flip: sint<42>>, !firrtl.bundle<addr: uint<4>, en: uint<1>, clk: clock, rdata flip: sint<42>, wmode: uint<1>, wdata: sint<42>, wmask: uint<1>>, !firrtl.bundle<addr: uint<4>, en: uint<1>, clk: clock, data: sint<42>, mask: uint<1>>
  // CHECK: %[[v1:.+]] = comb.and bin %true, %inpred : i1
  // CHECK: %[[v2:.+]] = comb.and bin %inpred, %true : i1
  // CHECK: %_M_ext.R0_data, %_M_ext.RW0_rdata = hw.instance "_M_ext" @_M_combMem_0(R0_addr: %c0_i4: i4, R0_en: %true: i1, R0_clk: %clock1: i1, RW0_addr: %c0_i4_0: i4, RW0_en: %true: i1, RW0_clk: %clock1: i1, RW0_wmode: %[[v1]]: i1, RW0_wdata: %1: i42, W0_addr: %c0_i4_1: i4, W0_en: %[[v2]]: i1, W0_clk: %clock2: i1, W0_data: %indata: i42) -> (R0_data: i42, RW0_rdata: i42)
  // CHECK: hw.output %_M_ext.R0_data, %_M_ext.RW0_rdata : i42, i42

      %0 = firrtl.subfield %_M_read(3) : (!firrtl.bundle<addr: uint<4>, en: uint<1>, clk: clock, data flip: sint<42>>) -> !firrtl.sint<42>
      firrtl.connect %result, %0 : !firrtl.sint<42>, !firrtl.sint<42>
      %1 = firrtl.subfield %_M_rw(3) : (!firrtl.bundle<addr: uint<4>, en: uint<1>, clk: clock, rdata flip: sint<42>, wmode: uint<1>, wdata: sint<42>, wmask: uint<1>>) -> !firrtl.sint<42>
      firrtl.connect %result2, %1 : !firrtl.sint<42>, !firrtl.sint<42>
      %2 = firrtl.subfield %_M_read(0) : (!firrtl.bundle<addr: uint<4>, en: uint<1>, clk: clock, data flip: sint<42>>) -> !firrtl.uint<4>
      firrtl.connect %2, %c0_ui1 : !firrtl.uint<4>, !firrtl.uint<1>
      %3 = firrtl.subfield %_M_read(1) : (!firrtl.bundle<addr: uint<4>, en: uint<1>, clk: clock, data flip: sint<42>>) -> !firrtl.uint<1>
      firrtl.connect %3, %c1_ui1 : !firrtl.uint<1>, !firrtl.uint<1>
      %4 = firrtl.subfield %_M_read(2) : (!firrtl.bundle<addr: uint<4>, en: uint<1>, clk: clock, data flip: sint<42>>) -> !firrtl.clock
      firrtl.connect %4, %clock1 : !firrtl.clock, !firrtl.clock

      %5 = firrtl.subfield %_M_rw(0) : (!firrtl.bundle<addr: uint<4>, en: uint<1>, clk: clock, rdata flip: sint<42>, wmode: uint<1>, wdata: sint<42>, wmask: uint<1>>) -> !firrtl.uint<4>
      firrtl.connect %5, %c0_ui1 : !firrtl.uint<4>, !firrtl.uint<1>
      %6 = firrtl.subfield %_M_rw(1) : (!firrtl.bundle<addr: uint<4>, en: uint<1>, clk: clock, rdata flip: sint<42>, wmode: uint<1>, wdata: sint<42>, wmask: uint<1>>) -> !firrtl.uint<1>
      firrtl.connect %6, %c1_ui1 : !firrtl.uint<1>, !firrtl.uint<1>
      %7 = firrtl.subfield %_M_rw(2) : (!firrtl.bundle<addr: uint<4>, en: uint<1>, clk: clock, rdata flip: sint<42>, wmode: uint<1>, wdata: sint<42>, wmask: uint<1>>) -> !firrtl.clock
      firrtl.connect %7, %clock1 : !firrtl.clock, !firrtl.clock
      %8 = firrtl.subfield %_M_rw(6) : (!firrtl.bundle<addr: uint<4>, en: uint<1>, clk: clock, rdata flip: sint<42>, wmode: uint<1>, wdata: sint<42>, wmask: uint<1>>) -> !firrtl.uint<1>
      firrtl.connect %8, %inpred : !firrtl.uint<1>, !firrtl.uint<1>
      %9 = firrtl.subfield %_M_rw(4) : (!firrtl.bundle<addr: uint<4>, en: uint<1>, clk: clock, rdata flip: sint<42>, wmode: uint<1>, wdata: sint<42>, wmask: uint<1>>) -> !firrtl.uint<1>
      firrtl.connect %9, %c1_ui1 : !firrtl.uint<1>, !firrtl.uint<1>

      %10 = firrtl.subfield %_M_write(0) : (!firrtl.bundle<addr: uint<4>, en: uint<1>, clk: clock, data: sint<42>, mask: uint<1>>) -> !firrtl.uint<4>
      firrtl.connect %10, %c0_ui3 : !firrtl.uint<4>, !firrtl.uint<3>
      %11 = firrtl.subfield %_M_write(1) : (!firrtl.bundle<addr: uint<4>, en: uint<1>, clk: clock, data: sint<42>, mask: uint<1>>) -> !firrtl.uint<1>
      firrtl.connect %11, %inpred : !firrtl.uint<1>, !firrtl.uint<1>
      %12 = firrtl.subfield %_M_write(2) : (!firrtl.bundle<addr: uint<4>, en: uint<1>, clk: clock, data: sint<42>, mask: uint<1>>) -> !firrtl.clock
      firrtl.connect %12, %clock2 : !firrtl.clock, !firrtl.clock
      %13 = firrtl.subfield %_M_write(3) : (!firrtl.bundle<addr: uint<4>, en: uint<1>, clk: clock, data: sint<42>, mask: uint<1>>) -> !firrtl.sint<42>
      firrtl.connect %13, %indata : !firrtl.sint<42>, !firrtl.sint<42>
      %14 = firrtl.subfield %_M_write(4) : (!firrtl.bundle<addr: uint<4>, en: uint<1>, clk: clock, data: sint<42>, mask: uint<1>>) -> !firrtl.uint<1>
      firrtl.connect %14, %c1_ui1 : !firrtl.uint<1>, !firrtl.uint<1>
  }

  // CHECK-LABEL: hw.module private @MemSimple_mask(
  firrtl.module private @MemSimple_mask(in %clock1: !firrtl.clock, in %clock2: !firrtl.clock,
                           in %inpred: !firrtl.uint<1>, in %indata: !firrtl.sint<40>,
                           out %result: !firrtl.sint<40>,
                           out %result2: !firrtl.sint<40>) {
    %c0_ui1 = firrtl.constant 0 : !firrtl.uint<1>
    %c0_ui10 = firrtl.constant 0 : !firrtl.uint<10>
    %c1_ui1 = firrtl.constant 1 : !firrtl.uint<1>
    %c0_ui3 = firrtl.constant 0 : !firrtl.uint<3>
    %c0_ui4 = firrtl.constant 0 : !firrtl.uint<4>
    %c1_ui5 = firrtl.constant 1 : !firrtl.uint<5>
    %_M_read, %_M_rw, %_M_write = firrtl.mem Undefined {depth = 1022 : i64, name = "_M_mask", portNames = ["read", "rw", "write"], readLatency = 1 : i32, writeLatency = 1 : i32} : !firrtl.bundle<addr: uint<10>, en: uint<1>, clk: clock, data flip: sint<40>>, !firrtl.bundle<addr: uint<10>, en: uint<1>, clk: clock, rdata flip: sint<40>, wmode: uint<1>, wdata: sint<40>, wmask: uint<4>>, !firrtl.bundle<addr: uint<10>, en: uint<1>, clk: clock, data: sint<40>, mask: uint<4>>
    // CHECK: %_M_mask_ext.R0_data, %_M_mask_ext.RW0_rdata = hw.instance "_M_mask_ext" @_M_mask_combMem(R0_addr: %c0_i10: i10, R0_en: %true: i1, R0_clk: %clock1: i1, RW0_addr: %c0_i10: i10, RW0_en: %true: i1, RW0_clk: %clock1: i1, RW0_wmode: %true: i1, RW0_wdata: %0: i40, RW0_wmask: %c0_i4: i4, W0_addr: %c0_i10: i10, W0_en: %inpred: i1, W0_clk: %clock2: i1, W0_data: %indata: i40, W0_mask: %c0_i4: i4) -> (R0_data: i40, RW0_rdata: i40)
    // CHECK: hw.output %_M_mask_ext.R0_data, %_M_mask_ext.RW0_rdata : i40, i40

      %0 = firrtl.subfield %_M_read(3) : (!firrtl.bundle<addr: uint<10>, en: uint<1>, clk: clock, data flip: sint<40>>) -> !firrtl.sint<40>
      firrtl.connect %result, %0 : !firrtl.sint<40>, !firrtl.sint<40>
      %1 = firrtl.subfield %_M_rw(3) : (!firrtl.bundle<addr: uint<10>, en: uint<1>, clk: clock, rdata flip: sint<40>, wmode: uint<1>, wdata: sint<40>, wmask: uint<4>>) -> !firrtl.sint<40>
      firrtl.connect %result2, %1 : !firrtl.sint<40>, !firrtl.sint<40>
      %2 = firrtl.subfield %_M_read(0) : (!firrtl.bundle<addr: uint<10>, en: uint<1>, clk: clock, data flip: sint<40>>)      -> !firrtl.uint<10>
      firrtl.connect %2, %c0_ui10 : !firrtl.uint<10>, !firrtl.uint<10>
      %3 = firrtl.subfield %_M_read(1) : (!firrtl.bundle<addr: uint<10>, en: uint<1>, clk: clock, data flip: sint<40>>) -> !firrtl.uint<1>
      firrtl.connect %3, %c1_ui1 : !firrtl.uint<1>, !firrtl.uint<1>
      %4 = firrtl.subfield %_M_read(2) : (!firrtl.bundle<addr: uint<10>, en: uint<1>, clk: clock, data flip: sint<40>>) -> !firrtl.clock
      firrtl.connect %4, %clock1 : !firrtl.clock, !firrtl.clock

      %5 = firrtl.subfield %_M_rw(0) : (!firrtl.bundle<addr: uint<10>, en: uint<1>, clk: clock, rdata flip: sint<40>,  wmode: uint<1>, wdata: sint<40>, wmask: uint<4>>) -> !firrtl.uint<10>
      firrtl.connect %5, %c0_ui10 : !firrtl.uint<10>, !firrtl.uint<10>
      %6 = firrtl.subfield %_M_rw(1) : (!firrtl.bundle<addr: uint<10>, en: uint<1>, clk: clock, rdata flip: sint<40>, wmode: uint<1>, wdata: sint<40>, wmask: uint<4>>) -> !firrtl.uint<1>
      firrtl.connect %6, %c1_ui1 : !firrtl.uint<1>, !firrtl.uint<1>
      %7 = firrtl.subfield %_M_rw(2) : (!firrtl.bundle<addr: uint<10>, en: uint<1>, clk: clock, rdata flip: sint<40>, wmode: uint<1>, wdata: sint<40>, wmask: uint<4>>) -> !firrtl.clock
      firrtl.connect %7, %clock1 : !firrtl.clock, !firrtl.clock
      %8 = firrtl.subfield %_M_rw(6) : (!firrtl.bundle<addr: uint<10>, en: uint<1>, clk: clock, rdata flip: sint<40>, wmode: uint<1>, wdata: sint<40>, wmask: uint<4>>) -> !firrtl.uint<4>
      firrtl.connect %8, %c0_ui4 : !firrtl.uint<4>, !firrtl.uint<4>
      %9 = firrtl.subfield %_M_rw(4) : (!firrtl.bundle<addr: uint<10>, en: uint<1>, clk: clock, rdata flip: sint<40>, wmode: uint<1>, wdata: sint<40>, wmask: uint<4>>) -> !firrtl.uint<1>
      firrtl.connect %9, %c1_ui1 : !firrtl.uint<1>, !firrtl.uint<1>

      %10 = firrtl.subfield %_M_write(0) : (!firrtl.bundle<addr: uint<10>, en: uint<1>, clk: clock, data: sint<40>,
      mask: uint<4>>) -> !firrtl.uint<10>
      firrtl.connect %10, %c0_ui10 : !firrtl.uint<10>, !firrtl.uint<10>
      %11 = firrtl.subfield %_M_write(1) : (!firrtl.bundle<addr: uint<10>, en: uint<1>, clk: clock, data: sint<40>, mask: uint<4>>) -> !firrtl.uint<1>
      firrtl.connect %11, %inpred : !firrtl.uint<1>, !firrtl.uint<1>
      %12 = firrtl.subfield %_M_write(2) : (!firrtl.bundle<addr: uint<10>, en: uint<1>, clk: clock, data: sint<40>, mask: uint<4>>) -> !firrtl.clock
      firrtl.connect %12, %clock2 : !firrtl.clock, !firrtl.clock
      %13 = firrtl.subfield %_M_write(3) : (!firrtl.bundle<addr: uint<10>, en: uint<1>, clk: clock, data: sint<40>, mask: uint<4>>) -> !firrtl.sint<40>
      firrtl.connect %13, %indata : !firrtl.sint<40>, !firrtl.sint<40>
      %14 = firrtl.subfield %_M_write(4) : (!firrtl.bundle<addr: uint<10>, en: uint<1>, clk: clock, data: sint<40>, mask: uint<4>>) -> !firrtl.uint<4>
      firrtl.connect %14, %c0_ui4 : !firrtl.uint<4>, !firrtl.uint<4>
  }
  // CHECK-LABEL: hw.module private @IncompleteRead(
  // The read port has no use of the data field.
  firrtl.module private @IncompleteRead(in %clock1: !firrtl.clock) {
    %c0_ui1 = firrtl.constant 0 : !firrtl.uint<1>
    %c1_ui1 = firrtl.constant 1 : !firrtl.uint<1>

    // CHECK:  %_M_ext.R0_data = hw.instance "_M_ext" @_M_combMem(R0_addr: %c0_i4: i4, R0_en: %true: i1, R0_clk: %clock1: i1) -> (R0_data: i42)
    %_M_read = firrtl.mem Undefined {depth = 12 : i64, name = "_M", portNames = ["read"], readLatency = 0 : i32, writeLatency = 1 : i32} : !firrtl.bundle<addr: uint<4>, en: uint<1>, clk: clock, data flip: sint<42>>
    // Read port.
    %6 = firrtl.subfield %_M_read(0) : (!firrtl.bundle<addr: uint<4>, en: uint<1>, clk: clock, data flip: sint<42>>) -> !firrtl.uint<4>
    firrtl.connect %6, %c0_ui1 : !firrtl.uint<4>, !firrtl.uint<1>
    %7 = firrtl.subfield %_M_read(1) : (!firrtl.bundle<addr: uint<4>, en: uint<1>, clk: clock, data flip: sint<42>>) -> !firrtl.uint<1>
    firrtl.connect %7, %c1_ui1 : !firrtl.uint<1>, !firrtl.uint<1>
    %8 = firrtl.subfield %_M_read(2) : (!firrtl.bundle<addr: uint<4>, en: uint<1>, clk: clock, data flip: sint<42>>) -> !firrtl.clock
    firrtl.connect %8, %clock1 : !firrtl.clock, !firrtl.clock
  }

  // CHECK-LABEL: hw.module private @top_modx() -> (tmp27: i23) {
  // CHECK-NEXT:    %c0_i23 = hw.constant 0 : i23
  // CHECK-NEXT:    %c42_i23 = hw.constant 42 : i23
  // CHECK-NEXT:    hw.output %c0_i23 : i23
  // CHECK-NEXT:  }
  firrtl.module private @top_modx(out %tmp27: !firrtl.uint<23>) {
    %0 = firrtl.wire : !firrtl.uint<0>
    %c42_ui23 = firrtl.constant 42 : !firrtl.uint<23>
    %1 = firrtl.tail %c42_ui23, 23 : (!firrtl.uint<23>) -> !firrtl.uint<0>
    firrtl.connect %0, %1 : !firrtl.uint<0>, !firrtl.uint<0>
    %2 = firrtl.head %c42_ui23, 0 : (!firrtl.uint<23>) -> !firrtl.uint<0>
    %3 = firrtl.pad %2, 23 : (!firrtl.uint<0>) -> !firrtl.uint<23>
    firrtl.connect %tmp27, %3 : !firrtl.uint<23>, !firrtl.uint<23>
  }

  // CHECK-LABEL: hw.module private @SimpleStruct(%source: !hw.struct<valid: i1, ready: i1, data: i64>) -> (fldout: i64) {
  // CHECK-NEXT:    %data = hw.struct_extract %source["data"] : !hw.struct<valid: i1, ready: i1, data: i64>
  // CHECK-NEXT:    hw.output %data : i64
  // CHECK-NEXT:  }
  firrtl.module private @SimpleStruct(in %source: !firrtl.bundle<valid: uint<1>, ready: uint<1>, data: uint<64>>,
                              out %fldout: !firrtl.uint<64>) {
    %2 = firrtl.subfield %source (2) : (!firrtl.bundle<valid: uint<1>, ready: uint<1>, data: uint<64>>) -> !firrtl.uint<64>
    firrtl.connect %fldout, %2 : !firrtl.uint<64>, !firrtl.uint<64>
  }

  // CHECK-LABEL: IsInvalidIssue572
  // https://github.com/llvm/circt/issues/572
  firrtl.module private @IsInvalidIssue572(in %a: !firrtl.analog<1>) {
    // CHECK-NEXT: %0 = sv.read_inout %a : !hw.inout<i1>

    // CHECK-NEXT: %.invalid_analog = sv.wire : !hw.inout<i1>
    // CHECK-NEXT: %1 = sv.read_inout %.invalid_analog : !hw.inout<i1>
    %0 = firrtl.invalidvalue : !firrtl.analog<1>

    // CHECK-NEXT: sv.ifdef "SYNTHESIS"  {
    // CHECK-NEXT:   sv.assign %a, %1 : i1
    // CHECK-NEXT:   sv.assign %.invalid_analog, %0 : i1
    // CHECK-NEXT: } else {
    // CHECK-NEXT:   sv.ifdef "verilator" {
    // CHECK-NEXT:     sv.verbatim "`error \22Verilator does not support alias and thus cannot arbitrarily connect bidirectional wires and ports\22"
    // CHECK-NEXT:   } else {
    // CHECK-NEXT:     sv.alias %a, %.invalid_analog : !hw.inout<i1>, !hw.inout<i1>
    // CHECK-NEXT:   }
    // CHECK-NEXT: }
    firrtl.attach %a, %0 : !firrtl.analog<1>, !firrtl.analog<1>
  }

  // CHECK-LABEL: IsInvalidIssue654
  // https://github.com/llvm/circt/issues/654
  firrtl.module private @IsInvalidIssue654() {
    %w = firrtl.wire : !firrtl.uint<0>
    %0 = firrtl.invalidvalue : !firrtl.uint<0>
    firrtl.connect %w, %0 : !firrtl.uint<0>, !firrtl.uint<0>
  }

  // CHECK-LABEL: ASQ
  // https://github.com/llvm/circt/issues/699
  firrtl.module private @ASQ(in %clock: !firrtl.clock, in %reset: !firrtl.asyncreset) {
    %c0_ui1 = firrtl.constant 0 : !firrtl.uint<1>
    %widx_widx_bin = firrtl.regreset %clock, %reset, %c0_ui1 {name = "widx_widx_bin"} : !firrtl.asyncreset, !firrtl.uint<1>, !firrtl.uint<4>
  }

  // CHECK-LABEL: hw.module private @Struct0bits(%source: !hw.struct<valid: i1, ready: i1, data: i0>) {
  // CHECK-NEXT:    hw.output
  // CHECK-NEXT:  }
  firrtl.module private @Struct0bits(in %source: !firrtl.bundle<valid: uint<1>, ready: uint<1>, data: uint<0>>) {
    %2 = firrtl.subfield %source (2) : (!firrtl.bundle<valid: uint<1>, ready: uint<1>, data: uint<0>>) -> !firrtl.uint<0>
  }

  // CHECK-LABEL: hw.module private @MemDepth1
  firrtl.module private @MemDepth1(in %clock: !firrtl.clock, in %en: !firrtl.uint<1>,
                           in %addr: !firrtl.uint<1>, out %data: !firrtl.uint<32>) {
    // CHECK: %mem0_ext.R0_data = hw.instance "mem0_ext" @mem0_combMem(R0_addr: %addr: i1, R0_en: %en: i1, R0_clk: %clock: i1) -> (R0_data: i32)
    // CHECK: hw.output %mem0_ext.R0_data : i32
    %mem0_load0 = firrtl.mem Old {depth = 1 : i64, name = "mem0", portNames = ["load0"], readLatency = 0 : i32, writeLatency = 1 : i32} : !firrtl.bundle<addr: uint<1>, en: uint<1>, clk: clock, data flip: uint<32>>
    %0 = firrtl.subfield %mem0_load0(2) : (!firrtl.bundle<addr: uint<1>, en: uint<1>, clk: clock, data flip: uint<32>>) -> !firrtl.clock
    firrtl.connect %0, %clock : !firrtl.clock, !firrtl.clock
    %1 = firrtl.subfield %mem0_load0(0) : (!firrtl.bundle<addr: uint<1>, en: uint<1>, clk: clock, data flip: uint<32>>) -> !firrtl.uint<1>
    firrtl.connect %1, %addr : !firrtl.uint<1>, !firrtl.uint<1>
    %2 = firrtl.subfield %mem0_load0(3) : (!firrtl.bundle<addr: uint<1>, en: uint<1>, clk: clock, data flip: uint<32>>) -> !firrtl.uint<32>
    firrtl.connect %data, %2 : !firrtl.uint<32>, !firrtl.uint<32>
    %3 = firrtl.subfield %mem0_load0(1) : (!firrtl.bundle<addr: uint<1>, en: uint<1>, clk: clock, data flip: uint<32>>) -> !firrtl.uint<1>
    firrtl.connect %3, %en : !firrtl.uint<1>, !firrtl.uint<1>
}

  // https://github.com/llvm/circt/issues/1115
  // CHECK-LABEL: hw.module private @issue1115
  firrtl.module private @issue1115(in %a: !firrtl.sint<20>, out %tmp59: !firrtl.sint<2>) {
    %0 = firrtl.shr %a, 21 : (!firrtl.sint<20>) -> !firrtl.sint<1>
    firrtl.connect %tmp59, %0 : !firrtl.sint<2>, !firrtl.sint<1>
  }

  // CHECK-LABEL: issue1303
  firrtl.module private @issue1303(out %out: !firrtl.reset) {
    %c1_ui = firrtl.constant 1 : !firrtl.uint<1>
    firrtl.connect %out, %c1_ui : !firrtl.reset, !firrtl.uint<1>
    // CHECK-NEXT: %true = hw.constant true
    // CHECK-NEXT: hw.output %true
  }

  // CHECK-LABEL: hw.module private @Force
  firrtl.module private @Force(in %in: !firrtl.uint<42>) {
    // CHECK: %out = sv.wire
    // CHECK: sv.initial {
    // CHECK:   sv.force %out, %in : i42
    // CHECK: }
    %out = firrtl.wire : !firrtl.uint<42>
    firrtl.force %out, %in : !firrtl.uint<42>, !firrtl.uint<42>
  }

  firrtl.extmodule @chkcoverAnno(in clock: !firrtl.clock) attributes {annotations = [{class = "freechips.rocketchip.annotations.InternalVerifBlackBoxAnnotation"}]}
  // chckcoverAnno is extracted because it is instantiated inside the DUT.
  // CHECK-LABEL: hw.module.extern @chkcoverAnno(%clock: i1)
  // CHECK-SAME: attributes {firrtl.extract.cover.extra}

  firrtl.extmodule @chkcoverAnno2(in clock: !firrtl.clock) attributes {annotations = [{class = "freechips.rocketchip.annotations.InternalVerifBlackBoxAnnotation"}]}
  // checkcoverAnno2 is NOT extracted because it is not instantiated under the
  // DUT.
  // CHECK-LABEL: hw.module.extern @chkcoverAnno2(%clock: i1)
  // CHECK-NOT: attributes {firrtl.extract.cover.extra}

  // CHECK-LABEL: hw.module.extern @InnerNamesExt
  // CHECK-SAME:  (
  // CHECK-SAME:    clockIn: i1 {hw.exportPort = @extClockInSym}
  // CHECK-SAME:  ) -> (
  // CHECK-SAME:    clockOut: i1 {hw.exportPort = @extClockOutSym}
  // CHECK-SAME:  )
  firrtl.extmodule @InnerNamesExt(
    in clockIn: !firrtl.clock sym @extClockInSym,
    out clockOut: !firrtl.clock sym @extClockOutSym
  )
  attributes {annotations = [{class = "freechips.rocketchip.annotations.InternalVerifBlackBoxAnnotation"}]}

  // CHECK-LABEL: hw.module private @FooDUT
  firrtl.module private @FooDUT() attributes {annotations = [
      {class = "sifive.enterprise.firrtl.MarkDUTAnnotation"}]} {
    %chckcoverAnno_clock = firrtl.instance chkcoverAnno @chkcoverAnno(in clock: !firrtl.clock)
  }

  // CHECK-LABEL: hw.module private @MemoryWritePortBehavior
  firrtl.module private @MemoryWritePortBehavior(in %clock1: !firrtl.clock, in %clock2: !firrtl.clock) {
    // This memory has both write ports driven by the same clock.  It should be
    // lowered to an "aa" memory. Even if the clock is passed via different wires,
    // we should identify the clocks to be same.
    //
    // CHECK: hw.instance "aa_ext" @aa_combMem
    %memory_aa_w0, %memory_aa_w1 = firrtl.mem Undefined {depth = 16 : i64, name = "aa", portNames = ["w0", "w1"], readLatency = 1 : i32, writeLatency = 1 : i32} : !firrtl.bundle<addr: uint<4>, en: uint<1>, clk: clock, data: uint<8>, mask: uint<1>>, !firrtl.bundle<addr: uint<4>, en: uint<1>, clk: clock, data: uint<8>, mask: uint<1>>
    %clk_aa_w0 = firrtl.subfield %memory_aa_w0(2) : (!firrtl.bundle<addr: uint<4>, en: uint<1>, clk: clock, data: uint<8>, mask: uint<1>>) -> !firrtl.clock
    %clk_aa_w1 = firrtl.subfield %memory_aa_w1(2) : (!firrtl.bundle<addr: uint<4>, en: uint<1>, clk: clock, data: uint<8>, mask: uint<1>>) -> !firrtl.clock
    %cwire1 = firrtl.wire : !firrtl.clock
    %cwire2 = firrtl.wire : !firrtl.clock
    firrtl.connect %cwire1, %clock1 : !firrtl.clock, !firrtl.clock
    firrtl.connect %cwire2, %clock1 : !firrtl.clock, !firrtl.clock
    firrtl.connect %clk_aa_w0, %cwire1 : !firrtl.clock, !firrtl.clock
    firrtl.connect %clk_aa_w1, %cwire2 : !firrtl.clock, !firrtl.clock

    // This memory has different clocks for each write port.  It should be
    // lowered to an "ab" memory.
    //
    // CHECK: hw.instance "ab_ext" @ab_combMem
    %memory_ab_w0, %memory_ab_w1 = firrtl.mem Undefined {depth = 16 : i64, name = "ab", portNames = ["w0", "w1"], readLatency = 1 : i32, writeLatency = 1 : i32} : !firrtl.bundle<addr: uint<4>, en: uint<1>, clk: clock, data: uint<8>, mask: uint<1>>, !firrtl.bundle<addr: uint<4>, en: uint<1>, clk: clock, data: uint<8>, mask: uint<1>>
    %clk_ab_w0 = firrtl.subfield %memory_ab_w0(2) : (!firrtl.bundle<addr: uint<4>, en: uint<1>, clk: clock, data: uint<8>, mask: uint<1>>) -> !firrtl.clock
    %clk_ab_w1 = firrtl.subfield %memory_ab_w1(2) : (!firrtl.bundle<addr: uint<4>, en: uint<1>, clk: clock, data: uint<8>, mask: uint<1>>) -> !firrtl.clock
    firrtl.connect %clk_ab_w0, %clock1 : !firrtl.clock, !firrtl.clock
    firrtl.connect %clk_ab_w1, %clock2 : !firrtl.clock, !firrtl.clock

    // This memory is the same as the first memory, but a node is used to alias
    // the second write port clock (e.g., this could be due to a dont touch
    // annotation blocking this from being optimized away).  This should be
    // lowered to an "aa" since they are identical.
    //
    // CHECK: hw.instance "ab_node_ext" @aa_combMem
    %memory_ab_node_w0, %memory_ab_node_w1 = firrtl.mem Undefined {depth = 16 : i64, name = "ab_node", portNames = ["w0", "w1"], readLatency = 1 : i32, writeLatency = 1 : i32} : !firrtl.bundle<addr: uint<4>, en: uint<1>, clk: clock, data: uint<8>, mask: uint<1>>, !firrtl.bundle<addr: uint<4>, en: uint<1>, clk: clock, data: uint<8>, mask: uint<1>>
    %clk_ab_node_w0 = firrtl.subfield %memory_ab_node_w0(2) : (!firrtl.bundle<addr: uint<4>, en: uint<1>, clk: clock, data: uint<8>, mask: uint<1>>) -> !firrtl.clock
    %clk_ab_node_w1 = firrtl.subfield %memory_ab_node_w1(2) : (!firrtl.bundle<addr: uint<4>, en: uint<1>, clk: clock, data: uint<8>, mask: uint<1>>) -> !firrtl.clock
    firrtl.connect %clk_ab_node_w0, %clock1 : !firrtl.clock, !firrtl.clock
    %tmp = firrtl.node %clock1 : !firrtl.clock
    firrtl.connect %clk_ab_node_w1, %tmp : !firrtl.clock, !firrtl.clock
  }

  // CHECK-LABEL: hw.module private @AsyncResetBasic(
  firrtl.module private @AsyncResetBasic(in %clock: !firrtl.clock, in %arst: !firrtl.asyncreset, in %srst: !firrtl.uint<1>) {
    %c9_ui42 = firrtl.constant 9 : !firrtl.uint<42>
    %c-9_si42 = firrtl.constant -9 : !firrtl.sint<42>
    // The following should not error because the reset values are constant.
    %r0 = firrtl.regreset %clock, %arst, %c9_ui42 : !firrtl.asyncreset, !firrtl.uint<42>, !firrtl.uint<42>
    %r1 = firrtl.regreset %clock, %srst, %c9_ui42 : !firrtl.uint<1>, !firrtl.uint<42>, !firrtl.uint<42>
    %r2 = firrtl.regreset %clock, %arst, %c-9_si42 : !firrtl.asyncreset, !firrtl.sint<42>, !firrtl.sint<42>
    %r3 = firrtl.regreset %clock, %srst, %c-9_si42 : !firrtl.uint<1>, !firrtl.sint<42>, !firrtl.sint<42>
  }

  // CHECK-LABEL: hw.module private @BitCast1
  firrtl.module private @BitCast1() {
    %a = firrtl.wire : !firrtl.vector<uint<2>, 13>
    %b = firrtl.bitcast %a : (!firrtl.vector<uint<2>, 13>) -> (!firrtl.uint<26>)
    // CHECK: hw.bitcast %0 : (!hw.array<13xi2>) -> i26
  }

  // CHECK-LABEL: hw.module private @BitCast2
  firrtl.module private @BitCast2() {
    %a = firrtl.wire : !firrtl.bundle<valid: uint<1>, ready: uint<1>, data: uint<1>>
    %b = firrtl.bitcast %a : (!firrtl.bundle<valid: uint<1>, ready: uint<1>, data: uint<1>>) -> (!firrtl.uint<3>)
    // CHECK: hw.bitcast %0 : (!hw.struct<valid: i1, ready: i1, data: i1>) -> i3

  }
  // CHECK-LABEL: hw.module private @BitCast3
  firrtl.module private @BitCast3() {
    %a = firrtl.wire : !firrtl.uint<26>
    %b = firrtl.bitcast %a : (!firrtl.uint<26>) -> (!firrtl.vector<uint<2>, 13>)
    // CHECK: hw.bitcast %0 : (i26) -> !hw.array<13xi2>
  }

  // CHECK-LABEL: hw.module private @BitCast4
  firrtl.module private @BitCast4() {
    %a = firrtl.wire : !firrtl.uint<3>
    %b = firrtl.bitcast %a : (!firrtl.uint<3>) -> (!firrtl.bundle<valid: uint<1>, ready: uint<1>, data: uint<1>>)
    // CHECK: hw.bitcast %0 : (i3) -> !hw.struct<valid: i1, ready: i1, data: i1>

  }
  // CHECK-LABEL: hw.module private @BitCast5
  firrtl.module private @BitCast5() {
    %a = firrtl.wire : !firrtl.bundle<valid: uint<2>, ready: uint<1>, data: uint<3>>
    %b = firrtl.bitcast %a : (!firrtl.bundle<valid: uint<2>, ready: uint<1>, data: uint<3>>) -> (!firrtl.vector<uint<2>, 3>)
    // CHECK: hw.bitcast %0 : (!hw.struct<valid: i2, ready: i1, data: i3>) -> !hw.array<3xi2>
  }

  // CHECK-LABEL: hw.module private @InnerNames
  // CHECK-SAME:  (
  // CHECK-SAME:    %value: i42 {hw.exportPort = @portValueSym}
  // CHECK-SAME:    %clock: i1 {hw.exportPort = @portClockSym}
  // CHECK-SAME:    %reset: i1 {hw.exportPort = @portResetSym}
  // CHECK-SAME:  ) -> (
  // CHECK-SAME:    out: i1 {hw.exportPort = @portOutSym}
  // CHECK-SAME:  )
  firrtl.module private @InnerNames(
    in %value: !firrtl.uint<42> sym @portValueSym,
    in %clock: !firrtl.clock sym @portClockSym,
    in %reset: !firrtl.uint<1> sym @portResetSym,
    out %out: !firrtl.uint<1> sym @portOutSym
  ) {
    firrtl.instance instName sym @instSym @BitCast1()
    // CHECK: hw.instance "instName" sym @instSym @BitCast1
    %nodeName = firrtl.node sym @nodeSym %value : !firrtl.uint<42>
    // CHECK: [[WIRE:%nodeName]] = sv.wire sym @nodeSym : !hw.inout<i42>
    // CHECK-NEXT: sv.assign [[WIRE]], %value
    %wireName = firrtl.wire sym @wireSym : !firrtl.uint<42>
    // CHECK: %wireName = sv.wire sym @wireSym : !hw.inout<i42>
    %regName = firrtl.reg sym @regSym %clock : !firrtl.uint<42>
    // CHECK: %regName = seq.firreg %regName clock %clock sym @regSym : i42
    %regResetName = firrtl.regreset sym @regResetSym %clock, %reset, %value : !firrtl.uint<1>, !firrtl.uint<42>, !firrtl.uint<42>
    // CHECK: %regResetName = seq.firreg %regResetName clock %clock sym @regResetSym reset sync %reset, %value : i42
    %memName_port = firrtl.mem sym @memSym Undefined {depth = 12 : i64, name = "memName", portNames = ["port"], readLatency = 0 : i32, writeLatency = 1 : i32} : !firrtl.bundle<addr: uint<4>, en: uint<1>, clk: clock, data flip: uint<42>>
    // CHECK: {{%.+}} = hw.instance "memName_ext" sym @memSym
    firrtl.connect %out, %reset : !firrtl.uint<1>, !firrtl.uint<1>
  }

  // CHECK-LABEL: hw.module private @connectNarrowUIntVector
  firrtl.module private @connectNarrowUIntVector(in %clock: !firrtl.clock, in %a: !firrtl.vector<uint<1>, 1>, out %b: !firrtl.vector<uint<3>, 1>) {
    %r1 = firrtl.reg %clock  : !firrtl.vector<uint<2>, 1>
    firrtl.connect %r1, %a : !firrtl.vector<uint<2>, 1>, !firrtl.vector<uint<1>, 1>
    firrtl.connect %b, %r1 : !firrtl.vector<uint<3>, 1>, !firrtl.vector<uint<2>, 1>
    // CHECK:      %r1 = seq.firreg %3 clock %clock : !hw.array<1xi2>
    // CHECK-NEXT: %1 = hw.array_get %a[%false] : !hw.array<1xi1>
    // CHECK-NEXT: %2 = comb.concat %false, %1 : i1, i1
    // CHECK-NEXT: %3 = hw.array_create %2 : i2
    // CHECK-NEXT: %4 = hw.array_get %r1[%false] : !hw.array<1xi2>
    // CHECK-NEXT: %5 = comb.concat %false, %4 : i1, i2
    // CHECK-NEXT: %6 = hw.array_create %5 : i3
    // CHECK-NEXT: sv.assign %.b.output, %6 : !hw.array<1xi3>
    // CHECK-NEXT: hw.output %0 : !hw.array<1xi3>
  }

  // CHECK-LABEL: hw.module private @connectNarrowSIntVector
  firrtl.module private @connectNarrowSIntVector(in %clock: !firrtl.clock, in %a: !firrtl.vector<sint<1>, 1>, out %b: !firrtl.vector<sint<3>, 1>) {
    %r1 = firrtl.reg %clock  : !firrtl.vector<sint<2>, 1>
    firrtl.connect %r1, %a : !firrtl.vector<sint<2>, 1>, !firrtl.vector<sint<1>, 1>
    firrtl.connect %b, %r1 : !firrtl.vector<sint<3>, 1>, !firrtl.vector<sint<2>, 1>
    // CHECK:      %r1 = seq.firreg %3 clock %clock : !hw.array<1xi2>
    // CHECK-NEXT: %1 = hw.array_get %a[%false] : !hw.array<1xi1>
    // CHECK-NEXT: %2 = comb.concat %1, %1 : i1, i1
    // CHECK-NEXT: %3 = hw.array_create %2 : i2
    // CHECK-NEXT: %4 = hw.array_get %r1[%false] : !hw.array<1xi2>
    // CHECK-NEXT: %5 = comb.extract %4 from 1 : (i2) -> i1
    // CHECK-NEXT: %6 = comb.concat %5, %4 : i1, i2
    // CHECK-NEXT: %7 = hw.array_create %6 : i3
    // CHECK-NEXT: sv.assign %.b.output, %7 : !hw.array<1xi3>
    // CHECK-NEXT: hw.output %0 : !hw.array<1xi3>
  }

  // CHECK-LABEL: hw.module private @SubIndex
  firrtl.module private @SubIndex(in %a: !firrtl.vector<vector<uint<1>, 1>, 1>, in %clock: !firrtl.clock, out %o1: !firrtl.uint<1>, out %o2: !firrtl.vector<uint<1>, 1>) {
    %r1 = firrtl.reg %clock  : !firrtl.uint<1>
    %r2 = firrtl.reg %clock  : !firrtl.vector<uint<1>, 1>
    %0 = firrtl.subindex %a[0] : !firrtl.vector<vector<uint<1>, 1>, 1>
    %1 = firrtl.subindex %0[0] : !firrtl.vector<uint<1>, 1>
    firrtl.connect %r1, %1 : !firrtl.uint<1>, !firrtl.uint<1>
    firrtl.connect %r2, %0 : !firrtl.vector<uint<1>, 1>, !firrtl.vector<uint<1>, 1>
    firrtl.connect %o1, %r1 : !firrtl.uint<1>, !firrtl.uint<1>
    firrtl.connect %o2, %r2 : !firrtl.vector<uint<1>, 1>, !firrtl.vector<uint<1>, 1>
    // CHECK:      %r1 = seq.firreg %1 clock %clock : i1
    // CHECK-NEXT: %r2 = seq.firreg %0 clock %clock : !hw.array<1xi1>
    // CHECK-NEXT: %0 = hw.array_get %a[%false] : !hw.array<1xarray<1xi1>>
    // CHECK-NEXT: %1 = hw.array_get %0[%false] : !hw.array<1xi1>
    // CHECK-NEXT: hw.output %r1, %r2 : i1, !hw.array<1xi1>
  }

  // CHECK-LABEL: hw.module private @SubAccess
  firrtl.module private @SubAccess(in %x: !firrtl.uint<1>, in %y: !firrtl.uint<1>, in %a: !firrtl.vector<vector<uint<1>, 1>, 1>, in %clock: !firrtl.clock, out %o1: !firrtl.uint<1>, out %o2: !firrtl.vector<uint<1>, 1>) {
    %r1 = firrtl.reg %clock  : !firrtl.uint<1>
    %r2 = firrtl.reg %clock  : !firrtl.vector<uint<1>, 1>
    %0 = firrtl.subaccess %a[%x] : !firrtl.vector<vector<uint<1>, 1>, 1>, !firrtl.uint<1>
    %1 = firrtl.subaccess %0[%y] : !firrtl.vector<uint<1>, 1>, !firrtl.uint<1>
    firrtl.connect %r1, %1 : !firrtl.uint<1>, !firrtl.uint<1>
    firrtl.connect %r2, %0 : !firrtl.vector<uint<1>, 1>, !firrtl.vector<uint<1>, 1>
    firrtl.connect %o1, %r1 : !firrtl.uint<1>, !firrtl.uint<1>
    firrtl.connect %o2, %r2 : !firrtl.vector<uint<1>, 1>, !firrtl.vector<uint<1>, 1>
    // CHECK:      %r1 = seq.firreg %1 clock %clock : i1
    // CHECK-NEXT: %r2 = seq.firreg %0 clock %clock : !hw.array<1xi1>
    // CHECK-NEXT: %0 = hw.array_get %a[%x] : !hw.array<1xarray<1xi1>>, i1
    // CHECK-NEXT: %1 = hw.array_get %0[%y] : !hw.array<1xi1>, i1
    // CHECK-NEXT: hw.output %r1, %r2 : i1, !hw.array<1xi1>
  }

  // CHECK-LABEL: hw.module private @SubindexDestination
  firrtl.module private @SubindexDestination(in %clock: !firrtl.clock, in %a: !firrtl.vector<uint<1>, 3>, out %b: !firrtl.vector<uint<1>, 3>) {
    %0 = firrtl.subindex %b[2] : !firrtl.vector<uint<1>, 3>
    %1 = firrtl.subindex %a[2] : !firrtl.vector<uint<1>, 3>
    firrtl.connect %0, %1 : !firrtl.uint<1>, !firrtl.uint<1>
    // CHECK:      %c-2_i2 = hw.constant -2 : i2
    // CHECK-NEXT: %.b.output = sv.wire
    // CHECK-NEXT: %0 = sv.read_inout %.b.output : !hw.inout<array<3xi1>>
    // CHECK-NEXT: %1 = sv.array_index_inout %.b.output[%c-2_i2] : !hw.inout<array<3xi1>>, i2
    // CHECK-NEXT: %2 = hw.array_get %a[%c-2_i2] : !hw.array<3xi1>
    // CHECK-NEXT: sv.assign %1, %2 : i1
    // CHECK-NEXT: hw.output %0 : !hw.array<3xi1>
  }

  // CHECK-LABEL: hw.module private @SubAccessDestination
  firrtl.module private @SubAccessDestination(in %x: !firrtl.uint<2>, in %y: !firrtl.uint<2>, in %clock: !firrtl.clock, in %a: !firrtl.vector<uint<1>, 5>, out %b: !firrtl.vector<uint<1>, 5>) {
    %0 = firrtl.subaccess %b[%x] : !firrtl.vector<uint<1>, 5>, !firrtl.uint<2>
    %1 = firrtl.subaccess %a[%y] : !firrtl.vector<uint<1>, 5>, !firrtl.uint<2>
    firrtl.connect %0, %1 : !firrtl.uint<1>, !firrtl.uint<1>
    // CHECK:      %[[EXTIndex:.+]] = hw.constant 0 : i3 
    // CHECK:      %.b.output = sv.wire  : !hw.inout<array<5xi1>>
    // CHECK-NEXT: %0 = sv.read_inout %.b.output
    // CHECK-NEXT: %[[indexExt:.+]] = comb.concat %false, %x : i1, i2
    // CHECK-NEXT: %2 = sv.array_index_inout %.b.output[%[[indexExt]]]
    // CHECK-NEXT: %3 = comb.concat %false, %y : i1, i2
    // CHECK-NEXT: %[[EXTValue:.+]] = hw.array_get %a[%[[EXTIndex]]] 
    // CHECK-NEXT: %[[EXTArray:.+]] = hw.array_create %[[EXTValue]], %[[EXTValue]], %[[EXTValue]] 
    // CHECK-NEXT: %[[Array:.+]] = hw.array_concat %[[EXTArray]], %a 
    // CHECK-NEXT: %[[READ:.+]] = hw.array_get %[[Array]][%3]
    // CHECK-NEXT: %[[valWire:.+]] = sv.wire  : !hw.inout<i1>
    // CHECK-NEXT: sv.assign %[[valWire]], %[[READ]]
    // CHECK-NEXT: %[[RD:.+]] = sv.read_inout %[[valWire]] : !hw.inout<i1> 
    // CHECK-NEXT: sv.assign %2, %[[RD]] : i1 
    // CHECK-NEXT: hw.output %0 : !hw.array<5xi1>
  }

  // CHECK-LABEL: hw.module private @zero_width_constant()
  // https://github.com/llvm/circt/issues/2269
  firrtl.module private @zero_width_constant(out %a: !firrtl.uint<0>) {
    // CHECK-NEXT: hw.output
    %c0_ui0 = firrtl.constant 0 : !firrtl.uint<0>
    firrtl.connect %a, %c0_ui0 : !firrtl.uint<0>, !firrtl.uint<0>
  }

  // CHECK-LABEL: @subfield_write1(
  firrtl.module private @subfield_write1(out %a: !firrtl.bundle<a: uint<1>>) {
    %0 = firrtl.subfield %a(0) : (!firrtl.bundle<a: uint<1>>) -> !firrtl.uint<1>
    %c0_ui1 = firrtl.constant 1 : !firrtl.uint<1>
    firrtl.connect %0, %c0_ui1 : !firrtl.uint<1>, !firrtl.uint<1>
    // CHECK:      %true = hw.constant true
    // CHECK-NEXT: %.a.output = sv.wire
    // CHECK-NEXT: %0 = sv.read_inout %.a.output : !hw.inout<struct<a: i1>>
    // CHECK-NEXT: %1 = sv.struct_field_inout %.a.output["a"] : !hw.inout<struct<a: i1>>
    // CHECK-NEXT: sv.assign %1, %true : i1
    // CHECK-NEXT: hw.output %0 : !hw.struct<a: i1>
  }

  // CHECK-LABEL: @subfield_write2(
  firrtl.module private @subfield_write2(in %in: !firrtl.uint<1>, out %sink: !firrtl.bundle<a: bundle<b: bundle<c: uint<1>>>>) {
    %0 = firrtl.subfield %sink(0) : (!firrtl.bundle<a: bundle<b: bundle<c: uint<1>>>>) -> !firrtl.bundle<b: bundle<c: uint<1>>>
    %1 = firrtl.subfield %0(0) : (!firrtl.bundle<b: bundle<c: uint<1>>>) -> !firrtl.bundle<c: uint<1>>
    %2 = firrtl.subfield %1(0) : (!firrtl.bundle<c: uint<1>>) -> !firrtl.uint<1>
    firrtl.connect %2, %in : !firrtl.uint<1>, !firrtl.uint<1>
    // CHECK:      %.sink.output = sv.wire
    // CHECK-NEXT: %0 = sv.read_inout %.sink.output : !hw.inout<struct<a: !hw.struct<b: !hw.struct<c: i1>>>>
    // CHECK-NEXT: %1 = sv.struct_field_inout %.sink.output["a"] : !hw.inout<struct<a: !hw.struct<b: !hw.struct<c: i1>>>>
    // CHECK-NEXT: %2 = sv.struct_field_inout %1["b"] : !hw.inout<struct<b: !hw.struct<c: i1>>>
    // CHECK-NEXT: %3 = sv.struct_field_inout %2["c"] : !hw.inout<struct<c: i1>>
    // CHECK-NEXT: sv.assign %3, %in : i1
    // CHECK-NEXT: hw.output %0 : !hw.struct<a: !hw.struct<b: !hw.struct<c: i1>>>
  }

  // CHECK-LABEL: hw.module private @RegResetStructWiden
  firrtl.module private @RegResetStructWiden(in %clock: !firrtl.clock, in %reset: !firrtl.uint<1>, in %init: !firrtl.bundle<a: uint<2>>) {
    // CHECK:      [[FALSE:%.*]] = hw.constant false
    // CHECK-NEXT: [[A:%.*]] = hw.struct_extract %init["a"] : !hw.struct<a: i2>
    // CHECK-NEXT: [[PADDED:%.*]] = comb.concat [[FALSE]], [[A]] : i1, i2
    // CHECK-NEXT: [[STRUCT:%.*]] = hw.struct_create ([[PADDED]]) : !hw.struct<a: i3>
    // CHECK-NEXT: %reg = seq.firreg %reg clock %clock reset sync %reset, [[STRUCT]] : !hw.struct<a: i3>
    %reg = firrtl.regreset %clock, %reset, %init  : !firrtl.uint<1>, !firrtl.bundle<a: uint<2>>, !firrtl.bundle<a: uint<3>>
  }

  // CHECK-LABEL: hw.module private @BundleConnection
  firrtl.module private @BundleConnection(in %source: !firrtl.bundle<a: bundle<b: uint<1>>>, out %sink: !firrtl.bundle<a: bundle<b: uint<1>>>) {
    %0 = firrtl.subfield %sink(0) : (!firrtl.bundle<a: bundle<b: uint<1>>>) -> !firrtl.bundle<b: uint<1>>
    %1 = firrtl.subfield %source(0) : (!firrtl.bundle<a: bundle<b: uint<1>>>) -> !firrtl.bundle<b: uint<1>>
    firrtl.connect %0, %1 : !firrtl.bundle<b: uint<1>>, !firrtl.bundle<b: uint<1>>
    // CHECK:      %.sink.output = sv.wire
    // CHECK-NEXT: %0 = sv.read_inout %.sink.output : !hw.inout<struct<a: !hw.struct<b: i1>>>
    // CHECK-NEXT: %1 = sv.struct_field_inout %.sink.output["a"] : !hw.inout<struct<a: !hw.struct<b: i1>>>
    // CHECK-NEXT: %a = hw.struct_extract %source["a"] : !hw.struct<a: !hw.struct<b: i1>>
    // CHECK-NEXT: sv.assign %1, %a : !hw.struct<b: i1>
    // CHECK-NEXT: hw.output %0 : !hw.struct<a: !hw.struct<b: i1>>
  }

  // CHECK-LABEL: hw.module private @AggregateInvalidValue
  firrtl.module private @AggregateInvalidValue(in %clock: !firrtl.clock, in %reset: !firrtl.uint<1>) {
    %invalid = firrtl.invalidvalue : !firrtl.bundle<a: uint<1>, b: vector<uint<10>, 10>>
    %reg = firrtl.regreset %clock, %reset, %invalid : !firrtl.uint<1>, !firrtl.bundle<a: uint<1>, b: vector<uint<10>, 10>>, !firrtl.bundle<a: uint<1>, b: vector<uint<10>, 10>>
    // CHECK:      %c0_i101 = hw.constant 0 : i101
    // CHECK-NEXT: %0 = hw.bitcast %c0_i101 : (i101) -> !hw.struct<a: i1, b: !hw.array<10xi10>>
    // CHECK-NEXT: %reg = seq.firreg %reg clock %clock reset sync %reset, %0 : !hw.struct<a: i1, b: !hw.array<10xi10>>
  }

  // CHECK-LABEL: hw.module private @AggregateRegAssign
  firrtl.module private @AggregateRegAssign(in %clock: !firrtl.clock, in %value: !firrtl.uint<1>) {
    %reg = firrtl.reg %clock : !firrtl.vector<uint<1>, 1>
    %reg_0 = firrtl.subindex %reg[0] : !firrtl.vector<uint<1>, 1>
    firrtl.connect %reg_0, %value : !firrtl.uint<1>, !firrtl.uint<1>
    // CHECK: %reg = seq.firreg [[INPUT:%.+]] clock %clock : !hw.array<1xi1>
    // CHECK: [[INPUT]] = hw.array_create %value : i1
  }

  // CHECK-LABEL: hw.module private @AggregateRegResetAssign
  firrtl.module private @AggregateRegResetAssign(in %clock: !firrtl.clock, in %reset: !firrtl.uint<1>,
                                         in %init: !firrtl.vector<uint<1>, 1>, in %value: !firrtl.uint<1>) {
    %reg = firrtl.regreset %clock, %reset, %init  : !firrtl.uint<1>, !firrtl.vector<uint<1>, 1>, !firrtl.vector<uint<1>, 1>
    %reg_0 = firrtl.subindex %reg[0] : !firrtl.vector<uint<1>, 1>
    firrtl.connect %reg_0, %value : !firrtl.uint<1>, !firrtl.uint<1>
    // CHECK: %reg = seq.firreg [[INPUT:%.+]] clock %clock reset sync %reset, %init : !hw.array<1xi1>
    // CHECK: [[INPUT]] = hw.array_create %value : i1
  }

  // CHECK-LABEL: hw.module private @ForceNameSubmodule
  firrtl.hierpath private @nla_1 [@ForceNameTop::@sym_foo, @ForceNameSubmodule]
  firrtl.hierpath private @nla_2 [@ForceNameTop::@sym_bar, @ForceNameSubmodule]
  firrtl.module private @ForceNameSubmodule() attributes {annotations = [
    {circt.nonlocal = @nla_2,
     class = "chisel3.util.experimental.ForceNameAnnotation", name = "Bar"},
    {circt.nonlocal = @nla_1,
     class = "chisel3.util.experimental.ForceNameAnnotation", name = "Foo"}]} {}
  // CHECK: hw.module private @ForceNameTop
  firrtl.module private @ForceNameTop() {
    firrtl.instance foo sym @sym_foo
      {annotations = [{circt.nonlocal = @nla_1, class = "circt.nonlocal"}]}
      @ForceNameSubmodule()
    firrtl.instance bar sym @sym_bar
      {annotations = [{circt.nonlocal = @nla_2, class = "circt.nonlocal"}]}
      @ForceNameSubmodule()
    // CHECK:      hw.instance "foo" sym @sym_foo {{.+}} {hw.verilogName = "Foo"}
    // CHECK-NEXT: hw.instance "bar" sym @sym_bar {{.+}} {hw.verilogName = "Bar"}
  }

  // CHECK-LABEL: hw.module private @PreserveName
  firrtl.module private @PreserveName(in %a : !firrtl.uint<1>, in %b : !firrtl.uint<1>, out %c : !firrtl.uint<1>) {
    //CHECK comb.or %a, %b {sv.namehint = "myname"}
    %foo = firrtl.or %a, %b {name = "myname"} : (!firrtl.uint<1>, !firrtl.uint<1>) -> !firrtl.uint<1>
    firrtl.connect %c, %foo : !firrtl.uint<1>, !firrtl.uint<1>

    // CHECK: comb.shl bin {{.*}} {sv.namehint = "anothername"}
    %bar = firrtl.dshl %a, %b {name = "anothername"} : (!firrtl.uint<1>, !firrtl.uint<1>) -> !firrtl.uint<2>
  }

  // CHECK-LABEL: hw.module private @MultibitMux(%source_0: i1, %source_1: i1, %source_2: i1, %index: i2) -> (sink: i1) {
  firrtl.module private @MultibitMux(in %source_0: !firrtl.uint<1>, in %source_1: !firrtl.uint<1>, in %source_2: !firrtl.uint<1>, out %sink: !firrtl.uint<1>, in %index: !firrtl.uint<2>) {
    %0 = firrtl.multibit_mux %index, %source_2, %source_1, %source_0 : !firrtl.uint<2>, !firrtl.uint<1>
    firrtl.connect %sink, %0 : !firrtl.uint<1>, !firrtl.uint<1>
    // CHECK:      %c0_i2 = hw.constant 0 : i2
    // CHECK:      %0 = hw.array_create %source_2, %source_1, %source_0 : i1
    // CHECK-NEXT: %1 = hw.array_get %0[%c0_i2]
    // CHECK-NEXT: %2 = hw.array_create %1 : i1
    // CHECK-NEXT: %3 = hw.array_concat %2, %0
    // CHECK-NEXT: %4 = hw.array_get %3[%index] {sv.attributes = #sv.attributes<[#sv.attribute<"cadence map_to_mux">], emitAsComments>}
    // CHECK-NEXT: %5 = sv.wire : !hw.inout<i1>
    // CHECK-NEXT: sv.assign %5, %4 {sv.attributes = #sv.attributes<[#sv.attribute<"synopsys infer_mux_override">], emitAsComments>}
    // CHECK-NEXT: %6 = sv.read_inout %5 : !hw.inout<i1>
    // CHECK-NEXT: hw.output %6 : i1
  }

  firrtl.module private @inferUnmaskedMemory(in %clock: !firrtl.clock, in %rAddr: !firrtl.uint<4>, in %rEn: !firrtl.uint<1>, out %rData: !firrtl.uint<8>, in %wMask: !firrtl.uint<1>, in %wData: !firrtl.uint<8>) {
    %tbMemoryKind1_r, %tbMemoryKind1_w = firrtl.mem Undefined  {depth = 16 : i64, modName = "tbMemoryKind1_ext", name = "tbMemoryKind1", portNames = ["r", "w"], readLatency = 1 : i32, writeLatency = 1 : i32} : !firrtl.bundle<addr: uint<4>, en: uint<1>, clk: clock, data flip: uint<8>>, !firrtl.bundle<addr: uint<4>, en: uint<1>, clk: clock, data: uint<8>, mask: uint<1>>
    %0 = firrtl.subfield %tbMemoryKind1_w(3) : (!firrtl.bundle<addr: uint<4>, en: uint<1>, clk: clock, data: uint<8>, mask: uint<1>>) -> !firrtl.uint<8>
    %1 = firrtl.subfield %tbMemoryKind1_w(4) : (!firrtl.bundle<addr: uint<4>, en: uint<1>, clk: clock, data: uint<8>, mask: uint<1>>) -> !firrtl.uint<1>
    %2 = firrtl.subfield %tbMemoryKind1_w(0) : (!firrtl.bundle<addr: uint<4>, en: uint<1>, clk: clock, data: uint<8>, mask: uint<1>>) -> !firrtl.uint<4>
    %3 = firrtl.subfield %tbMemoryKind1_w(1) : (!firrtl.bundle<addr: uint<4>, en: uint<1>, clk: clock, data: uint<8>, mask: uint<1>>) -> !firrtl.uint<1>
    %4 = firrtl.subfield %tbMemoryKind1_w(2) : (!firrtl.bundle<addr: uint<4>, en: uint<1>, clk: clock, data: uint<8>, mask: uint<1>>) -> !firrtl.clock
    %5 = firrtl.subfield %tbMemoryKind1_r(3) : (!firrtl.bundle<addr: uint<4>, en: uint<1>, clk: clock, data flip: uint<8>>) -> !firrtl.uint<8>
    %6 = firrtl.subfield %tbMemoryKind1_r(0) : (!firrtl.bundle<addr: uint<4>, en: uint<1>, clk: clock, data flip: uint<8>>) -> !firrtl.uint<4>
    %7 = firrtl.subfield %tbMemoryKind1_r(1) : (!firrtl.bundle<addr: uint<4>, en: uint<1>, clk: clock, data flip: uint<8>>) -> !firrtl.uint<1>
    %8 = firrtl.subfield %tbMemoryKind1_r(2) : (!firrtl.bundle<addr: uint<4>, en: uint<1>, clk: clock, data flip: uint<8>>) -> !firrtl.clock
    firrtl.connect %8, %clock : !firrtl.clock, !firrtl.clock
    firrtl.connect %7, %rEn : !firrtl.uint<1>, !firrtl.uint<1>
    firrtl.connect %6, %rAddr : !firrtl.uint<4>, !firrtl.uint<4>
    firrtl.connect %rData, %5 : !firrtl.uint<8>, !firrtl.uint<8>
    firrtl.connect %4, %clock : !firrtl.clock, !firrtl.clock
    firrtl.connect %3, %rEn : !firrtl.uint<1>, !firrtl.uint<1>
    firrtl.connect %2, %rAddr : !firrtl.uint<4>, !firrtl.uint<4>
    firrtl.connect %1, %wMask : !firrtl.uint<1>, !firrtl.uint<1>
    firrtl.connect %0, %wData : !firrtl.uint<8>, !firrtl.uint<8>
  }
  // CHECK-LABEL: hw.module private @inferUnmaskedMemory
  // CHECK-NEXT:   %[[v0:.+]] = comb.and bin %rEn, %wMask : i1
  // CHECK-NEXT:   %tbMemoryKind1_ext.R0_data = hw.instance "tbMemoryKind1_ext" @tbMemoryKind1_combMem(R0_addr: %rAddr: i4, R0_en: %rEn: i1, R0_clk: %clock: i1, W0_addr: %rAddr: i4, W0_en: %[[v0]]: i1, W0_clk: %clock: i1, W0_data: %wData: i8) -> (R0_data: i8)
  // CHECK-NEXT:   hw.output %tbMemoryKind1_ext.R0_data : i8

  // CHECK-LABEL: hw.module private @eliminateSingleOutputConnects
  // CHECK-NOT:     [[WIRE:%.+]] = sv.wire
  // CHECK-NEXT:    hw.output %a : i1
  firrtl.module private @eliminateSingleOutputConnects(in %a: !firrtl.uint<1>, out %b: !firrtl.uint<1>) {
    firrtl.strictconnect %b, %a : !firrtl.uint<1>
  }

  // Check that modules with comments are lowered.
  // CHECK-LABEL: hw.module private @Commented() attributes {
  // CHECK-SAME:      comment = "this module is commented"
  // CHECK-SAME:  }
  firrtl.module private @Commented() attributes {
      comment = "this module is commented"
  } {}

  // CHECK-LABEL: hw.module @preLoweredOps
  firrtl.module @preLoweredOps() {
    // CHECK-NEXT: %0 = builtin.unrealized_conversion_cast to f32
    // CHECK-NEXT: %1 = arith.addf %0, %0 : f32
    // CHECK-NEXT: builtin.unrealized_conversion_cast %1 : f32 to index
    %0 = builtin.unrealized_conversion_cast to f32
    %1 = arith.addf %0, %0 : f32
    builtin.unrealized_conversion_cast %1 : f32 to index
  }

  // Used for testing.
  firrtl.extmodule private @Blackbox(in inst: !firrtl.uint<1>)

  // Check that the following doesn't crash, when we have a no-op cast which
  // uses an input port.
  // CHECK-LABEL: hw.module private @BackedgesAndNoopCasts
  // CHECK-NEXT:    hw.instance "blackbox" @Blackbox(inst: %clock: i1) -> ()
  // CHECK-NEXT:    hw.output %clock : i1
  firrtl.module private @BackedgesAndNoopCasts(in %clock: !firrtl.uint<1>, out %out : !firrtl.clock) {
    // Following comments describe why this used to crash.
    // Blackbox input port creates a backedge.
    %inst = firrtl.instance blackbox @Blackbox(in inst: !firrtl.uint<1>)
    // No-op cast is removed, %cast lowered to point directly to the backedge.
    %cast = firrtl.asClock %inst : (!firrtl.uint<1>) -> !firrtl.clock
    // Finalize the backedge, replacing all uses with %clock.
    firrtl.strictconnect %inst, %clock : !firrtl.uint<1>
    // %cast accidentally still points to the back edge in the lowering table.
    firrtl.strictconnect %out, %cast : !firrtl.clock
  }

  // Check that when inputs are connected to other inputs, the backedges are
  // properly resolved to the final real value.
  // CHECK-LABEL: hw.module @ChainedBackedges
  // CHECK-NEXT:    hw.instance "a" @Blackbox
  // CHECK-NEXT:    hw.instance "b" @Blackbox
  // CHECK-NEXT:    hw.output %in : i1
  firrtl.module @ChainedBackedges(in %in: !firrtl.uint<1>, out %out: !firrtl.uint<1>) {
    %a_inst = firrtl.instance a @Blackbox(in inst: !firrtl.uint<1>)
    %b_inst = firrtl.instance b @Blackbox(in inst: !firrtl.uint<1>)
    firrtl.strictconnect %a_inst, %in : !firrtl.uint<1>
    firrtl.strictconnect %b_inst, %a_inst : !firrtl.uint<1>
    firrtl.strictconnect %out, %b_inst : !firrtl.uint<1>
  }

  // Check that combinational cycles with no outside driver are lowered to
  // be driven from a wire.
  // CHECK-LABEL: hw.module @UndrivenInputPort()
  // CHECK-NEXT:    %undriven = sv.wire : !hw.inout<i1>
  // CHECK-NEXT:    %0 = sv.read_inout %undriven : !hw.inout<i1>
  // CHECK-NEXT:    hw.instance "blackbox" @Blackbox(inst: %0: i1) -> ()
  // CHECK-NEXT:    hw.instance "blackbox" @Blackbox(inst: %0: i1) -> ()
  firrtl.module @UndrivenInputPort() {
    %0 = firrtl.instance blackbox @Blackbox(in inst : !firrtl.uint<1>)
    %1 = firrtl.instance blackbox @Blackbox(in inst : !firrtl.uint<1>)
    firrtl.strictconnect %0, %1 : !firrtl.uint<1>
    firrtl.strictconnect %1, %0 : !firrtl.uint<1>
  }

  // CHECK-LABEL: hw.module @LowerToFirReg(%clock: i1, %reset: i1, %value: i2) -> (result: i2)
  firrtl.module @LowerToFirReg(in %clock: !firrtl.clock,
                     in %reset: !firrtl.uint<1>,
                     in %value: !firrtl.uint<2>,
                     out %result: !firrtl.uint<2>) {
    %count = firrtl.reg %clock: !firrtl.uint<2>
    // CHECK: %count = seq.firreg %value clock %clock : i2

    firrtl.strictconnect %result, %count : !firrtl.uint<2>
    firrtl.strictconnect %count, %value : !firrtl.uint<2>

    // CHECK: hw.output %count : i2
  }

  // CHECK-LABEL: hw.module @ConnectSubfield(%clock: i1, %reset: i1, %value: i2) -> (result: !hw.struct<a: i2>)
  firrtl.module @ConnectSubfield(in %clock: !firrtl.clock,
                                 in %reset: !firrtl.uint<1>,
                                 in %value: !firrtl.uint<2>,
                                 out %result: !firrtl.bundle<a: uint<2>>) {
    %count = firrtl.reg %clock: !firrtl.bundle<a: uint<2>>
    // CHECK: %count = seq.firreg %0 clock %clock : !hw.struct<a: i2>

    firrtl.strictconnect %result, %count : !firrtl.bundle<a: uint<2>>
    %field = firrtl.subfield %count(0) : (!firrtl.bundle<a: uint<2>>) -> !firrtl.uint<2>
    firrtl.strictconnect %field, %value : !firrtl.uint<2>

    // CHECK: %a = hw.struct_extract %count["a"] : !hw.struct<a: i2>
    // CHECK: %0 = hw.struct_inject %count["a"], %value : !hw.struct<a: i2>

    // CHECK: hw.output %count : !hw.struct<a: i2>
  }

  // CHECK-LABEL: hw.module @ConnectSubfields(%clock: i1, %reset: i1, %value2: i2, %value3: i3) -> (result: !hw.struct<a: i2, b: i3>)
  firrtl.module @ConnectSubfields(in %clock: !firrtl.clock,
                                 in %reset: !firrtl.uint<1>,
                                 in %value2: !firrtl.uint<2>,
                                 in %value3: !firrtl.uint<3>,
                                 out %result: !firrtl.bundle<a: uint<2>, b: uint<3>>) {
    %count = firrtl.reg %clock: !firrtl.bundle<a: uint<2>, b: uint<3>>
    // CHECK: %count = seq.firreg [[AFTER_B:%.+]] clock %clock : !hw.struct<a: i2, b: i3>

    firrtl.strictconnect %result, %count : !firrtl.bundle<a: uint<2>, b: uint<3>>

    %fieldA = firrtl.subfield %count(0) : (!firrtl.bundle<a: uint<2>, b: uint<3>>) -> !firrtl.uint<2>
    firrtl.strictconnect %fieldA, %value2 : !firrtl.uint<2>
    %fieldB = firrtl.subfield %count(1) : (!firrtl.bundle<a: uint<2>, b: uint<3>>) -> !firrtl.uint<3>
    firrtl.strictconnect %fieldB, %value3 : !firrtl.uint<3>

    // CHECK: [[AFTER_A:%.+]] = hw.struct_inject %count["a"], %value2 : !hw.struct<a: i2, b: i3>
    // CHECK: [[AFTER_B]] = hw.struct_inject [[AFTER_A]]["b"], %value3 : !hw.struct<a: i2, b: i3>

    // CHECK: hw.output %count : !hw.struct<a: i2, b: i3>
  }

  // CHECK-LABEL: hw.module @ConnectNestedSubfield(%clock: i1, %reset: i1, %value: i2) -> (result: !hw.struct<a: !hw.struct<b: i2>>)
  firrtl.module @ConnectNestedSubfield(in %clock: !firrtl.clock,
                                       in %reset: !firrtl.uint<1>,
                                       in %value: !firrtl.uint<2>,
                                       out %result: !firrtl.bundle<a: bundle<b: uint<2>>>) {
    %count = firrtl.reg %clock: !firrtl.bundle<a: bundle<b: uint<2>>>
    // CHECK: %count = seq.firreg %1 clock %clock : !hw.struct<a: !hw.struct<b: i2>>

    firrtl.strictconnect %result, %count : !firrtl.bundle<a: bundle<b: uint<2>>>
    %field0 = firrtl.subfield %count(0) : (!firrtl.bundle<a: bundle<b: uint<2>>>) -> !firrtl.bundle<b: uint<2>>
    %field1 = firrtl.subfield %field0(0) : (!firrtl.bundle<b: uint<2>>) -> !firrtl.uint<2>
    firrtl.strictconnect %field1, %value : !firrtl.uint<2>

    // CHECK: %a = hw.struct_extract %count["a"] : !hw.struct<a: !hw.struct<b: i2>>
    // CHECK: %b = hw.struct_extract %a["b"] : !hw.struct<b: i2>
    // CHECK: %a_0 = hw.struct_extract %count["a"] : !hw.struct<a: !hw.struct<b: i2>>
    // CHECK: %0 = hw.struct_inject %a_0["b"], %value : !hw.struct<b: i2>
    // CHECK: %1 = hw.struct_inject %count["a"], %0 : !hw.struct<a: !hw.struct<b: i2>>

    // CHECK: hw.output %count : !hw.struct<a: !hw.struct<b: i2>>
  }

  // CHECK-LABEL: hw.module @ConnectNestedFieldsAndIndices(%clock: i1, %reset: i1, %value: i2) -> (result: !hw.array<5xstruct<a: !hw.array<3xstruct<b: i2>>>>)
  firrtl.module @ConnectNestedFieldsAndIndices(in %clock: !firrtl.clock,
                                 in %reset: !firrtl.uint<1>,
                                 in %value: !firrtl.uint<2>,
                                 out %result: !firrtl.vector<bundle<a: vector<bundle<b: uint<2>>, 3>>, 5>) {
    %count = firrtl.reg %clock: !firrtl.vector<bundle<a: vector<bundle<b: uint<2>>, 3>>, 5>
    %field1 = firrtl.subindex %count[1] : !firrtl.vector<bundle<a: vector<bundle<b: uint<2>>, 3>>, 5>
    %field2 = firrtl.subfield %field1(0) : (!firrtl.bundle<a: vector<bundle<b: uint<2>>, 3>>) -> !firrtl.vector<bundle<b: uint<2>>, 3>
    %field3 = firrtl.subindex %field2[1] : !firrtl.vector<bundle<b: uint<2>>, 3>
    %field4 = firrtl.subfield %field3(0) : (!firrtl.bundle<b: uint<2>>) -> !firrtl.uint<2>
    firrtl.strictconnect %field4, %value : !firrtl.uint<2>

    // CHECK:           %[[VAL_10:.*]] = seq.firreg %[[VAL_11:.*]] clock %clock : !hw.array<5xstruct<a: !hw.array<3xstruct<b: i2>>>>
    // CHECK:           %[[VAL_12:.*]] = hw.array_get %[[VAL_10]]{{\[}}%c1_i3] : !hw.array<5xstruct<a: !hw.array<3xstruct<b: i2>>>>
    // CHECK:           %[[VAL_13:.*]] = hw.struct_extract %[[VAL_12]]["a"] : !hw.struct<a: !hw.array<3xstruct<b: i2>>>
    // CHECK:           %[[VAL_14:.*]] = hw.array_get %[[VAL_13]]{{\[}}%c1_i2] : !hw.array<3xstruct<b: i2>>
    // CHECK:           %[[VAL_15:.*]] = hw.struct_extract %[[VAL_14]]["b"] : !hw.struct<b: i2>
    // CHECK:           %[[VAL_16:.*]] = hw.constant 1 : i3
    // CHECK:           %[[VAL_17:.*]] = hw.array_get %[[VAL_10]]{{\[}}%[[VAL_16]]] : !hw.array<5xstruct<a: !hw.array<3xstruct<b: i2>>>>
    // CHECK:           %[[VAL_18:.*]] = hw.struct_extract %[[VAL_17]]["a"] : !hw.struct<a: !hw.array<3xstruct<b: i2>>>
    // CHECK:           %[[VAL_20:.*]] = hw.array_get %[[VAL_18]]{{\[}}%c1_i2_2] : !hw.array<3xstruct<b: i2>>
    // CHECK:           %[[VAL_21:.*]] = hw.struct_inject %[[VAL_20]]["b"], %value : !hw.struct<b: i2>
    // CHECK:           %[[L0_HI:.*]] = hw.array_slice %[[VAL_18]]{{\[}}%c-2_i2] : (!hw.array<3xstruct<b: i2>>) -> !hw.array<1xstruct<b: i2>>
    // CHECK:           %[[L0_MID:.*]] = hw.array_create %[[VAL_21]] : !hw.struct<b: i2>
    // CHECK:           %[[L0_LO:.*]] = hw.array_slice %[[VAL_18]]{{\[}}%c0_i2] : (!hw.array<3xstruct<b: i2>>) -> !hw.array<1xstruct<b: i2>>
    // CHECK:           %[[VAL_25:.*]] = hw.array_concat %[[L0_HI]], %[[L0_MID]], %[[L0_LO]] : !hw.array<1xstruct<b: i2>>, !hw.array<1xstruct<b: i2>>, !hw.array<1xstruct<b: i2>>
    // CHECK:           %[[VAL_26:.*]] = hw.struct_inject %[[VAL_17]]["a"], %[[VAL_25]] : !hw.struct<a: !hw.array<3xstruct<b: i2>>>
    // CHECK:           %[[L1_HI:.*]] = hw.array_slice %[[VAL_10]]{{\[}}%c2_i3] : (!hw.array<5xstruct<a: !hw.array<3xstruct<b: i2>>>>) -> !hw.array<3xstruct<a: !hw.array<3xstruct<b: i2>>>>
    // CHECK:           %[[L1_MID:.*]] = hw.array_create %[[VAL_26]] : !hw.struct<a: !hw.array<3xstruct<b: i2>>>
    // CHECK:           %[[L1_LO:.*]] = hw.array_slice %[[VAL_10]]{{\[}}%c0_i3] : (!hw.array<5xstruct<a: !hw.array<3xstruct<b: i2>>>>) -> !hw.array<1xstruct<a: !hw.array<3xstruct<b: i2>>>>
    // CHECK:           %[[VAL_11]] = hw.array_concat %[[L1_HI]], %[[L1_MID]], %[[L1_LO]] : !hw.array<3xstruct<a: !hw.array<3xstruct<b: i2>>>>, !hw.array<1xstruct<a: !hw.array<3xstruct<b: i2>>>>, !hw.array<1xstruct<a: !hw.array<3xstruct<b: i2>>>>
  }

  // CHECK-LABEL: hw.module @ConnectSubindex(%clock: i1, %reset: i1, %value: i2) -> (result: !hw.array<3xi2>)
  firrtl.module @ConnectSubindex(in %clock: !firrtl.clock,
                                 in %reset: !firrtl.uint<1>,
                                 in %value: !firrtl.uint<2>,
                                 out %result: !firrtl.vector<uint<2>, 3>) {
    %count = firrtl.reg %clock: !firrtl.vector<uint<2>, 3>
    // CHECK: %count = seq.firreg [[REG:%.+]] clock %clock : !hw.array<3xi2>

    firrtl.strictconnect %result, %count : !firrtl.vector<uint<2>, 3>
    %field = firrtl.subindex %count[1] : !firrtl.vector<uint<2>, 3>
    firrtl.strictconnect %field, %value : !firrtl.uint<2>

    // CHECK: [[HI:%.+]] = hw.array_slice %count[%c-2_i2] : (!hw.array<3xi2>) -> !hw.array<1xi2>
    // CHECK: [[MID:%.+]] = hw.array_create %value : i2
    // CHECK: [[LO:%.+]] = hw.array_slice %count[%c0_i2] : (!hw.array<3xi2>) -> !hw.array<1xi2>
    // CHECK: [[REG]] = hw.array_concat [[HI]], [[MID]], [[LO]] : !hw.array<1xi2>, !hw.array<1xi2>, !hw.array<1xi2>

    // CHECK: hw.output %count : !hw.array<3xi2>
  }

  // CHECK-LABEL: hw.module @ConnectSubindexSingleton(%clock: i1, %reset: i1, %value: i2) -> (result: !hw.array<1xi2>)
  firrtl.module @ConnectSubindexSingleton(in %clock: !firrtl.clock,
                                 in %reset: !firrtl.uint<1>,
                                 in %value: !firrtl.uint<2>,
                                 out %result: !firrtl.vector<uint<2>, 1>) {
    %count = firrtl.reg %clock: !firrtl.vector<uint<2>, 1>
    // CHECK: %count = seq.firreg [[INPUT:%.+]] clock %clock : !hw.array<1xi2>

    firrtl.strictconnect %result, %count : !firrtl.vector<uint<2>, 1>
    %field = firrtl.subindex %count[0] : !firrtl.vector<uint<2>, 1>
    firrtl.strictconnect %field, %value : !firrtl.uint<2>

    // CHECK: [[INPUT]] = hw.array_create %value : i2
    // CHECK: hw.output %count : !hw.array<1xi2>
  }

  // CHECK-LABEL: hw.module @ConnectSubindexLHS(%clock: i1, %reset: i1, %value: i2) -> (result: !hw.array<3xi2>)
  firrtl.module @ConnectSubindexLHS(in %clock: !firrtl.clock,
                                 in %reset: !firrtl.uint<1>,
                                 in %value: !firrtl.uint<2>,
                                 out %result: !firrtl.vector<uint<2>, 3>) {
    %count = firrtl.reg %clock: !firrtl.vector<uint<2>, 3>
    // CHECK: %count = seq.firreg [[INPUT:%.+]] clock %clock : !hw.array<3xi2>

    firrtl.strictconnect %result, %count : !firrtl.vector<uint<2>, 3>
    %field = firrtl.subindex %count[0] : !firrtl.vector<uint<2>, 3>
    firrtl.strictconnect %field, %value : !firrtl.uint<2>

    // CHECK: [[REST:%.+]] = hw.array_slice %count[%c1_i2] : (!hw.array<3xi2>) -> !hw.array<2xi2>
    // CHECK: [[ELEM:%.+]] = hw.array_create %value : i2
    // CHECK: [[INPUT]] = hw.array_concat [[REST]], [[ELEM]] : !hw.array<2xi2>, !hw.array<1xi2>

    // CHECK: hw.output %count : !hw.array<3xi2>
  }

  // CHECK-LABEL: hw.module @ConnectSubindexRHS(%clock: i1, %reset: i1, %value: i2) -> (result: !hw.array<3xi2>)
  firrtl.module @ConnectSubindexRHS(in %clock: !firrtl.clock,
                                 in %reset: !firrtl.uint<1>,
                                 in %value: !firrtl.uint<2>,
                                 out %result: !firrtl.vector<uint<2>, 3>) {
    %count = firrtl.reg %clock: !firrtl.vector<uint<2>, 3>
    // CHECK: %count = seq.firreg [[INPUT:%.+]] clock %clock : !hw.array<3xi2>

    firrtl.strictconnect %result, %count : !firrtl.vector<uint<2>, 3>
    %field = firrtl.subindex %count[2] : !firrtl.vector<uint<2>, 3>
    firrtl.strictconnect %field, %value : !firrtl.uint<2>

    // CHECK: [[ELEM:%.+]] = hw.array_create %value : i2
    // CHECK: [[REST:%.+]] = hw.array_slice %count[%c0_i2] : (!hw.array<3xi2>) -> !hw.array<2xi2>
    // CHECK: [[INPUT]] = hw.array_concat  [[ELEM]], [[REST]] : !hw.array<1xi2>, !hw.array<2xi2>

    // CHECK: hw.output %count : !hw.array<3xi2>
  }

  // CHECK-LABEL: hw.module @ConnectSubindices(%clock: i1, %reset: i1, %value: i2) -> (result: !hw.array<5xi2>)
  firrtl.module @ConnectSubindices(in %clock: !firrtl.clock,
                                 in %reset: !firrtl.uint<1>,
                                 in %value: !firrtl.uint<2>,
                                 out %result: !firrtl.vector<uint<2>, 5>) {
    %count = firrtl.reg %clock: !firrtl.vector<uint<2>, 5>
    // CHECK: %count = seq.firreg [[NEXT:%.+]] clock %clock : !hw.array<5xi2>

    firrtl.strictconnect %result, %count : !firrtl.vector<uint<2>, 5>

    %field1 = firrtl.subindex %count[1] : !firrtl.vector<uint<2>, 5>
    firrtl.strictconnect %field1, %value : !firrtl.uint<2>
    %field2 = firrtl.subindex %count[2] : !firrtl.vector<uint<2>, 5>
    firrtl.strictconnect %field2, %value : !firrtl.uint<2>
    %field5 = firrtl.subindex %count[4] : !firrtl.vector<uint<2>, 5>
    firrtl.strictconnect %field5, %value : !firrtl.uint<2>

    // CHECK-DAG: [[L0_LO:%.+]] = hw.array_slice %count[%c0_i3] : (!hw.array<5xi2>) -> !hw.array<1xi2>
    // CHECK-DAG: [[L0_MID:%.+]] = hw.array_create %value : i2
    // CHECK-DAG: [[L0_HI:%.+]] = hw.array_slice %count[%c2_i3] : (!hw.array<5xi2>) -> !hw.array<3xi2>
    // CHECK-DAG: [[IN:%.+]] = hw.array_concat [[L0_HI]], [[L0_MID]], [[L0_LO]] : !hw.array<3xi2>, !hw.array<1xi2>, !hw.array<1xi2>
    // CHECK-DAG: [[L1_LO:%.+]] = hw.array_slice [[IN]][%c0_i3] : (!hw.array<5xi2>) -> !hw.array<2xi2>
    // CHECK-DAG: [[L1_MID:%.+]] = hw.array_create %value : i2
    // CHECK-DAG: [[L1_HI:%.+]] = hw.array_slice %4[%c3_i3] : (!hw.array<5xi2>) -> !hw.array<2xi2>
    // CHECK-DAG: [[L2_LO:%.+]] = hw.array_concat [[L1_HI]], [[L1_MID]], [[L1_LO]] : !hw.array<2xi2>, !hw.array<1xi2>, !hw.array<2xi2>
    // CHECK-DAG: [[L3_LO:%.+]] = hw.array_slice [[L2_LO]][%c0_i3] : (!hw.array<5xi2>) -> !hw.array<4xi2>
    // CHECK-DAG: [[L3_MID:%.+]] = hw.array_create %value : i2
    // CHECK-DAG: [[NEXT]] = hw.array_concat [[L3_MID]], [[L3_LO]] : !hw.array<1xi2>, !hw.array<4xi2>

    // CHECK: hw.output %count : !hw.array<5xi2>
  }

  // CHECK-LABEL: hw.module @ConnectNestedSubindex(%clock: i1, %reset: i1, %value: i2) -> (result: !hw.array<3xarray<3xi2>>)
  firrtl.module @ConnectNestedSubindex(in %clock: !firrtl.clock,
                                       in %reset: !firrtl.uint<1>,
                                       in %value: !firrtl.uint<2>,
                                       out %result: !firrtl.vector<vector<uint<2>, 3>, 3>) {
    %count = firrtl.reg %clock: !firrtl.vector<vector<uint<2>, 3>, 3>
    // CHECK: %count = seq.firreg %10 clock %clock : !hw.array<3xarray<3xi2>>

    firrtl.strictconnect %result, %count : !firrtl.vector<vector<uint<2>, 3>, 3>
    %field0 = firrtl.subindex %count[1] : !firrtl.vector<vector<uint<2>, 3>, 3>
    %field1 = firrtl.subindex %field0[1] : !firrtl.vector<uint<2>, 3>
    firrtl.strictconnect %field1, %value : !firrtl.uint<2>

    // CHECK-DAG: %0 = hw.array_get %count[%c1_i2] : !hw.array<3xarray<3xi2>>
    // CHECK-DAG: %1 = hw.array_get %0[%c1_i2] : !hw.array<3xi2>
    // CHECK-DAG: %2 = hw.array_get %count[%c1_i2_0] : !hw.array<3xarray<3xi2>>
    // CHECK-DAG: [[IN_LO:%.+]] = hw.array_slice %2[%c0_i2] : (!hw.array<3xi2>) -> !hw.array<1xi2>
    // CHECK-DAG: %4 = hw.array_create %value : i2
    // CHECK-DAG: [[IN_HI:%.+]] = hw.array_slice %2[%c-2_i2] : (!hw.array<3xi2>) -> !hw.array<1xi2>
    // CHECK-DAG: %6 = hw.array_concat [[IN_HI]], %4, [[IN_LO]] : !hw.array<1xi2>, !hw.array<1xi2>, !hw.array<1xi2>
    // CHECK-DAG: [[OUT_LO:%.+]] = hw.array_slice %count[%c0_i2] : (!hw.array<3xarray<3xi2>>) -> !hw.array<1xarray<3xi2>>
    // CHECK-DAG: [[OUT_MID:%.+]] = hw.array_create %6 : !hw.array<3xi2>
    // CHECK-DAG: [[OUT_HI:%.+]] = hw.array_slice %count[%c-2_i2] : (!hw.array<3xarray<3xi2>>) -> !hw.array<1xarray<3xi2>>

    // CHECK-DAG: %10 = hw.array_concat [[OUT_HI]], [[OUT_MID]], [[OUT_LO]] : !hw.array<1xarray<3xi2>>, !hw.array<1xarray<3xi2>>, !hw.array<1xarray<3xi2>>

    // CHECK: hw.output %count : !hw.array<3xarray<3xi2>>
  }

  // CHECK-LABEL: hw.module @SyncReset(%clock: i1, %reset: i1, %value: i2) -> (result: i2)
  firrtl.module @SyncReset(in %clock: !firrtl.clock,
                           in %reset: !firrtl.uint<1>,
                           in %value: !firrtl.uint<2>,
                           out %result: !firrtl.uint<2>) {
    %count = firrtl.regreset %clock, %reset, %value : !firrtl.uint<1>, !firrtl.uint<2>, !firrtl.uint<2>

    // CHECK: %count = seq.firreg %count clock %clock reset sync %reset, %value : i2
    // CHECK: hw.output %count : i2

    firrtl.strictconnect %result, %count : !firrtl.uint<2>
  }

  // CHECK-LABEL: hw.module @AsyncReset(%clock: i1, %reset: i1, %value: i2) -> (result: i2)
  firrtl.module @AsyncReset(in %clock: !firrtl.clock,
                           in %reset: !firrtl.asyncreset,
                           in %value: !firrtl.uint<2>,
                           out %result: !firrtl.uint<2>) {
    %count = firrtl.regreset %clock, %reset, %value : !firrtl.asyncreset, !firrtl.uint<2>, !firrtl.uint<2>

    // CHECK: %count = seq.firreg %value clock %clock reset async %reset, %value : i2
    // CHECK: hw.output %count : i2

    firrtl.strictconnect %count, %value : !firrtl.uint<2>
    firrtl.strictconnect %result, %count : !firrtl.uint<2>
  }

  // CHECK-LABEL: hw.module @NoConnect(%clock: i1, %reset: i1) -> (result: i2)
  firrtl.module @NoConnect(in %clock: !firrtl.clock,
                     in %reset: !firrtl.uint<1>,
                     out %result: !firrtl.uint<2>) {
    %count = firrtl.reg %clock: !firrtl.uint<2>
    // CHECK: %count = seq.firreg %count clock %clock : i2

    firrtl.strictconnect %result, %count : !firrtl.uint<2>

    // CHECK: hw.output %count : i2
  }
  // CHECK-LABEL: hw.module @passThroughForeignTypes
  // CHECK-SAME:      (%inOpaque: index) -> (outOpaque: index) {
  // CHECK-NEXT:    %sub2.bar = hw.instance "sub2" @moreForeignTypes(foo: %sub1.bar: index) -> (bar: index)
  // CHECK-NEXT:    %sub1.bar = hw.instance "sub1" @moreForeignTypes(foo: %inOpaque: index) -> (bar: index)
  // CHECK-NEXT:    hw.output %sub2.bar : index
  // CHECK-NEXT:  }
  // CHECK-LABEL: hw.module @moreForeignTypes
  // CHECK-SAME:      (%foo: index) -> (bar: index) {
  // CHECK-NEXT:    hw.output %foo : index
  // CHECK-NEXT:  }
  firrtl.module @passThroughForeignTypes(in %inOpaque: index, out %outOpaque: index) {
    // Declaration order intentionally reversed to enforce use-before-def in HW
    %sub2_foo, %sub2_bar = firrtl.instance sub2 @moreForeignTypes(in foo: index, out bar: index)
    %sub1_foo, %sub1_bar = firrtl.instance sub1 @moreForeignTypes(in foo: index, out bar: index)
    firrtl.strictconnect %sub1_foo, %inOpaque : index
    firrtl.strictconnect %sub2_foo, %sub1_bar : index
    firrtl.strictconnect %outOpaque, %sub2_bar : index
  }
  firrtl.module @moreForeignTypes(in %foo: index, out %bar: index) {
    firrtl.strictconnect %bar, %foo : index
  }

  // CHECK-LABEL: hw.module @foreignOpsOnForeignTypes
  // CHECK-SAME:      (%x: f32) -> (y: f32) {
  // CHECK-NEXT:    [[TMP:%.+]] = arith.addf %x, %x : f32
  // CHECK-NEXT:    hw.output [[TMP]] : f32
  // CHECK-NEXT:  }
  firrtl.module @foreignOpsOnForeignTypes(in %x: f32, out %y: f32) {
    %0 = arith.addf %x, %x : f32
    firrtl.strictconnect %y, %0 : f32
  }

  // CHECK-LABEL: hw.module @wiresWithForeignTypes
  // CHECK-SAME:      (%in: f32) -> (out: f32) {
  // CHECK-NEXT:    [[ADD1:%.+]] = arith.addf [[ADD2:%.+]], [[ADD2]] : f32
  // CHECK-NEXT:    [[ADD2]] = arith.addf %in, [[ADD2]] : f32
  // CHECK-NEXT:    hw.output [[ADD1]] : f32
  // CHECK-NEXT:  }
  firrtl.module @wiresWithForeignTypes(in %in: f32, out %out: f32) {
    %w1 = firrtl.wire : f32
    %w2 = firrtl.wire : f32
    firrtl.strictconnect %out, %w2 : f32
    %0 = arith.addf %w1, %w1 : f32
    firrtl.strictconnect %w2, %0 : f32
    %1 = arith.addf %in, %w1 : f32
    firrtl.strictconnect %w1, %1 : f32
  }

  // CHECK-LABEL: LowerReadArrayInoutIntoArrayGet
  firrtl.module @LowerReadArrayInoutIntoArrayGet(in %a: !firrtl.uint<10>, out %b: !firrtl.uint<10>) {
    %r = firrtl.wire   : !firrtl.vector<uint<10>, 1>
    %0 = firrtl.subindex %r[0] : !firrtl.vector<uint<10>, 1>
    // CHECK: %r = sv.wire  : !hw.inout<array<1xi10>>
    // CHECK: %[[WIRE_VAL:.+]] = sv.read_inout %r : !hw.inout<array<1xi10>>
    // CHECK: %[[RET:.+]] = hw.array_get %[[WIRE_VAL]][%false] : !hw.array<1xi10>, i1
    // CHECK: hw.output %[[RET]]
    firrtl.strictconnect %0, %a : !firrtl.uint<10>
    firrtl.strictconnect %b, %0 : !firrtl.uint<10>
  }

  // CHECK-LABEL: hw.module @MergeBundle
  firrtl.module @MergeBundle(out %o: !firrtl.bundle<valid: uint<1>, ready: uint<1>>, in %i: !firrtl.uint<1>) 
  {
    %a = firrtl.wire   : !firrtl.bundle<valid: uint<1>, ready: uint<1>>
    firrtl.strictconnect %o, %a : !firrtl.bundle<valid: uint<1>, ready: uint<1>>
    %0 = firrtl.bundlecreate %i, %i : (!firrtl.uint<1>, !firrtl.uint<1>) -> !firrtl.bundle<valid: uint<1>, ready: uint<1>>
    firrtl.strictconnect %a, %0 : !firrtl.bundle<valid: uint<1>, ready: uint<1>>
    // CHECK:  %a = sv.wire : !hw.inout<struct<valid: i1, ready: i1>> 
    // CHECK:  %0 = sv.read_inout %a : !hw.inout<struct<valid: i1, ready: i1>> 
    // CHECK:  %1 = hw.struct_create (%i, %i) : !hw.struct<valid: i1, ready: i1> 
    // CHECK:  sv.assign %a, %1 : !hw.struct<valid: i1, ready: i1> 
    // CHECK:  hw.output %0 : !hw.struct<valid: i1, ready: i1> 
  }
 
  // CHECK-LABEL: hw.module @MergeVector
  firrtl.module @MergeVector(out %o: !firrtl.vector<uint<1>, 3>, in %i: !firrtl.uint<1>, in %j: !firrtl.uint<1>) {
    %a = firrtl.wire   : !firrtl.vector<uint<1>, 3>
    firrtl.strictconnect %o, %a : !firrtl.vector<uint<1>, 3>
    %0 = firrtl.vectorcreate %i, %i, %j : (!firrtl.uint<1>, !firrtl.uint<1>, !firrtl.uint<1>) -> !firrtl.vector<uint<1>, 3>
    firrtl.strictconnect %a, %0 : !firrtl.vector<uint<1>, 3>
    // CHECK:  %a = sv.wire : !hw.inout<array<3xi1>> 
    // CHECK:  %0 = sv.read_inout %a : !hw.inout<array<3xi1>> 
    // CHECK:  %1 = hw.array_create %j, %i, %i : i1
    // CHECK:  sv.assign %a, %1 : !hw.array<3xi1> 
    // CHECK:  hw.output %0 : !hw.array<3xi1> 
  }

}
