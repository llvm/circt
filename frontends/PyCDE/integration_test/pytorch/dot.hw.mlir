// As of circt e64dfed9323920d7e85e89ec8119fb8d57713984
// circt-opt dot.cf.mlir --flatten-memref --flatten-memref-calls --handshake-legalize-memrefs --lower-std-to-handshake --canonicalize --handshake-lower-extmem-to-hw=wrap-esi --handshake-insert-buffers --canonicalize --handshake-remove-block-structure --handshake-materialize-forks-sinks --lower-handshake-to-hw --canonicalize

hw.module @handshake_buffer_2slots_seq_1ins_1outs_ctrl(%in0: !esi.channel<i0>, %clock: i1, %reset: i1) -> (out0: !esi.channel<i0>) {
  %true = hw.constant true
  %false = hw.constant false
  %c0_i0 = hw.constant 0 : i0
  %rawOutput, %valid = esi.unwrap.vr %in0, %1 : i0
  %chanOutput, %ready = esi.wrap.vr %27, %19 : i0
  %valid0_reg = seq.compreg %2, %clock, %reset, %false  : i1
  %0 = comb.xor %valid0_reg, %true : i1
  %1 = comb.or %0, %5 : i1
  %2 = comb.mux %1, %valid, %valid0_reg : i1
  %3 = comb.mux %1, %rawOutput, %data0_reg : i0
  %data0_reg = seq.compreg %3, %clock, %reset, %c0_i0  : i0
  %ready0_reg = seq.compreg %11, %clock, %reset, %false  : i1
  %4 = comb.mux %ready0_reg, %ready0_reg, %valid0_reg : i1
  %5 = comb.xor %ready0_reg, %true : i1
  %6 = comb.xor %16, %true : i1
  %7 = comb.and %6, %5 : i1
  %8 = comb.mux %7, %valid0_reg, %ready0_reg : i1
  %9 = comb.and %16, %ready0_reg : i1
  %10 = comb.xor %9, %true : i1
  %11 = comb.and %10, %8 : i1
  %ctrl_data0_reg = seq.compreg %14, %clock, %reset, %c0_i0  : i0
  %12 = comb.mux %ready0_reg, %ctrl_data0_reg, %data0_reg : i0
  %13 = comb.mux %7, %data0_reg, %ctrl_data0_reg : i0
  %14 = comb.mux %9, %c0_i0, %13 : i0
  %valid1_reg = seq.compreg %17, %clock, %reset, %false  : i1
  %15 = comb.xor %valid1_reg, %true : i1
  %16 = comb.or %15, %20 : i1
  %17 = comb.mux %16, %4, %valid1_reg : i1
  %18 = comb.mux %16, %12, %data1_reg : i0
  %data1_reg = seq.compreg %18, %clock, %reset, %c0_i0  : i0
  %ready1_reg = seq.compreg %26, %clock, %reset, %false  : i1
  %19 = comb.mux %ready1_reg, %ready1_reg, %valid1_reg : i1
  %20 = comb.xor %ready1_reg, %true : i1
  %21 = comb.xor %ready, %true : i1
  %22 = comb.and %21, %20 : i1
  %23 = comb.mux %22, %valid1_reg, %ready1_reg : i1
  %24 = comb.and %ready, %ready1_reg : i1
  %25 = comb.xor %24, %true : i1
  %26 = comb.and %25, %23 : i1
  %ctrl_data1_reg = seq.compreg %29, %clock, %reset, %c0_i0  : i0
  %27 = comb.mux %ready1_reg, %ctrl_data1_reg, %data1_reg : i0
  %28 = comb.mux %22, %data1_reg, %ctrl_data1_reg : i0
  %29 = comb.mux %24, %c0_i0, %28 : i0
  hw.output %chanOutput : !esi.channel<i0>
}
hw.module @handshake_buffer_in_ui32_out_ui32_2slots_seq(%in0: !esi.channel<i32>, %clock: i1, %reset: i1) -> (out0: !esi.channel<i32>) {
  %true = hw.constant true
  %false = hw.constant false
  %c0_i32 = hw.constant 0 : i32
  %rawOutput, %valid = esi.unwrap.vr %in0, %1 : i32
  %chanOutput, %ready = esi.wrap.vr %27, %19 : i32
  %valid0_reg = seq.compreg %2, %clock, %reset, %false  : i1
  %0 = comb.xor %valid0_reg, %true : i1
  %1 = comb.or %0, %5 : i1
  %2 = comb.mux %1, %valid, %valid0_reg : i1
  %3 = comb.mux %1, %rawOutput, %data0_reg : i32
  %data0_reg = seq.compreg %3, %clock, %reset, %c0_i32  : i32
  %ready0_reg = seq.compreg %11, %clock, %reset, %false  : i1
  %4 = comb.mux %ready0_reg, %ready0_reg, %valid0_reg : i1
  %5 = comb.xor %ready0_reg, %true : i1
  %6 = comb.xor %16, %true : i1
  %7 = comb.and %6, %5 : i1
  %8 = comb.mux %7, %valid0_reg, %ready0_reg : i1
  %9 = comb.and %16, %ready0_reg : i1
  %10 = comb.xor %9, %true : i1
  %11 = comb.and %10, %8 : i1
  %ctrl_data0_reg = seq.compreg %14, %clock, %reset, %c0_i32  : i32
  %12 = comb.mux %ready0_reg, %ctrl_data0_reg, %data0_reg : i32
  %13 = comb.mux %7, %data0_reg, %ctrl_data0_reg : i32
  %14 = comb.mux %9, %c0_i32, %13 : i32
  %valid1_reg = seq.compreg %17, %clock, %reset, %false  : i1
  %15 = comb.xor %valid1_reg, %true : i1
  %16 = comb.or %15, %20 : i1
  %17 = comb.mux %16, %4, %valid1_reg : i1
  %18 = comb.mux %16, %12, %data1_reg : i32
  %data1_reg = seq.compreg %18, %clock, %reset, %c0_i32  : i32
  %ready1_reg = seq.compreg %26, %clock, %reset, %false  : i1
  %19 = comb.mux %ready1_reg, %ready1_reg, %valid1_reg : i1
  %20 = comb.xor %ready1_reg, %true : i1
  %21 = comb.xor %ready, %true : i1
  %22 = comb.and %21, %20 : i1
  %23 = comb.mux %22, %valid1_reg, %ready1_reg : i1
  %24 = comb.and %ready, %ready1_reg : i1
  %25 = comb.xor %24, %true : i1
  %26 = comb.and %25, %23 : i1
  %ctrl_data1_reg = seq.compreg %29, %clock, %reset, %c0_i32  : i32
  %27 = comb.mux %ready1_reg, %ctrl_data1_reg, %data1_reg : i32
  %28 = comb.mux %22, %data1_reg, %ctrl_data1_reg : i32
  %29 = comb.mux %24, %c0_i32, %28 : i32
  hw.output %chanOutput : !esi.channel<i32>
}
hw.module @handshake_memory_out_ui32_id3(%stData0: !esi.channel<i32>, %stAddr0: !esi.channel<i64>, %stData1: !esi.channel<i32>, %stAddr1: !esi.channel<i64>, %ldAddr0: !esi.channel<i64>, %ldAddr1: !esi.channel<i64>, %clock: i1, %reset: i1) -> (ldData0: !esi.channel<i32>, ldData1: !esi.channel<i32>, stDone0: !esi.channel<i0>, stDone1: !esi.channel<i0>, ldDone0: !esi.channel<i0>, ldDone1: !esi.channel<i0>) {
  %c0_i0 = hw.constant 0 : i0
  %true = hw.constant true
  %false = hw.constant false
  %rawOutput, %valid = esi.unwrap.vr %stData0, %28 : i32
  %rawOutput_0, %valid_1 = esi.unwrap.vr %stAddr0, %28 : i64
  %rawOutput_2, %valid_3 = esi.unwrap.vr %stData1, %33 : i32
  %rawOutput_4, %valid_5 = esi.unwrap.vr %stAddr1, %33 : i64
  %rawOutput_6, %valid_7 = esi.unwrap.vr %ldAddr0, %12 : i64
  %rawOutput_8, %valid_9 = esi.unwrap.vr %ldAddr1, %25 : i64
  %chanOutput, %ready = esi.wrap.vr %_handshake_memory_3_rdata, %3 : i32
  %chanOutput_10, %ready_11 = esi.wrap.vr %_handshake_memory_3_rdata_20, %16 : i32
  %chanOutput_12, %ready_13 = esi.wrap.vr %c0_i0, %writeValidBuffer : i0
  %chanOutput_14, %ready_15 = esi.wrap.vr %c0_i0, %writeValidBuffer_23 : i0
  %chanOutput_16, %ready_17 = esi.wrap.vr %c0_i0, %9 : i0
  %chanOutput_18, %ready_19 = esi.wrap.vr %c0_i0, %22 : i0
  %_handshake_memory_3 = seq.hlmem @_handshake_memory_3 %clock, %reset : <1xi32>
  %_handshake_memory_3_rdata = seq.read %_handshake_memory_3[%c0_i0] rden %valid_7 {latency = 0 : i64} : !seq.hlmem<1xi32>
  %0 = comb.xor %12, %true : i1
  %1 = comb.and %5, %0 : i1
  %emitted_0 = seq.compreg %1, %clock, %reset, %false  : i1
  %2 = comb.xor %emitted_0, %true : i1
  %3 = comb.and %2, %valid_7 : i1
  %4 = comb.and %ready, %3 : i1
  %5 = comb.or %4, %emitted_0 {sv.namehint = "done0"} : i1
  %6 = comb.xor %12, %true : i1
  %7 = comb.and %11, %6 : i1
  %emitted_1 = seq.compreg %7, %clock, %reset, %false  : i1
  %8 = comb.xor %emitted_1, %true : i1
  %9 = comb.and %8, %valid_7 : i1
  %10 = comb.and %ready_17, %9 : i1
  %11 = comb.or %10, %emitted_1 {sv.namehint = "done1"} : i1
  %12 = comb.and %5, %11 {sv.namehint = "allDone"} : i1
  %_handshake_memory_3_rdata_20 = seq.read %_handshake_memory_3[%c0_i0] rden %valid_9 {latency = 0 : i64} : !seq.hlmem<1xi32>
  %13 = comb.xor %25, %true : i1
  %14 = comb.and %18, %13 : i1
  %emitted_0_21 = seq.compreg %14, %clock, %reset, %false  {name = "emitted_0"} : i1
  %15 = comb.xor %emitted_0_21, %true : i1
  %16 = comb.and %15, %valid_9 : i1
  %17 = comb.and %ready_11, %16 : i1
  %18 = comb.or %17, %emitted_0_21 {sv.namehint = "done0"} : i1
  %19 = comb.xor %25, %true : i1
  %20 = comb.and %24, %19 : i1
  %emitted_1_22 = seq.compreg %20, %clock, %reset, %false  {name = "emitted_1"} : i1
  %21 = comb.xor %emitted_1_22, %true : i1
  %22 = comb.and %21, %valid_9 : i1
  %23 = comb.and %ready_19, %22 : i1
  %24 = comb.or %23, %emitted_1_22 {sv.namehint = "done1"} : i1
  %25 = comb.and %18, %24 {sv.namehint = "allDone"} : i1
  %writeValidBuffer = seq.compreg %30, %clock, %reset, %false  : i1
  %26 = comb.and %ready_13, %writeValidBuffer {sv.namehint = "storeCompleted"} : i1
  %27 = comb.xor %writeValidBuffer, %true : i1
  %28 = comb.or %27, %26 {sv.namehint = "emptyOrComplete"} : i1
  %29 = comb.and %valid_1, %valid {sv.namehint = "writeValid"} : i1
  %30 = comb.mux %28, %29, %writeValidBuffer : i1
  seq.write %_handshake_memory_3[%c0_i0] %rawOutput wren %29 {latency = 1 : i64} : !seq.hlmem<1xi32>
  %writeValidBuffer_23 = seq.compreg %35, %clock, %reset, %false  {name = "writeValidBuffer"} : i1
  %31 = comb.and %ready_15, %writeValidBuffer_23 {sv.namehint = "storeCompleted"} : i1
  %32 = comb.xor %writeValidBuffer_23, %true : i1
  %33 = comb.or %32, %31 {sv.namehint = "emptyOrComplete"} : i1
  %34 = comb.and %valid_5, %valid_3 {sv.namehint = "writeValid"} : i1
  %35 = comb.mux %33, %34, %writeValidBuffer_23 : i1
  seq.write %_handshake_memory_3[%c0_i0] %rawOutput_2 wren %34 {latency = 1 : i64} : !seq.hlmem<1xi32>
  hw.output %chanOutput, %chanOutput_10, %chanOutput_12, %chanOutput_14, %chanOutput_16, %chanOutput_18 : !esi.channel<i32>, !esi.channel<i32>, !esi.channel<i0>, !esi.channel<i0>, !esi.channel<i0>, !esi.channel<i0>
}
hw.module @handshake_fork_1ins_2outs_ctrl(%in0: !esi.channel<i0>, %clock: i1, %reset: i1) -> (out0: !esi.channel<i0>, out1: !esi.channel<i0>) {
  %true = hw.constant true
  %false = hw.constant false
  %rawOutput, %valid = esi.unwrap.vr %in0, %12 : i0
  %chanOutput, %ready = esi.wrap.vr %rawOutput, %3 : i0
  %chanOutput_0, %ready_1 = esi.wrap.vr %rawOutput, %9 : i0
  %0 = comb.xor %12, %true : i1
  %1 = comb.and %5, %0 : i1
  %emitted_0 = seq.compreg %1, %clock, %reset, %false  : i1
  %2 = comb.xor %emitted_0, %true : i1
  %3 = comb.and %2, %valid : i1
  %4 = comb.and %ready, %3 : i1
  %5 = comb.or %4, %emitted_0 {sv.namehint = "done0"} : i1
  %6 = comb.xor %12, %true : i1
  %7 = comb.and %11, %6 : i1
  %emitted_1 = seq.compreg %7, %clock, %reset, %false  : i1
  %8 = comb.xor %emitted_1, %true : i1
  %9 = comb.and %8, %valid : i1
  %10 = comb.and %ready_1, %9 : i1
  %11 = comb.or %10, %emitted_1 {sv.namehint = "done1"} : i1
  %12 = comb.and %5, %11 {sv.namehint = "allDone"} : i1
  hw.output %chanOutput, %chanOutput_0 : !esi.channel<i0>, !esi.channel<i0>
}
hw.module @arith_index_cast_in_ui64_out_ui0(%in0: !esi.channel<i64>) -> (out0: !esi.channel<i0>) {
  %c0_i0 = hw.constant 0 : i0
  %rawOutput, %valid = esi.unwrap.vr %in0, %0 : i64
  %chanOutput, %ready = esi.wrap.vr %c0_i0, %valid : i0
  %0 = comb.and %ready, %valid : i1
  hw.output %chanOutput : !esi.channel<i0>
}
hw.module @handshake_buffer_in_ui0_out_ui0_2slots_seq(%in0: !esi.channel<i0>, %clock: i1, %reset: i1) -> (out0: !esi.channel<i0>) {
  %true = hw.constant true
  %false = hw.constant false
  %c0_i0 = hw.constant 0 : i0
  %rawOutput, %valid = esi.unwrap.vr %in0, %1 : i0
  %chanOutput, %ready = esi.wrap.vr %27, %19 : i0
  %valid0_reg = seq.compreg %2, %clock, %reset, %false  : i1
  %0 = comb.xor %valid0_reg, %true : i1
  %1 = comb.or %0, %5 : i1
  %2 = comb.mux %1, %valid, %valid0_reg : i1
  %3 = comb.mux %1, %rawOutput, %data0_reg : i0
  %data0_reg = seq.compreg %3, %clock, %reset, %c0_i0  : i0
  %ready0_reg = seq.compreg %11, %clock, %reset, %false  : i1
  %4 = comb.mux %ready0_reg, %ready0_reg, %valid0_reg : i1
  %5 = comb.xor %ready0_reg, %true : i1
  %6 = comb.xor %16, %true : i1
  %7 = comb.and %6, %5 : i1
  %8 = comb.mux %7, %valid0_reg, %ready0_reg : i1
  %9 = comb.and %16, %ready0_reg : i1
  %10 = comb.xor %9, %true : i1
  %11 = comb.and %10, %8 : i1
  %ctrl_data0_reg = seq.compreg %14, %clock, %reset, %c0_i0  : i0
  %12 = comb.mux %ready0_reg, %ctrl_data0_reg, %data0_reg : i0
  %13 = comb.mux %7, %data0_reg, %ctrl_data0_reg : i0
  %14 = comb.mux %9, %c0_i0, %13 : i0
  %valid1_reg = seq.compreg %17, %clock, %reset, %false  : i1
  %15 = comb.xor %valid1_reg, %true : i1
  %16 = comb.or %15, %20 : i1
  %17 = comb.mux %16, %4, %valid1_reg : i1
  %18 = comb.mux %16, %12, %data1_reg : i0
  %data1_reg = seq.compreg %18, %clock, %reset, %c0_i0  : i0
  %ready1_reg = seq.compreg %26, %clock, %reset, %false  : i1
  %19 = comb.mux %ready1_reg, %ready1_reg, %valid1_reg : i1
  %20 = comb.xor %ready1_reg, %true : i1
  %21 = comb.xor %ready, %true : i1
  %22 = comb.and %21, %20 : i1
  %23 = comb.mux %22, %valid1_reg, %ready1_reg : i1
  %24 = comb.and %ready, %ready1_reg : i1
  %25 = comb.xor %24, %true : i1
  %26 = comb.and %25, %23 : i1
  %ctrl_data1_reg = seq.compreg %29, %clock, %reset, %c0_i0  : i0
  %27 = comb.mux %ready1_reg, %ctrl_data1_reg, %data1_reg : i0
  %28 = comb.mux %22, %data1_reg, %ctrl_data1_reg : i0
  %29 = comb.mux %24, %c0_i0, %28 : i0
  hw.output %chanOutput : !esi.channel<i0>
}
hw.module @hw_struct_create_in_ui0_ui32_out_struct_address_ui0_data_ui32(%in0: !esi.channel<i0>, %in1: !esi.channel<i32>) -> (out0: !esi.channel<!hw.struct<address: i0, data: i32>>) {
  %rawOutput, %valid = esi.unwrap.vr %in0, %1 : i0
  %rawOutput_0, %valid_1 = esi.unwrap.vr %in1, %1 : i32
  %chanOutput, %ready = esi.wrap.vr %2, %0 : !hw.struct<address: i0, data: i32>
  %0 = comb.and %valid, %valid_1 : i1
  %1 = comb.and %ready, %0 : i1
  %2 = hw.struct_create (%rawOutput, %rawOutput_0) : !hw.struct<address: i0, data: i32>
  hw.output %chanOutput : !esi.channel<!hw.struct<address: i0, data: i32>>
}
hw.module @handshake_buffer_in_struct_address_ui0_data_ui32_out_struct_address_ui0_data_ui32_2slots_seq(%in0: !esi.channel<!hw.struct<address: i0, data: i32>>, %clock: i1, %reset: i1) -> (out0: !esi.channel<!hw.struct<address: i0, data: i32>>) {
  %true = hw.constant true
  %false = hw.constant false
  %c0_i32 = hw.constant 0 : i32
  %c0_i0 = hw.constant 0 : i0
  %rawOutput, %valid = esi.unwrap.vr %in0, %2 : !hw.struct<address: i0, data: i32>
  %chanOutput, %ready = esi.wrap.vr %29, %21 : !hw.struct<address: i0, data: i32>
  %0 = hw.struct_create (%c0_i0, %c0_i32) : !hw.struct<address: i0, data: i32>
  %valid0_reg = seq.compreg %3, %clock, %reset, %false  : i1
  %1 = comb.xor %valid0_reg, %true : i1
  %2 = comb.or %1, %6 : i1
  %3 = comb.mux %2, %valid, %valid0_reg : i1
  %4 = comb.mux %2, %rawOutput, %data0_reg : !hw.struct<address: i0, data: i32>
  %data0_reg = seq.compreg %4, %clock, %reset, %0  : !hw.struct<address: i0, data: i32>
  %ready0_reg = seq.compreg %12, %clock, %reset, %false  : i1
  %5 = comb.mux %ready0_reg, %ready0_reg, %valid0_reg : i1
  %6 = comb.xor %ready0_reg, %true : i1
  %7 = comb.xor %18, %true : i1
  %8 = comb.and %7, %6 : i1
  %9 = comb.mux %8, %valid0_reg, %ready0_reg : i1
  %10 = comb.and %18, %ready0_reg : i1
  %11 = comb.xor %10, %true : i1
  %12 = comb.and %11, %9 : i1
  %ctrl_data0_reg = seq.compreg %15, %clock, %reset, %0  : !hw.struct<address: i0, data: i32>
  %13 = comb.mux %ready0_reg, %ctrl_data0_reg, %data0_reg : !hw.struct<address: i0, data: i32>
  %14 = comb.mux %8, %data0_reg, %ctrl_data0_reg : !hw.struct<address: i0, data: i32>
  %15 = comb.mux %10, %0, %14 : !hw.struct<address: i0, data: i32>
  %16 = hw.struct_create (%c0_i0, %c0_i32) : !hw.struct<address: i0, data: i32>
  %valid1_reg = seq.compreg %19, %clock, %reset, %false  : i1
  %17 = comb.xor %valid1_reg, %true : i1
  %18 = comb.or %17, %22 : i1
  %19 = comb.mux %18, %5, %valid1_reg : i1
  %20 = comb.mux %18, %13, %data1_reg : !hw.struct<address: i0, data: i32>
  %data1_reg = seq.compreg %20, %clock, %reset, %16  : !hw.struct<address: i0, data: i32>
  %ready1_reg = seq.compreg %28, %clock, %reset, %false  : i1
  %21 = comb.mux %ready1_reg, %ready1_reg, %valid1_reg : i1
  %22 = comb.xor %ready1_reg, %true : i1
  %23 = comb.xor %ready, %true : i1
  %24 = comb.and %23, %22 : i1
  %25 = comb.mux %24, %valid1_reg, %ready1_reg : i1
  %26 = comb.and %ready, %ready1_reg : i1
  %27 = comb.xor %26, %true : i1
  %28 = comb.and %27, %25 : i1
  %ctrl_data1_reg = seq.compreg %31, %clock, %reset, %16  : !hw.struct<address: i0, data: i32>
  %29 = comb.mux %ready1_reg, %ctrl_data1_reg, %data1_reg : !hw.struct<address: i0, data: i32>
  %30 = comb.mux %24, %data1_reg, %ctrl_data1_reg : !hw.struct<address: i0, data: i32>
  %31 = comb.mux %26, %16, %30 : !hw.struct<address: i0, data: i32>
  hw.output %chanOutput : !esi.channel<!hw.struct<address: i0, data: i32>>
}
hw.module @handshake_fork_in_ui32_out_ui32_ui32(%in0: !esi.channel<i32>, %clock: i1, %reset: i1) -> (out0: !esi.channel<i32>, out1: !esi.channel<i32>) {
  %true = hw.constant true
  %false = hw.constant false
  %rawOutput, %valid = esi.unwrap.vr %in0, %12 : i32
  %chanOutput, %ready = esi.wrap.vr %rawOutput, %3 : i32
  %chanOutput_0, %ready_1 = esi.wrap.vr %rawOutput, %9 : i32
  %0 = comb.xor %12, %true : i1
  %1 = comb.and %5, %0 : i1
  %emitted_0 = seq.compreg %1, %clock, %reset, %false  : i1
  %2 = comb.xor %emitted_0, %true : i1
  %3 = comb.and %2, %valid : i1
  %4 = comb.and %ready, %3 : i1
  %5 = comb.or %4, %emitted_0 {sv.namehint = "done0"} : i1
  %6 = comb.xor %12, %true : i1
  %7 = comb.and %11, %6 : i1
  %emitted_1 = seq.compreg %7, %clock, %reset, %false  : i1
  %8 = comb.xor %emitted_1, %true : i1
  %9 = comb.and %8, %valid : i1
  %10 = comb.and %ready_1, %9 : i1
  %11 = comb.or %10, %emitted_1 {sv.namehint = "done1"} : i1
  %12 = comb.and %5, %11 {sv.namehint = "allDone"} : i1
  hw.output %chanOutput, %chanOutput_0 : !esi.channel<i32>, !esi.channel<i32>
}
hw.module @handshake_join_in_ui32_1ins_1outs_ctrl(%in0: !esi.channel<i32>) -> (out0: !esi.channel<i0>) {
  %c0_i0 = hw.constant 0 : i0
  %rawOutput, %valid = esi.unwrap.vr %in0, %0 : i32
  %chanOutput, %ready = esi.wrap.vr %c0_i0, %valid : i0
  %0 = comb.and %ready, %valid : i1
  hw.output %chanOutput : !esi.channel<i0>
}
hw.module @arith_index_cast_in_ui64_out_ui3(%in0: !esi.channel<i64>) -> (out0: !esi.channel<i3>) {
  %rawOutput, %valid = esi.unwrap.vr %in0, %0 : i64
  %chanOutput, %ready = esi.wrap.vr %1, %valid : i3
  %0 = comb.and %ready, %valid : i1
  %1 = comb.extract %rawOutput from 0 : (i64) -> i3
  hw.output %chanOutput : !esi.channel<i3>
}
hw.module @handshake_buffer_in_ui3_out_ui3_2slots_seq(%in0: !esi.channel<i3>, %clock: i1, %reset: i1) -> (out0: !esi.channel<i3>) {
  %true = hw.constant true
  %false = hw.constant false
  %c0_i3 = hw.constant 0 : i3
  %rawOutput, %valid = esi.unwrap.vr %in0, %1 : i3
  %chanOutput, %ready = esi.wrap.vr %27, %19 : i3
  %valid0_reg = seq.compreg %2, %clock, %reset, %false  : i1
  %0 = comb.xor %valid0_reg, %true : i1
  %1 = comb.or %0, %5 : i1
  %2 = comb.mux %1, %valid, %valid0_reg : i1
  %3 = comb.mux %1, %rawOutput, %data0_reg : i3
  %data0_reg = seq.compreg %3, %clock, %reset, %c0_i3  : i3
  %ready0_reg = seq.compreg %11, %clock, %reset, %false  : i1
  %4 = comb.mux %ready0_reg, %ready0_reg, %valid0_reg : i1
  %5 = comb.xor %ready0_reg, %true : i1
  %6 = comb.xor %16, %true : i1
  %7 = comb.and %6, %5 : i1
  %8 = comb.mux %7, %valid0_reg, %ready0_reg : i1
  %9 = comb.and %16, %ready0_reg : i1
  %10 = comb.xor %9, %true : i1
  %11 = comb.and %10, %8 : i1
  %ctrl_data0_reg = seq.compreg %14, %clock, %reset, %c0_i3  : i3
  %12 = comb.mux %ready0_reg, %ctrl_data0_reg, %data0_reg : i3
  %13 = comb.mux %7, %data0_reg, %ctrl_data0_reg : i3
  %14 = comb.mux %9, %c0_i3, %13 : i3
  %valid1_reg = seq.compreg %17, %clock, %reset, %false  : i1
  %15 = comb.xor %valid1_reg, %true : i1
  %16 = comb.or %15, %20 : i1
  %17 = comb.mux %16, %4, %valid1_reg : i1
  %18 = comb.mux %16, %12, %data1_reg : i3
  %data1_reg = seq.compreg %18, %clock, %reset, %c0_i3  : i3
  %ready1_reg = seq.compreg %26, %clock, %reset, %false  : i1
  %19 = comb.mux %ready1_reg, %ready1_reg, %valid1_reg : i1
  %20 = comb.xor %ready1_reg, %true : i1
  %21 = comb.xor %ready, %true : i1
  %22 = comb.and %21, %20 : i1
  %23 = comb.mux %22, %valid1_reg, %ready1_reg : i1
  %24 = comb.and %ready, %ready1_reg : i1
  %25 = comb.xor %24, %true : i1
  %26 = comb.and %25, %23 : i1
  %ctrl_data1_reg = seq.compreg %29, %clock, %reset, %c0_i3  : i3
  %27 = comb.mux %ready1_reg, %ctrl_data1_reg, %data1_reg : i3
  %28 = comb.mux %22, %data1_reg, %ctrl_data1_reg : i3
  %29 = comb.mux %24, %c0_i3, %28 : i3
  hw.output %chanOutput : !esi.channel<i3>
}
hw.module @handshake_fork_1ins_7outs_ctrl(%in0: !esi.channel<i0>, %clock: i1, %reset: i1) -> (out0: !esi.channel<i0>, out1: !esi.channel<i0>, out2: !esi.channel<i0>, out3: !esi.channel<i0>, out4: !esi.channel<i0>, out5: !esi.channel<i0>, out6: !esi.channel<i0>) {
  %true = hw.constant true
  %false = hw.constant false
  %rawOutput, %valid = esi.unwrap.vr %in0, %42 : i0
  %chanOutput, %ready = esi.wrap.vr %rawOutput, %3 : i0
  %chanOutput_0, %ready_1 = esi.wrap.vr %rawOutput, %9 : i0
  %chanOutput_2, %ready_3 = esi.wrap.vr %rawOutput, %15 : i0
  %chanOutput_4, %ready_5 = esi.wrap.vr %rawOutput, %21 : i0
  %chanOutput_6, %ready_7 = esi.wrap.vr %rawOutput, %27 : i0
  %chanOutput_8, %ready_9 = esi.wrap.vr %rawOutput, %33 : i0
  %chanOutput_10, %ready_11 = esi.wrap.vr %rawOutput, %39 : i0
  %0 = comb.xor %42, %true : i1
  %1 = comb.and %5, %0 : i1
  %emitted_0 = seq.compreg %1, %clock, %reset, %false  : i1
  %2 = comb.xor %emitted_0, %true : i1
  %3 = comb.and %2, %valid : i1
  %4 = comb.and %ready, %3 : i1
  %5 = comb.or %4, %emitted_0 {sv.namehint = "done0"} : i1
  %6 = comb.xor %42, %true : i1
  %7 = comb.and %11, %6 : i1
  %emitted_1 = seq.compreg %7, %clock, %reset, %false  : i1
  %8 = comb.xor %emitted_1, %true : i1
  %9 = comb.and %8, %valid : i1
  %10 = comb.and %ready_1, %9 : i1
  %11 = comb.or %10, %emitted_1 {sv.namehint = "done1"} : i1
  %12 = comb.xor %42, %true : i1
  %13 = comb.and %17, %12 : i1
  %emitted_2 = seq.compreg %13, %clock, %reset, %false  : i1
  %14 = comb.xor %emitted_2, %true : i1
  %15 = comb.and %14, %valid : i1
  %16 = comb.and %ready_3, %15 : i1
  %17 = comb.or %16, %emitted_2 {sv.namehint = "done2"} : i1
  %18 = comb.xor %42, %true : i1
  %19 = comb.and %23, %18 : i1
  %emitted_3 = seq.compreg %19, %clock, %reset, %false  : i1
  %20 = comb.xor %emitted_3, %true : i1
  %21 = comb.and %20, %valid : i1
  %22 = comb.and %ready_5, %21 : i1
  %23 = comb.or %22, %emitted_3 {sv.namehint = "done3"} : i1
  %24 = comb.xor %42, %true : i1
  %25 = comb.and %29, %24 : i1
  %emitted_4 = seq.compreg %25, %clock, %reset, %false  : i1
  %26 = comb.xor %emitted_4, %true : i1
  %27 = comb.and %26, %valid : i1
  %28 = comb.and %ready_7, %27 : i1
  %29 = comb.or %28, %emitted_4 {sv.namehint = "done4"} : i1
  %30 = comb.xor %42, %true : i1
  %31 = comb.and %35, %30 : i1
  %emitted_5 = seq.compreg %31, %clock, %reset, %false  : i1
  %32 = comb.xor %emitted_5, %true : i1
  %33 = comb.and %32, %valid : i1
  %34 = comb.and %ready_9, %33 : i1
  %35 = comb.or %34, %emitted_5 {sv.namehint = "done5"} : i1
  %36 = comb.xor %42, %true : i1
  %37 = comb.and %41, %36 : i1
  %emitted_6 = seq.compreg %37, %clock, %reset, %false  : i1
  %38 = comb.xor %emitted_6, %true : i1
  %39 = comb.and %38, %valid : i1
  %40 = comb.and %ready_11, %39 : i1
  %41 = comb.or %40, %emitted_6 {sv.namehint = "done6"} : i1
  %42 = comb.and %5, %11, %17, %23, %29, %35, %41 {sv.namehint = "allDone"} : i1
  hw.output %chanOutput, %chanOutput_0, %chanOutput_2, %chanOutput_4, %chanOutput_6, %chanOutput_8, %chanOutput_10 : !esi.channel<i0>, !esi.channel<i0>, !esi.channel<i0>, !esi.channel<i0>, !esi.channel<i0>, !esi.channel<i0>, !esi.channel<i0>
}
hw.module @handshake_join_2ins_1outs_ctrl(%in0: !esi.channel<i0>, %in1: !esi.channel<i0>) -> (out0: !esi.channel<i0>) {
  %c0_i0 = hw.constant 0 : i0
  %rawOutput, %valid = esi.unwrap.vr %in0, %1 : i0
  %rawOutput_0, %valid_1 = esi.unwrap.vr %in1, %1 : i0
  %chanOutput, %ready = esi.wrap.vr %c0_i0, %0 : i0
  %0 = comb.and %valid, %valid_1 : i1
  %1 = comb.and %ready, %0 : i1
  hw.output %chanOutput : !esi.channel<i0>
}
hw.module @handshake_constant_c1_out_ui64(%ctrl: !esi.channel<i0>) -> (out0: !esi.channel<i64>) {
  %c1_i64 = hw.constant 1 : i64
  %rawOutput, %valid = esi.unwrap.vr %ctrl, %ready : i0
  %chanOutput, %ready = esi.wrap.vr %c1_i64, %valid : i64
  hw.output %chanOutput : !esi.channel<i64>
}
hw.module @handshake_buffer_in_ui64_out_ui64_2slots_seq(%in0: !esi.channel<i64>, %clock: i1, %reset: i1) -> (out0: !esi.channel<i64>) {
  %true = hw.constant true
  %false = hw.constant false
  %c0_i64 = hw.constant 0 : i64
  %rawOutput, %valid = esi.unwrap.vr %in0, %1 : i64
  %chanOutput, %ready = esi.wrap.vr %27, %19 : i64
  %valid0_reg = seq.compreg %2, %clock, %reset, %false  : i1
  %0 = comb.xor %valid0_reg, %true : i1
  %1 = comb.or %0, %5 : i1
  %2 = comb.mux %1, %valid, %valid0_reg : i1
  %3 = comb.mux %1, %rawOutput, %data0_reg : i64
  %data0_reg = seq.compreg %3, %clock, %reset, %c0_i64  : i64
  %ready0_reg = seq.compreg %11, %clock, %reset, %false  : i1
  %4 = comb.mux %ready0_reg, %ready0_reg, %valid0_reg : i1
  %5 = comb.xor %ready0_reg, %true : i1
  %6 = comb.xor %16, %true : i1
  %7 = comb.and %6, %5 : i1
  %8 = comb.mux %7, %valid0_reg, %ready0_reg : i1
  %9 = comb.and %16, %ready0_reg : i1
  %10 = comb.xor %9, %true : i1
  %11 = comb.and %10, %8 : i1
  %ctrl_data0_reg = seq.compreg %14, %clock, %reset, %c0_i64  : i64
  %12 = comb.mux %ready0_reg, %ctrl_data0_reg, %data0_reg : i64
  %13 = comb.mux %7, %data0_reg, %ctrl_data0_reg : i64
  %14 = comb.mux %9, %c0_i64, %13 : i64
  %valid1_reg = seq.compreg %17, %clock, %reset, %false  : i1
  %15 = comb.xor %valid1_reg, %true : i1
  %16 = comb.or %15, %20 : i1
  %17 = comb.mux %16, %4, %valid1_reg : i1
  %18 = comb.mux %16, %12, %data1_reg : i64
  %data1_reg = seq.compreg %18, %clock, %reset, %c0_i64  : i64
  %ready1_reg = seq.compreg %26, %clock, %reset, %false  : i1
  %19 = comb.mux %ready1_reg, %ready1_reg, %valid1_reg : i1
  %20 = comb.xor %ready1_reg, %true : i1
  %21 = comb.xor %ready, %true : i1
  %22 = comb.and %21, %20 : i1
  %23 = comb.mux %22, %valid1_reg, %ready1_reg : i1
  %24 = comb.and %ready, %ready1_reg : i1
  %25 = comb.xor %24, %true : i1
  %26 = comb.and %25, %23 : i1
  %ctrl_data1_reg = seq.compreg %29, %clock, %reset, %c0_i64  : i64
  %27 = comb.mux %ready1_reg, %ctrl_data1_reg, %data1_reg : i64
  %28 = comb.mux %22, %data1_reg, %ctrl_data1_reg : i64
  %29 = comb.mux %24, %c0_i64, %28 : i64
  hw.output %chanOutput : !esi.channel<i64>
}
hw.module @handshake_constant_c5_out_ui64(%ctrl: !esi.channel<i0>) -> (out0: !esi.channel<i64>) {
  %c5_i64 = hw.constant 5 : i64
  %rawOutput, %valid = esi.unwrap.vr %ctrl, %ready : i0
  %chanOutput, %ready = esi.wrap.vr %c5_i64, %valid : i64
  hw.output %chanOutput : !esi.channel<i64>
}
hw.module @handshake_constant_c0_out_ui64(%ctrl: !esi.channel<i0>) -> (out0: !esi.channel<i64>) {
  %c0_i64 = hw.constant 0 : i64
  %rawOutput, %valid = esi.unwrap.vr %ctrl, %ready : i0
  %chanOutput, %ready = esi.wrap.vr %c0_i64, %valid : i64
  hw.output %chanOutput : !esi.channel<i64>
}
hw.module @handshake_constant_c0_out_ui32(%ctrl: !esi.channel<i0>) -> (out0: !esi.channel<i32>) {
  %c0_i32 = hw.constant 0 : i32
  %rawOutput, %valid = esi.unwrap.vr %ctrl, %ready : i0
  %chanOutput, %ready = esi.wrap.vr %c0_i32, %valid : i32
  hw.output %chanOutput : !esi.channel<i32>
}
hw.module @handshake_store_in_ui64_ui32_out_ui32_ui64(%addrIn0: !esi.channel<i64>, %dataIn: !esi.channel<i32>, %ctrl: !esi.channel<i0>) -> (dataToMem: !esi.channel<i32>, addrOut0: !esi.channel<i64>) {
  %rawOutput, %valid = esi.unwrap.vr %addrIn0, %1 : i64
  %rawOutput_0, %valid_1 = esi.unwrap.vr %dataIn, %1 : i32
  %rawOutput_2, %valid_3 = esi.unwrap.vr %ctrl, %1 : i0
  %chanOutput, %ready = esi.wrap.vr %rawOutput_0, %0 : i32
  %chanOutput_4, %ready_5 = esi.wrap.vr %rawOutput, %0 : i64
  %0 = comb.and %valid_1, %valid, %valid_3 : i1
  %1 = comb.and %ready, %ready_5, %0 : i1
  hw.output %chanOutput, %chanOutput_4 : !esi.channel<i32>, !esi.channel<i64>
}
hw.module @handshake_buffer_in_ui1_out_ui1_1slots_seq_init_0(%in0: !esi.channel<i1>, %clock: i1, %reset: i1) -> (out0: !esi.channel<i1>) {
  %true = hw.constant true
  %false = hw.constant false
  %rawOutput, %valid = esi.unwrap.vr %in0, %1 : i1
  %chanOutput, %ready = esi.wrap.vr %12, %4 : i1
  %valid0_reg = seq.compreg %2, %clock, %reset, %true  : i1
  %0 = comb.xor %valid0_reg, %true : i1
  %1 = comb.or %0, %5 : i1
  %2 = comb.mux %1, %valid, %valid0_reg : i1
  %3 = comb.mux %1, %rawOutput, %data0_reg : i1
  %data0_reg = seq.compreg %3, %clock, %reset, %false  : i1
  %ready0_reg = seq.compreg %11, %clock, %reset, %false  : i1
  %4 = comb.mux %ready0_reg, %ready0_reg, %valid0_reg : i1
  %5 = comb.xor %ready0_reg, %true : i1
  %6 = comb.xor %ready, %true : i1
  %7 = comb.and %6, %5 : i1
  %8 = comb.mux %7, %valid0_reg, %ready0_reg : i1
  %9 = comb.and %ready, %ready0_reg : i1
  %10 = comb.xor %9, %true : i1
  %11 = comb.and %10, %8 : i1
  %ctrl_data0_reg = seq.compreg %15, %clock, %reset, %false  : i1
  %12 = comb.mux %ready0_reg, %ctrl_data0_reg, %data0_reg : i1
  %13 = comb.mux %7, %data0_reg, %ctrl_data0_reg : i1
  %14 = comb.xor %9, %true : i1
  %15 = comb.and %14, %13 : i1
  hw.output %chanOutput : !esi.channel<i1>
}
hw.module @handshake_fork_in_ui1_out_ui1_ui1_ui1_ui1(%in0: !esi.channel<i1>, %clock: i1, %reset: i1) -> (out0: !esi.channel<i1>, out1: !esi.channel<i1>, out2: !esi.channel<i1>, out3: !esi.channel<i1>) {
  %true = hw.constant true
  %false = hw.constant false
  %rawOutput, %valid = esi.unwrap.vr %in0, %24 : i1
  %chanOutput, %ready = esi.wrap.vr %rawOutput, %3 : i1
  %chanOutput_0, %ready_1 = esi.wrap.vr %rawOutput, %9 : i1
  %chanOutput_2, %ready_3 = esi.wrap.vr %rawOutput, %15 : i1
  %chanOutput_4, %ready_5 = esi.wrap.vr %rawOutput, %21 : i1
  %0 = comb.xor %24, %true : i1
  %1 = comb.and %5, %0 : i1
  %emitted_0 = seq.compreg %1, %clock, %reset, %false  : i1
  %2 = comb.xor %emitted_0, %true : i1
  %3 = comb.and %2, %valid : i1
  %4 = comb.and %ready, %3 : i1
  %5 = comb.or %4, %emitted_0 {sv.namehint = "done0"} : i1
  %6 = comb.xor %24, %true : i1
  %7 = comb.and %11, %6 : i1
  %emitted_1 = seq.compreg %7, %clock, %reset, %false  : i1
  %8 = comb.xor %emitted_1, %true : i1
  %9 = comb.and %8, %valid : i1
  %10 = comb.and %ready_1, %9 : i1
  %11 = comb.or %10, %emitted_1 {sv.namehint = "done1"} : i1
  %12 = comb.xor %24, %true : i1
  %13 = comb.and %17, %12 : i1
  %emitted_2 = seq.compreg %13, %clock, %reset, %false  : i1
  %14 = comb.xor %emitted_2, %true : i1
  %15 = comb.and %14, %valid : i1
  %16 = comb.and %ready_3, %15 : i1
  %17 = comb.or %16, %emitted_2 {sv.namehint = "done2"} : i1
  %18 = comb.xor %24, %true : i1
  %19 = comb.and %23, %18 : i1
  %emitted_3 = seq.compreg %19, %clock, %reset, %false  : i1
  %20 = comb.xor %emitted_3, %true : i1
  %21 = comb.and %20, %valid : i1
  %22 = comb.and %ready_5, %21 : i1
  %23 = comb.or %22, %emitted_3 {sv.namehint = "done3"} : i1
  %24 = comb.and %5, %11, %17, %23 {sv.namehint = "allDone"} : i1
  hw.output %chanOutput, %chanOutput_0, %chanOutput_2, %chanOutput_4 : !esi.channel<i1>, !esi.channel<i1>, !esi.channel<i1>, !esi.channel<i1>
}
hw.module @handshake_buffer_in_ui1_out_ui1_2slots_seq(%in0: !esi.channel<i1>, %clock: i1, %reset: i1) -> (out0: !esi.channel<i1>) {
  %true = hw.constant true
  %false = hw.constant false
  %rawOutput, %valid = esi.unwrap.vr %in0, %1 : i1
  %chanOutput, %ready = esi.wrap.vr %28, %20 : i1
  %valid0_reg = seq.compreg %2, %clock, %reset, %false  : i1
  %0 = comb.xor %valid0_reg, %true : i1
  %1 = comb.or %0, %5 : i1
  %2 = comb.mux %1, %valid, %valid0_reg : i1
  %3 = comb.mux %1, %rawOutput, %data0_reg : i1
  %data0_reg = seq.compreg %3, %clock, %reset, %false  : i1
  %ready0_reg = seq.compreg %11, %clock, %reset, %false  : i1
  %4 = comb.mux %ready0_reg, %ready0_reg, %valid0_reg : i1
  %5 = comb.xor %ready0_reg, %true : i1
  %6 = comb.xor %17, %true : i1
  %7 = comb.and %6, %5 : i1
  %8 = comb.mux %7, %valid0_reg, %ready0_reg : i1
  %9 = comb.and %17, %ready0_reg : i1
  %10 = comb.xor %9, %true : i1
  %11 = comb.and %10, %8 : i1
  %ctrl_data0_reg = seq.compreg %15, %clock, %reset, %false  : i1
  %12 = comb.mux %ready0_reg, %ctrl_data0_reg, %data0_reg : i1
  %13 = comb.mux %7, %data0_reg, %ctrl_data0_reg : i1
  %14 = comb.xor %9, %true : i1
  %15 = comb.and %14, %13 : i1
  %valid1_reg = seq.compreg %18, %clock, %reset, %false  : i1
  %16 = comb.xor %valid1_reg, %true : i1
  %17 = comb.or %16, %21 : i1
  %18 = comb.mux %17, %4, %valid1_reg : i1
  %19 = comb.mux %17, %12, %data1_reg : i1
  %data1_reg = seq.compreg %19, %clock, %reset, %false  : i1
  %ready1_reg = seq.compreg %27, %clock, %reset, %false  : i1
  %20 = comb.mux %ready1_reg, %ready1_reg, %valid1_reg : i1
  %21 = comb.xor %ready1_reg, %true : i1
  %22 = comb.xor %ready, %true : i1
  %23 = comb.and %22, %21 : i1
  %24 = comb.mux %23, %valid1_reg, %ready1_reg : i1
  %25 = comb.and %ready, %ready1_reg : i1
  %26 = comb.xor %25, %true : i1
  %27 = comb.and %26, %24 : i1
  %ctrl_data1_reg = seq.compreg %31, %clock, %reset, %false  : i1
  %28 = comb.mux %ready1_reg, %ctrl_data1_reg, %data1_reg : i1
  %29 = comb.mux %23, %data1_reg, %ctrl_data1_reg : i1
  %30 = comb.xor %25, %true : i1
  %31 = comb.and %30, %29 : i1
  hw.output %chanOutput : !esi.channel<i1>
}
hw.module @handshake_mux_in_ui1_3ins_1outs_ctrl(%select: !esi.channel<i1>, %in0: !esi.channel<i0>, %in1: !esi.channel<i0>) -> (out0: !esi.channel<i0>) {
  %true = hw.constant true
  %rawOutput, %valid = esi.unwrap.vr %select, %2 : i1
  %rawOutput_0, %valid_1 = esi.unwrap.vr %in0, %4 : i0
  %rawOutput_2, %valid_3 = esi.unwrap.vr %in1, %5 : i0
  %chanOutput, %ready = esi.wrap.vr %6, %1 : i0
  %0 = comb.mux %rawOutput, %valid_3, %valid_1 : i1
  %1 = comb.and %0, %valid : i1
  %2 = comb.and %1, %ready : i1
  %3 = comb.xor %rawOutput, %true : i1
  %4 = comb.and %3, %2 : i1
  %5 = comb.and %rawOutput, %2 : i1
  %6 = comb.mux %rawOutput, %rawOutput_2, %rawOutput_0 : i0
  hw.output %chanOutput : !esi.channel<i0>
}
hw.module @handshake_mux_in_ui1_ui64_ui64_out_ui64(%select: !esi.channel<i1>, %in0: !esi.channel<i64>, %in1: !esi.channel<i64>) -> (out0: !esi.channel<i64>) {
  %true = hw.constant true
  %rawOutput, %valid = esi.unwrap.vr %select, %2 : i1
  %rawOutput_0, %valid_1 = esi.unwrap.vr %in0, %4 : i64
  %rawOutput_2, %valid_3 = esi.unwrap.vr %in1, %5 : i64
  %chanOutput, %ready = esi.wrap.vr %6, %1 : i64
  %0 = comb.mux %rawOutput, %valid_3, %valid_1 : i1
  %1 = comb.and %0, %valid : i1
  %2 = comb.and %1, %ready : i1
  %3 = comb.xor %rawOutput, %true : i1
  %4 = comb.and %3, %2 : i1
  %5 = comb.and %rawOutput, %2 : i1
  %6 = comb.mux %rawOutput, %rawOutput_2, %rawOutput_0 : i64
  hw.output %chanOutput : !esi.channel<i64>
}
hw.module @handshake_fork_in_ui64_out_ui64_ui64(%in0: !esi.channel<i64>, %clock: i1, %reset: i1) -> (out0: !esi.channel<i64>, out1: !esi.channel<i64>) {
  %true = hw.constant true
  %false = hw.constant false
  %rawOutput, %valid = esi.unwrap.vr %in0, %12 : i64
  %chanOutput, %ready = esi.wrap.vr %rawOutput, %3 : i64
  %chanOutput_0, %ready_1 = esi.wrap.vr %rawOutput, %9 : i64
  %0 = comb.xor %12, %true : i1
  %1 = comb.and %5, %0 : i1
  %emitted_0 = seq.compreg %1, %clock, %reset, %false  : i1
  %2 = comb.xor %emitted_0, %true : i1
  %3 = comb.and %2, %valid : i1
  %4 = comb.and %ready, %3 : i1
  %5 = comb.or %4, %emitted_0 {sv.namehint = "done0"} : i1
  %6 = comb.xor %12, %true : i1
  %7 = comb.and %11, %6 : i1
  %emitted_1 = seq.compreg %7, %clock, %reset, %false  : i1
  %8 = comb.xor %emitted_1, %true : i1
  %9 = comb.and %8, %valid : i1
  %10 = comb.and %ready_1, %9 : i1
  %11 = comb.or %10, %emitted_1 {sv.namehint = "done1"} : i1
  %12 = comb.and %5, %11 {sv.namehint = "allDone"} : i1
  hw.output %chanOutput, %chanOutput_0 : !esi.channel<i64>, !esi.channel<i64>
}
hw.module @arith_cmpi_in_ui64_ui64_out_ui1_slt(%in0: !esi.channel<i64>, %in1: !esi.channel<i64>) -> (out0: !esi.channel<i1>) {
  %rawOutput, %valid = esi.unwrap.vr %in0, %1 : i64
  %rawOutput_0, %valid_1 = esi.unwrap.vr %in1, %1 : i64
  %chanOutput, %ready = esi.wrap.vr %2, %0 : i1
  %0 = comb.and %valid, %valid_1 : i1
  %1 = comb.and %ready, %0 : i1
  %2 = comb.icmp slt %rawOutput, %rawOutput_0 : i64
  hw.output %chanOutput : !esi.channel<i1>
}
hw.module @handshake_fork_in_ui1_out_ui1_ui1_ui1_ui1_ui1(%in0: !esi.channel<i1>, %clock: i1, %reset: i1) -> (out0: !esi.channel<i1>, out1: !esi.channel<i1>, out2: !esi.channel<i1>, out3: !esi.channel<i1>, out4: !esi.channel<i1>) {
  %true = hw.constant true
  %false = hw.constant false
  %rawOutput, %valid = esi.unwrap.vr %in0, %30 : i1
  %chanOutput, %ready = esi.wrap.vr %rawOutput, %3 : i1
  %chanOutput_0, %ready_1 = esi.wrap.vr %rawOutput, %9 : i1
  %chanOutput_2, %ready_3 = esi.wrap.vr %rawOutput, %15 : i1
  %chanOutput_4, %ready_5 = esi.wrap.vr %rawOutput, %21 : i1
  %chanOutput_6, %ready_7 = esi.wrap.vr %rawOutput, %27 : i1
  %0 = comb.xor %30, %true : i1
  %1 = comb.and %5, %0 : i1
  %emitted_0 = seq.compreg %1, %clock, %reset, %false  : i1
  %2 = comb.xor %emitted_0, %true : i1
  %3 = comb.and %2, %valid : i1
  %4 = comb.and %ready, %3 : i1
  %5 = comb.or %4, %emitted_0 {sv.namehint = "done0"} : i1
  %6 = comb.xor %30, %true : i1
  %7 = comb.and %11, %6 : i1
  %emitted_1 = seq.compreg %7, %clock, %reset, %false  : i1
  %8 = comb.xor %emitted_1, %true : i1
  %9 = comb.and %8, %valid : i1
  %10 = comb.and %ready_1, %9 : i1
  %11 = comb.or %10, %emitted_1 {sv.namehint = "done1"} : i1
  %12 = comb.xor %30, %true : i1
  %13 = comb.and %17, %12 : i1
  %emitted_2 = seq.compreg %13, %clock, %reset, %false  : i1
  %14 = comb.xor %emitted_2, %true : i1
  %15 = comb.and %14, %valid : i1
  %16 = comb.and %ready_3, %15 : i1
  %17 = comb.or %16, %emitted_2 {sv.namehint = "done2"} : i1
  %18 = comb.xor %30, %true : i1
  %19 = comb.and %23, %18 : i1
  %emitted_3 = seq.compreg %19, %clock, %reset, %false  : i1
  %20 = comb.xor %emitted_3, %true : i1
  %21 = comb.and %20, %valid : i1
  %22 = comb.and %ready_5, %21 : i1
  %23 = comb.or %22, %emitted_3 {sv.namehint = "done3"} : i1
  %24 = comb.xor %30, %true : i1
  %25 = comb.and %29, %24 : i1
  %emitted_4 = seq.compreg %25, %clock, %reset, %false  : i1
  %26 = comb.xor %emitted_4, %true : i1
  %27 = comb.and %26, %valid : i1
  %28 = comb.and %ready_7, %27 : i1
  %29 = comb.or %28, %emitted_4 {sv.namehint = "done4"} : i1
  %30 = comb.and %5, %11, %17, %23, %29 {sv.namehint = "allDone"} : i1
  hw.output %chanOutput, %chanOutput_0, %chanOutput_2, %chanOutput_4, %chanOutput_6 : !esi.channel<i1>, !esi.channel<i1>, !esi.channel<i1>, !esi.channel<i1>, !esi.channel<i1>
}
hw.module @handshake_cond_br_in_ui1_ui64_out_ui64_ui64(%cond: !esi.channel<i1>, %data: !esi.channel<i64>) -> (outTrue: !esi.channel<i64>, outFalse: !esi.channel<i64>) {
  %true = hw.constant true
  %rawOutput, %valid = esi.unwrap.vr %cond, %5 : i1
  %rawOutput_0, %valid_1 = esi.unwrap.vr %data, %5 : i64
  %chanOutput, %ready = esi.wrap.vr %rawOutput_0, %1 : i64
  %chanOutput_2, %ready_3 = esi.wrap.vr %rawOutput_0, %3 : i64
  %0 = comb.and %valid, %valid_1 : i1
  %1 = comb.and %rawOutput, %0 : i1
  %2 = comb.xor %rawOutput, %true : i1
  %3 = comb.and %2, %0 : i1
  %4 = comb.mux %rawOutput, %ready, %ready_3 : i1
  %5 = comb.and %4, %0 : i1
  hw.output %chanOutput, %chanOutput_2 : !esi.channel<i64>, !esi.channel<i64>
}
hw.module @handshake_sink_in_ui64(%in0: !esi.channel<i64>) {
  hw.output
}
hw.module @handshake_cond_br_in_ui1_2ins_2outs_ctrl(%cond: !esi.channel<i1>, %data: !esi.channel<i0>) -> (outTrue: !esi.channel<i0>, outFalse: !esi.channel<i0>) {
  %true = hw.constant true
  %rawOutput, %valid = esi.unwrap.vr %cond, %5 : i1
  %rawOutput_0, %valid_1 = esi.unwrap.vr %data, %5 : i0
  %chanOutput, %ready = esi.wrap.vr %rawOutput_0, %1 : i0
  %chanOutput_2, %ready_3 = esi.wrap.vr %rawOutput_0, %3 : i0
  %0 = comb.and %valid, %valid_1 : i1
  %1 = comb.and %rawOutput, %0 : i1
  %2 = comb.xor %rawOutput, %true : i1
  %3 = comb.and %2, %0 : i1
  %4 = comb.mux %rawOutput, %ready, %ready_3 : i1
  %5 = comb.and %4, %0 : i1
  hw.output %chanOutput, %chanOutput_2 : !esi.channel<i0>, !esi.channel<i0>
}
hw.module @handshake_fork_in_ui64_out_ui64_ui64_ui64(%in0: !esi.channel<i64>, %clock: i1, %reset: i1) -> (out0: !esi.channel<i64>, out1: !esi.channel<i64>, out2: !esi.channel<i64>) {
  %true = hw.constant true
  %false = hw.constant false
  %rawOutput, %valid = esi.unwrap.vr %in0, %18 : i64
  %chanOutput, %ready = esi.wrap.vr %rawOutput, %3 : i64
  %chanOutput_0, %ready_1 = esi.wrap.vr %rawOutput, %9 : i64
  %chanOutput_2, %ready_3 = esi.wrap.vr %rawOutput, %15 : i64
  %0 = comb.xor %18, %true : i1
  %1 = comb.and %5, %0 : i1
  %emitted_0 = seq.compreg %1, %clock, %reset, %false  : i1
  %2 = comb.xor %emitted_0, %true : i1
  %3 = comb.and %2, %valid : i1
  %4 = comb.and %ready, %3 : i1
  %5 = comb.or %4, %emitted_0 {sv.namehint = "done0"} : i1
  %6 = comb.xor %18, %true : i1
  %7 = comb.and %11, %6 : i1
  %emitted_1 = seq.compreg %7, %clock, %reset, %false  : i1
  %8 = comb.xor %emitted_1, %true : i1
  %9 = comb.and %8, %valid : i1
  %10 = comb.and %ready_1, %9 : i1
  %11 = comb.or %10, %emitted_1 {sv.namehint = "done1"} : i1
  %12 = comb.xor %18, %true : i1
  %13 = comb.and %17, %12 : i1
  %emitted_2 = seq.compreg %13, %clock, %reset, %false  : i1
  %14 = comb.xor %emitted_2, %true : i1
  %15 = comb.and %14, %valid : i1
  %16 = comb.and %ready_3, %15 : i1
  %17 = comb.or %16, %emitted_2 {sv.namehint = "done2"} : i1
  %18 = comb.and %5, %11, %17 {sv.namehint = "allDone"} : i1
  hw.output %chanOutput, %chanOutput_0, %chanOutput_2 : !esi.channel<i64>, !esi.channel<i64>, !esi.channel<i64>
}
hw.module @handshake_join_5ins_1outs_ctrl(%in0: !esi.channel<i0>, %in1: !esi.channel<i0>, %in2: !esi.channel<i0>, %in3: !esi.channel<i0>, %in4: !esi.channel<i0>) -> (out0: !esi.channel<i0>) {
  %c0_i0 = hw.constant 0 : i0
  %rawOutput, %valid = esi.unwrap.vr %in0, %1 : i0
  %rawOutput_0, %valid_1 = esi.unwrap.vr %in1, %1 : i0
  %rawOutput_2, %valid_3 = esi.unwrap.vr %in2, %1 : i0
  %rawOutput_4, %valid_5 = esi.unwrap.vr %in3, %1 : i0
  %rawOutput_6, %valid_7 = esi.unwrap.vr %in4, %1 : i0
  %chanOutput, %ready = esi.wrap.vr %c0_i0, %0 : i0
  %0 = comb.and %valid, %valid_1, %valid_3, %valid_5, %valid_7 : i1
  %1 = comb.and %ready, %0 : i1
  hw.output %chanOutput : !esi.channel<i0>
}
hw.module @handshake_load_in_ui64_ui32_out_ui32_ui64(%addrIn0: !esi.channel<i64>, %dataFromMem: !esi.channel<i32>, %ctrl: !esi.channel<i0>) -> (dataOut: !esi.channel<i32>, addrOut0: !esi.channel<i64>) {
  %rawOutput, %valid = esi.unwrap.vr %addrIn0, %1 : i64
  %rawOutput_0, %valid_1 = esi.unwrap.vr %dataFromMem, %ready : i32
  %rawOutput_2, %valid_3 = esi.unwrap.vr %ctrl, %1 : i0
  %chanOutput, %ready = esi.wrap.vr %rawOutput_0, %valid_1 : i32
  %chanOutput_4, %ready_5 = esi.wrap.vr %rawOutput, %0 : i64
  %0 = comb.and %valid, %valid_3 : i1
  %1 = comb.and %ready_5, %0 : i1
  hw.output %chanOutput, %chanOutput_4 : !esi.channel<i32>, !esi.channel<i64>
}
hw.module @arith_muli_in_ui32_ui32_out_ui32(%in0: !esi.channel<i32>, %in1: !esi.channel<i32>) -> (out0: !esi.channel<i32>) {
  %rawOutput, %valid = esi.unwrap.vr %in0, %1 : i32
  %rawOutput_0, %valid_1 = esi.unwrap.vr %in1, %1 : i32
  %chanOutput, %ready = esi.wrap.vr %2, %0 : i32
  %0 = comb.and %valid, %valid_1 : i1
  %1 = comb.and %ready, %0 : i1
  %2 = comb.mul %rawOutput, %rawOutput_0 : i32
  hw.output %chanOutput : !esi.channel<i32>
}
hw.module @arith_addi_in_ui32_ui32_out_ui32(%in0: !esi.channel<i32>, %in1: !esi.channel<i32>) -> (out0: !esi.channel<i32>) {
  %rawOutput, %valid = esi.unwrap.vr %in0, %1 : i32
  %rawOutput_0, %valid_1 = esi.unwrap.vr %in1, %1 : i32
  %chanOutput, %ready = esi.wrap.vr %2, %0 : i32
  %0 = comb.and %valid, %valid_1 : i1
  %1 = comb.and %ready, %0 : i1
  %2 = comb.add %rawOutput, %rawOutput_0 : i32
  hw.output %chanOutput : !esi.channel<i32>
}
hw.module @arith_addi_in_ui64_ui64_out_ui64(%in0: !esi.channel<i64>, %in1: !esi.channel<i64>) -> (out0: !esi.channel<i64>) {
  %rawOutput, %valid = esi.unwrap.vr %in0, %1 : i64
  %rawOutput_0, %valid_1 = esi.unwrap.vr %in1, %1 : i64
  %chanOutput, %ready = esi.wrap.vr %2, %0 : i64
  %0 = comb.and %valid, %valid_1 : i1
  %1 = comb.and %ready, %0 : i1
  %2 = comb.add %rawOutput, %rawOutput_0 : i64
  hw.output %chanOutput : !esi.channel<i64>
}
hw.module @handshake_fork_1ins_4outs_ctrl(%in0: !esi.channel<i0>, %clock: i1, %reset: i1) -> (out0: !esi.channel<i0>, out1: !esi.channel<i0>, out2: !esi.channel<i0>, out3: !esi.channel<i0>) {
  %true = hw.constant true
  %false = hw.constant false
  %rawOutput, %valid = esi.unwrap.vr %in0, %24 : i0
  %chanOutput, %ready = esi.wrap.vr %rawOutput, %3 : i0
  %chanOutput_0, %ready_1 = esi.wrap.vr %rawOutput, %9 : i0
  %chanOutput_2, %ready_3 = esi.wrap.vr %rawOutput, %15 : i0
  %chanOutput_4, %ready_5 = esi.wrap.vr %rawOutput, %21 : i0
  %0 = comb.xor %24, %true : i1
  %1 = comb.and %5, %0 : i1
  %emitted_0 = seq.compreg %1, %clock, %reset, %false  : i1
  %2 = comb.xor %emitted_0, %true : i1
  %3 = comb.and %2, %valid : i1
  %4 = comb.and %ready, %3 : i1
  %5 = comb.or %4, %emitted_0 {sv.namehint = "done0"} : i1
  %6 = comb.xor %24, %true : i1
  %7 = comb.and %11, %6 : i1
  %emitted_1 = seq.compreg %7, %clock, %reset, %false  : i1
  %8 = comb.xor %emitted_1, %true : i1
  %9 = comb.and %8, %valid : i1
  %10 = comb.and %ready_1, %9 : i1
  %11 = comb.or %10, %emitted_1 {sv.namehint = "done1"} : i1
  %12 = comb.xor %24, %true : i1
  %13 = comb.and %17, %12 : i1
  %emitted_2 = seq.compreg %13, %clock, %reset, %false  : i1
  %14 = comb.xor %emitted_2, %true : i1
  %15 = comb.and %14, %valid : i1
  %16 = comb.and %ready_3, %15 : i1
  %17 = comb.or %16, %emitted_2 {sv.namehint = "done2"} : i1
  %18 = comb.xor %24, %true : i1
  %19 = comb.and %23, %18 : i1
  %emitted_3 = seq.compreg %19, %clock, %reset, %false  : i1
  %20 = comb.xor %emitted_3, %true : i1
  %21 = comb.and %20, %valid : i1
  %22 = comb.and %ready_5, %21 : i1
  %23 = comb.or %22, %emitted_3 {sv.namehint = "done3"} : i1
  %24 = comb.and %5, %11, %17, %23 {sv.namehint = "allDone"} : i1
  hw.output %chanOutput, %chanOutput_0, %chanOutput_2, %chanOutput_4 : !esi.channel<i0>, !esi.channel<i0>, !esi.channel<i0>, !esi.channel<i0>
}
hw.module @handshake_join_3ins_1outs_ctrl(%in0: !esi.channel<i0>, %in1: !esi.channel<i0>, %in2: !esi.channel<i0>) -> (out0: !esi.channel<i0>) {
  %c0_i0 = hw.constant 0 : i0
  %rawOutput, %valid = esi.unwrap.vr %in0, %1 : i0
  %rawOutput_0, %valid_1 = esi.unwrap.vr %in1, %1 : i0
  %rawOutput_2, %valid_3 = esi.unwrap.vr %in2, %1 : i0
  %chanOutput, %ready = esi.wrap.vr %c0_i0, %0 : i0
  %0 = comb.and %valid, %valid_1, %valid_3 : i1
  %1 = comb.and %ready, %0 : i1
  hw.output %chanOutput : !esi.channel<i0>
}
esi.mem.ram @in0 i32 x 5
esi.mem.ram @in1 i32 x 5
esi.mem.ram @in2 i32 x 1
hw.module @forward_esi_wrapper(%in3: !esi.channel<i0>, %clock: i1, %reset: i1) -> (out0: !esi.channel<i0>) {
  %0 = esi.service.req.inout %forward.in2_st0 -> <@in2::@write>([]) : !esi.channel<!hw.struct<address: i0, data: i32>> -> !esi.channel<i0>
  %forward.out0, %forward.in0_ld0.addr, %forward.in1_ld0.addr, %forward.in2_st0 = hw.instance "forward" @forward(in0_ld0.data: %2: !esi.channel<i32>, in1_ld0.data: %1: !esi.channel<i32>, in2_st0.done: %0: !esi.channel<i0>, in3: %in3: !esi.channel<i0>, clock: %clock: i1, reset: %reset: i1) -> (out0: !esi.channel<i0>, in0_ld0.addr: !esi.channel<i3>, in1_ld0.addr: !esi.channel<i3>, in2_st0: !esi.channel<!hw.struct<address: i0, data: i32>>)
  %1 = esi.service.req.inout %forward.in1_ld0.addr -> <@in1::@read>([]) : !esi.channel<i3> -> !esi.channel<i32>
  %2 = esi.service.req.inout %forward.in0_ld0.addr -> <@in0::@read>([]) : !esi.channel<i3> -> !esi.channel<i32>
  hw.output %forward.out0 : !esi.channel<i0>
}
hw.module @forward(%in0_ld0.data: !esi.channel<i32>, %in1_ld0.data: !esi.channel<i32>, %in2_st0.done: !esi.channel<i0>, %in3: !esi.channel<i0>, %clock: i1, %reset: i1) -> (out0: !esi.channel<i0>, in0_ld0.addr: !esi.channel<i3>, in1_ld0.addr: !esi.channel<i3>, in2_st0: !esi.channel<!hw.struct<address: i0, data: i32>>) {
  %handshake_buffer0.out0 = hw.instance "handshake_buffer0" @handshake_buffer_2slots_seq_1ins_1outs_ctrl(in0: %in3: !esi.channel<i0>, clock: %clock: i1, reset: %reset: i1) -> (out0: !esi.channel<i0>)
  %handshake_buffer1.out0 = hw.instance "handshake_buffer1" @handshake_buffer_2slots_seq_1ins_1outs_ctrl(in0: %in2_st0.done: !esi.channel<i0>, clock: %clock: i1, reset: %reset: i1) -> (out0: !esi.channel<i0>)
  %handshake_buffer2.out0 = hw.instance "handshake_buffer2" @handshake_buffer_in_ui32_out_ui32_2slots_seq(in0: %in1_ld0.data: !esi.channel<i32>, clock: %clock: i1, reset: %reset: i1) -> (out0: !esi.channel<i32>)
  %handshake_buffer3.out0 = hw.instance "handshake_buffer3" @handshake_buffer_in_ui32_out_ui32_2slots_seq(in0: %in0_ld0.data: !esi.channel<i32>, clock: %clock: i1, reset: %reset: i1) -> (out0: !esi.channel<i32>)
  %handshake_memory0.ldData0, %handshake_memory0.ldData1, %handshake_memory0.stDone0, %handshake_memory0.stDone1, %handshake_memory0.ldDone0, %handshake_memory0.ldDone1 = hw.instance "handshake_memory0" @handshake_memory_out_ui32_id3(stData0: %handshake_buffer36.out0: !esi.channel<i32>, stAddr0: %handshake_buffer35.out0: !esi.channel<i64>, stData1: %handshake_buffer85.out0: !esi.channel<i32>, stAddr1: %handshake_buffer84.out0: !esi.channel<i64>, ldAddr0: %handshake_buffer78.out0: !esi.channel<i64>, ldAddr1: %handshake_buffer95.out0: !esi.channel<i64>, clock: %clock: i1, reset: %reset: i1) -> (ldData0: !esi.channel<i32>, ldData1: !esi.channel<i32>, stDone0: !esi.channel<i0>, stDone1: !esi.channel<i0>, ldDone0: !esi.channel<i0>, ldDone1: !esi.channel<i0>)
  %handshake_buffer4.out0 = hw.instance "handshake_buffer4" @handshake_buffer_2slots_seq_1ins_1outs_ctrl(in0: %handshake_memory0.ldDone1: !esi.channel<i0>, clock: %clock: i1, reset: %reset: i1) -> (out0: !esi.channel<i0>)
  %handshake_buffer5.out0 = hw.instance "handshake_buffer5" @handshake_buffer_2slots_seq_1ins_1outs_ctrl(in0: %handshake_memory0.ldDone0: !esi.channel<i0>, clock: %clock: i1, reset: %reset: i1) -> (out0: !esi.channel<i0>)
  %handshake_buffer6.out0 = hw.instance "handshake_buffer6" @handshake_buffer_2slots_seq_1ins_1outs_ctrl(in0: %handshake_memory0.stDone1: !esi.channel<i0>, clock: %clock: i1, reset: %reset: i1) -> (out0: !esi.channel<i0>)
  %handshake_buffer7.out0 = hw.instance "handshake_buffer7" @handshake_buffer_2slots_seq_1ins_1outs_ctrl(in0: %handshake_memory0.stDone0: !esi.channel<i0>, clock: %clock: i1, reset: %reset: i1) -> (out0: !esi.channel<i0>)
  %handshake_buffer8.out0 = hw.instance "handshake_buffer8" @handshake_buffer_in_ui32_out_ui32_2slots_seq(in0: %handshake_memory0.ldData1: !esi.channel<i32>, clock: %clock: i1, reset: %reset: i1) -> (out0: !esi.channel<i32>)
  %handshake_buffer9.out0 = hw.instance "handshake_buffer9" @handshake_buffer_in_ui32_out_ui32_2slots_seq(in0: %handshake_memory0.ldData0: !esi.channel<i32>, clock: %clock: i1, reset: %reset: i1) -> (out0: !esi.channel<i32>)
  %handshake_fork0.out0, %handshake_fork0.out1 = hw.instance "handshake_fork0" @handshake_fork_1ins_2outs_ctrl(in0: %handshake_buffer5.out0: !esi.channel<i0>, clock: %clock: i1, reset: %reset: i1) -> (out0: !esi.channel<i0>, out1: !esi.channel<i0>)
  %handshake_buffer10.out0 = hw.instance "handshake_buffer10" @handshake_buffer_2slots_seq_1ins_1outs_ctrl(in0: %handshake_fork0.out1: !esi.channel<i0>, clock: %clock: i1, reset: %reset: i1) -> (out0: !esi.channel<i0>)
  %handshake_buffer11.out0 = hw.instance "handshake_buffer11" @handshake_buffer_2slots_seq_1ins_1outs_ctrl(in0: %handshake_fork0.out0: !esi.channel<i0>, clock: %clock: i1, reset: %reset: i1) -> (out0: !esi.channel<i0>)
  %arith_index_cast0.out0 = hw.instance "arith_index_cast0" @arith_index_cast_in_ui64_out_ui0(in0: %handshake_buffer97.out0: !esi.channel<i64>) -> (out0: !esi.channel<i0>)
  %handshake_buffer12.out0 = hw.instance "handshake_buffer12" @handshake_buffer_in_ui0_out_ui0_2slots_seq(in0: %arith_index_cast0.out0: !esi.channel<i0>, clock: %clock: i1, reset: %reset: i1) -> (out0: !esi.channel<i0>)
  %hw_struct_create0.out0 = hw.instance "hw_struct_create0" @hw_struct_create_in_ui0_ui32_out_struct_address_ui0_data_ui32(in0: %handshake_buffer12.out0: !esi.channel<i0>, in1: %handshake_buffer98.out0: !esi.channel<i32>) -> (out0: !esi.channel<!hw.struct<address: i0, data: i32>>)
  %handshake_buffer13.out0 = hw.instance "handshake_buffer13" @handshake_buffer_in_struct_address_ui0_data_ui32_out_struct_address_ui0_data_ui32_2slots_seq(in0: %hw_struct_create0.out0: !esi.channel<!hw.struct<address: i0, data: i32>>, clock: %clock: i1, reset: %reset: i1) -> (out0: !esi.channel<!hw.struct<address: i0, data: i32>>)
  %handshake_fork1.out0, %handshake_fork1.out1 = hw.instance "handshake_fork1" @handshake_fork_in_ui32_out_ui32_ui32(in0: %handshake_buffer2.out0: !esi.channel<i32>, clock: %clock: i1, reset: %reset: i1) -> (out0: !esi.channel<i32>, out1: !esi.channel<i32>)
  %handshake_buffer14.out0 = hw.instance "handshake_buffer14" @handshake_buffer_in_ui32_out_ui32_2slots_seq(in0: %handshake_fork1.out1: !esi.channel<i32>, clock: %clock: i1, reset: %reset: i1) -> (out0: !esi.channel<i32>)
  %handshake_buffer15.out0 = hw.instance "handshake_buffer15" @handshake_buffer_in_ui32_out_ui32_2slots_seq(in0: %handshake_fork1.out0: !esi.channel<i32>, clock: %clock: i1, reset: %reset: i1) -> (out0: !esi.channel<i32>)
  %handshake_join0.out0 = hw.instance "handshake_join0" @handshake_join_in_ui32_1ins_1outs_ctrl(in0: %handshake_buffer14.out0: !esi.channel<i32>) -> (out0: !esi.channel<i0>)
  %handshake_buffer16.out0 = hw.instance "handshake_buffer16" @handshake_buffer_2slots_seq_1ins_1outs_ctrl(in0: %handshake_join0.out0: !esi.channel<i0>, clock: %clock: i1, reset: %reset: i1) -> (out0: !esi.channel<i0>)
  %arith_index_cast1.out0 = hw.instance "arith_index_cast1" @arith_index_cast_in_ui64_out_ui3(in0: %handshake_buffer75.out0: !esi.channel<i64>) -> (out0: !esi.channel<i3>)
  %handshake_buffer17.out0 = hw.instance "handshake_buffer17" @handshake_buffer_in_ui3_out_ui3_2slots_seq(in0: %arith_index_cast1.out0: !esi.channel<i3>, clock: %clock: i1, reset: %reset: i1) -> (out0: !esi.channel<i3>)
  %handshake_fork2.out0, %handshake_fork2.out1 = hw.instance "handshake_fork2" @handshake_fork_in_ui32_out_ui32_ui32(in0: %handshake_buffer3.out0: !esi.channel<i32>, clock: %clock: i1, reset: %reset: i1) -> (out0: !esi.channel<i32>, out1: !esi.channel<i32>)
  %handshake_buffer18.out0 = hw.instance "handshake_buffer18" @handshake_buffer_in_ui32_out_ui32_2slots_seq(in0: %handshake_fork2.out1: !esi.channel<i32>, clock: %clock: i1, reset: %reset: i1) -> (out0: !esi.channel<i32>)
  %handshake_buffer19.out0 = hw.instance "handshake_buffer19" @handshake_buffer_in_ui32_out_ui32_2slots_seq(in0: %handshake_fork2.out0: !esi.channel<i32>, clock: %clock: i1, reset: %reset: i1) -> (out0: !esi.channel<i32>)
  %handshake_join1.out0 = hw.instance "handshake_join1" @handshake_join_in_ui32_1ins_1outs_ctrl(in0: %handshake_buffer18.out0: !esi.channel<i32>) -> (out0: !esi.channel<i0>)
  %handshake_buffer20.out0 = hw.instance "handshake_buffer20" @handshake_buffer_2slots_seq_1ins_1outs_ctrl(in0: %handshake_join1.out0: !esi.channel<i0>, clock: %clock: i1, reset: %reset: i1) -> (out0: !esi.channel<i0>)
  %arith_index_cast2.out0 = hw.instance "arith_index_cast2" @arith_index_cast_in_ui64_out_ui3(in0: %handshake_buffer73.out0: !esi.channel<i64>) -> (out0: !esi.channel<i3>)
  %handshake_buffer21.out0 = hw.instance "handshake_buffer21" @handshake_buffer_in_ui3_out_ui3_2slots_seq(in0: %arith_index_cast2.out0: !esi.channel<i3>, clock: %clock: i1, reset: %reset: i1) -> (out0: !esi.channel<i3>)
  %handshake_fork3.out0, %handshake_fork3.out1, %handshake_fork3.out2, %handshake_fork3.out3, %handshake_fork3.out4, %handshake_fork3.out5, %handshake_fork3.out6 = hw.instance "handshake_fork3" @handshake_fork_1ins_7outs_ctrl(in0: %handshake_buffer0.out0: !esi.channel<i0>, clock: %clock: i1, reset: %reset: i1) -> (out0: !esi.channel<i0>, out1: !esi.channel<i0>, out2: !esi.channel<i0>, out3: !esi.channel<i0>, out4: !esi.channel<i0>, out5: !esi.channel<i0>, out6: !esi.channel<i0>)
  %handshake_buffer22.out0 = hw.instance "handshake_buffer22" @handshake_buffer_2slots_seq_1ins_1outs_ctrl(in0: %handshake_fork3.out6: !esi.channel<i0>, clock: %clock: i1, reset: %reset: i1) -> (out0: !esi.channel<i0>)
  %handshake_buffer23.out0 = hw.instance "handshake_buffer23" @handshake_buffer_2slots_seq_1ins_1outs_ctrl(in0: %handshake_fork3.out5: !esi.channel<i0>, clock: %clock: i1, reset: %reset: i1) -> (out0: !esi.channel<i0>)
  %handshake_buffer24.out0 = hw.instance "handshake_buffer24" @handshake_buffer_2slots_seq_1ins_1outs_ctrl(in0: %handshake_fork3.out4: !esi.channel<i0>, clock: %clock: i1, reset: %reset: i1) -> (out0: !esi.channel<i0>)
  %handshake_buffer25.out0 = hw.instance "handshake_buffer25" @handshake_buffer_2slots_seq_1ins_1outs_ctrl(in0: %handshake_fork3.out3: !esi.channel<i0>, clock: %clock: i1, reset: %reset: i1) -> (out0: !esi.channel<i0>)
  %handshake_buffer26.out0 = hw.instance "handshake_buffer26" @handshake_buffer_2slots_seq_1ins_1outs_ctrl(in0: %handshake_fork3.out2: !esi.channel<i0>, clock: %clock: i1, reset: %reset: i1) -> (out0: !esi.channel<i0>)
  %handshake_buffer27.out0 = hw.instance "handshake_buffer27" @handshake_buffer_2slots_seq_1ins_1outs_ctrl(in0: %handshake_fork3.out1: !esi.channel<i0>, clock: %clock: i1, reset: %reset: i1) -> (out0: !esi.channel<i0>)
  %handshake_buffer28.out0 = hw.instance "handshake_buffer28" @handshake_buffer_2slots_seq_1ins_1outs_ctrl(in0: %handshake_fork3.out0: !esi.channel<i0>, clock: %clock: i1, reset: %reset: i1) -> (out0: !esi.channel<i0>)
  %handshake_join2.out0 = hw.instance "handshake_join2" @handshake_join_2ins_1outs_ctrl(in0: %handshake_buffer22.out0: !esi.channel<i0>, in1: %handshake_buffer7.out0: !esi.channel<i0>) -> (out0: !esi.channel<i0>)
  %handshake_buffer29.out0 = hw.instance "handshake_buffer29" @handshake_buffer_2slots_seq_1ins_1outs_ctrl(in0: %handshake_join2.out0: !esi.channel<i0>, clock: %clock: i1, reset: %reset: i1) -> (out0: !esi.channel<i0>)
  %handshake_constant0.out0 = hw.instance "handshake_constant0" @handshake_constant_c1_out_ui64(ctrl: %handshake_buffer23.out0: !esi.channel<i0>) -> (out0: !esi.channel<i64>)
  %handshake_buffer30.out0 = hw.instance "handshake_buffer30" @handshake_buffer_in_ui64_out_ui64_2slots_seq(in0: %handshake_constant0.out0: !esi.channel<i64>, clock: %clock: i1, reset: %reset: i1) -> (out0: !esi.channel<i64>)
  %handshake_constant1.out0 = hw.instance "handshake_constant1" @handshake_constant_c5_out_ui64(ctrl: %handshake_buffer24.out0: !esi.channel<i0>) -> (out0: !esi.channel<i64>)
  %handshake_buffer31.out0 = hw.instance "handshake_buffer31" @handshake_buffer_in_ui64_out_ui64_2slots_seq(in0: %handshake_constant1.out0: !esi.channel<i64>, clock: %clock: i1, reset: %reset: i1) -> (out0: !esi.channel<i64>)
  %handshake_constant2.out0 = hw.instance "handshake_constant2" @handshake_constant_c0_out_ui64(ctrl: %handshake_buffer25.out0: !esi.channel<i0>) -> (out0: !esi.channel<i64>)
  %handshake_buffer32.out0 = hw.instance "handshake_buffer32" @handshake_buffer_in_ui64_out_ui64_2slots_seq(in0: %handshake_constant2.out0: !esi.channel<i64>, clock: %clock: i1, reset: %reset: i1) -> (out0: !esi.channel<i64>)
  %handshake_constant3.out0 = hw.instance "handshake_constant3" @handshake_constant_c0_out_ui32(ctrl: %handshake_buffer26.out0: !esi.channel<i0>) -> (out0: !esi.channel<i32>)
  %handshake_buffer33.out0 = hw.instance "handshake_buffer33" @handshake_buffer_in_ui32_out_ui32_2slots_seq(in0: %handshake_constant3.out0: !esi.channel<i32>, clock: %clock: i1, reset: %reset: i1) -> (out0: !esi.channel<i32>)
  %handshake_constant4.out0 = hw.instance "handshake_constant4" @handshake_constant_c0_out_ui64(ctrl: %handshake_buffer27.out0: !esi.channel<i0>) -> (out0: !esi.channel<i64>)
  %handshake_buffer34.out0 = hw.instance "handshake_buffer34" @handshake_buffer_in_ui64_out_ui64_2slots_seq(in0: %handshake_constant4.out0: !esi.channel<i64>, clock: %clock: i1, reset: %reset: i1) -> (out0: !esi.channel<i64>)
  %handshake_store0.dataToMem, %handshake_store0.addrOut0 = hw.instance "handshake_store0" @handshake_store_in_ui64_ui32_out_ui32_ui64(addrIn0: %handshake_buffer34.out0: !esi.channel<i64>, dataIn: %handshake_buffer33.out0: !esi.channel<i32>, ctrl: %handshake_buffer28.out0: !esi.channel<i0>) -> (dataToMem: !esi.channel<i32>, addrOut0: !esi.channel<i64>)
  %handshake_buffer35.out0 = hw.instance "handshake_buffer35" @handshake_buffer_in_ui64_out_ui64_2slots_seq(in0: %handshake_store0.addrOut0: !esi.channel<i64>, clock: %clock: i1, reset: %reset: i1) -> (out0: !esi.channel<i64>)
  %handshake_buffer36.out0 = hw.instance "handshake_buffer36" @handshake_buffer_in_ui32_out_ui32_2slots_seq(in0: %handshake_store0.dataToMem: !esi.channel<i32>, clock: %clock: i1, reset: %reset: i1) -> (out0: !esi.channel<i32>)
  %handshake_buffer37.out0 = hw.instance "handshake_buffer37" @handshake_buffer_in_ui1_out_ui1_1slots_seq_init_0(in0: %handshake_fork7.out0: !esi.channel<i1>, clock: %clock: i1, reset: %reset: i1) -> (out0: !esi.channel<i1>)
  %handshake_fork4.out0, %handshake_fork4.out1, %handshake_fork4.out2, %handshake_fork4.out3 = hw.instance "handshake_fork4" @handshake_fork_in_ui1_out_ui1_ui1_ui1_ui1(in0: %handshake_buffer37.out0: !esi.channel<i1>, clock: %clock: i1, reset: %reset: i1) -> (out0: !esi.channel<i1>, out1: !esi.channel<i1>, out2: !esi.channel<i1>, out3: !esi.channel<i1>)
  %handshake_buffer38.out0 = hw.instance "handshake_buffer38" @handshake_buffer_in_ui1_out_ui1_2slots_seq(in0: %handshake_fork4.out3: !esi.channel<i1>, clock: %clock: i1, reset: %reset: i1) -> (out0: !esi.channel<i1>)
  %handshake_buffer39.out0 = hw.instance "handshake_buffer39" @handshake_buffer_in_ui1_out_ui1_2slots_seq(in0: %handshake_fork4.out2: !esi.channel<i1>, clock: %clock: i1, reset: %reset: i1) -> (out0: !esi.channel<i1>)
  %handshake_buffer40.out0 = hw.instance "handshake_buffer40" @handshake_buffer_in_ui1_out_ui1_2slots_seq(in0: %handshake_fork4.out1: !esi.channel<i1>, clock: %clock: i1, reset: %reset: i1) -> (out0: !esi.channel<i1>)
  %handshake_buffer41.out0 = hw.instance "handshake_buffer41" @handshake_buffer_in_ui1_out_ui1_2slots_seq(in0: %handshake_fork4.out0: !esi.channel<i1>, clock: %clock: i1, reset: %reset: i1) -> (out0: !esi.channel<i1>)
  %handshake_mux0.out0 = hw.instance "handshake_mux0" @handshake_mux_in_ui1_3ins_1outs_ctrl(select: %handshake_buffer38.out0: !esi.channel<i1>, in0: %handshake_buffer29.out0: !esi.channel<i0>, in1: %handshake_buffer72.out0: !esi.channel<i0>) -> (out0: !esi.channel<i0>)
  %handshake_buffer42.out0 = hw.instance "handshake_buffer42" @handshake_buffer_2slots_seq_1ins_1outs_ctrl(in0: %handshake_mux0.out0: !esi.channel<i0>, clock: %clock: i1, reset: %reset: i1) -> (out0: !esi.channel<i0>)
  %handshake_mux1.out0 = hw.instance "handshake_mux1" @handshake_mux_in_ui1_ui64_ui64_out_ui64(select: %handshake_buffer39.out0: !esi.channel<i1>, in0: %handshake_buffer31.out0: !esi.channel<i64>, in1: %handshake_buffer55.out0: !esi.channel<i64>) -> (out0: !esi.channel<i64>)
  %handshake_buffer43.out0 = hw.instance "handshake_buffer43" @handshake_buffer_in_ui64_out_ui64_2slots_seq(in0: %handshake_mux1.out0: !esi.channel<i64>, clock: %clock: i1, reset: %reset: i1) -> (out0: !esi.channel<i64>)
  %handshake_fork5.out0, %handshake_fork5.out1 = hw.instance "handshake_fork5" @handshake_fork_in_ui64_out_ui64_ui64(in0: %handshake_buffer43.out0: !esi.channel<i64>, clock: %clock: i1, reset: %reset: i1) -> (out0: !esi.channel<i64>, out1: !esi.channel<i64>)
  %handshake_buffer44.out0 = hw.instance "handshake_buffer44" @handshake_buffer_in_ui64_out_ui64_2slots_seq(in0: %handshake_fork5.out1: !esi.channel<i64>, clock: %clock: i1, reset: %reset: i1) -> (out0: !esi.channel<i64>)
  %handshake_buffer45.out0 = hw.instance "handshake_buffer45" @handshake_buffer_in_ui64_out_ui64_2slots_seq(in0: %handshake_fork5.out0: !esi.channel<i64>, clock: %clock: i1, reset: %reset: i1) -> (out0: !esi.channel<i64>)
  %handshake_mux2.out0 = hw.instance "handshake_mux2" @handshake_mux_in_ui1_ui64_ui64_out_ui64(select: %handshake_buffer40.out0: !esi.channel<i1>, in0: %handshake_buffer30.out0: !esi.channel<i64>, in1: %handshake_buffer64.out0: !esi.channel<i64>) -> (out0: !esi.channel<i64>)
  %handshake_buffer46.out0 = hw.instance "handshake_buffer46" @handshake_buffer_in_ui64_out_ui64_2slots_seq(in0: %handshake_mux2.out0: !esi.channel<i64>, clock: %clock: i1, reset: %reset: i1) -> (out0: !esi.channel<i64>)
  %handshake_mux3.out0 = hw.instance "handshake_mux3" @handshake_mux_in_ui1_ui64_ui64_out_ui64(select: %handshake_buffer41.out0: !esi.channel<i1>, in0: %handshake_buffer32.out0: !esi.channel<i64>, in1: %handshake_buffer86.out0: !esi.channel<i64>) -> (out0: !esi.channel<i64>)
  %handshake_buffer47.out0 = hw.instance "handshake_buffer47" @handshake_buffer_in_ui64_out_ui64_2slots_seq(in0: %handshake_mux3.out0: !esi.channel<i64>, clock: %clock: i1, reset: %reset: i1) -> (out0: !esi.channel<i64>)
  %handshake_fork6.out0, %handshake_fork6.out1 = hw.instance "handshake_fork6" @handshake_fork_in_ui64_out_ui64_ui64(in0: %handshake_buffer47.out0: !esi.channel<i64>, clock: %clock: i1, reset: %reset: i1) -> (out0: !esi.channel<i64>, out1: !esi.channel<i64>)
  %handshake_buffer48.out0 = hw.instance "handshake_buffer48" @handshake_buffer_in_ui64_out_ui64_2slots_seq(in0: %handshake_fork6.out1: !esi.channel<i64>, clock: %clock: i1, reset: %reset: i1) -> (out0: !esi.channel<i64>)
  %handshake_buffer49.out0 = hw.instance "handshake_buffer49" @handshake_buffer_in_ui64_out_ui64_2slots_seq(in0: %handshake_fork6.out0: !esi.channel<i64>, clock: %clock: i1, reset: %reset: i1) -> (out0: !esi.channel<i64>)
  %arith_cmpi0.out0 = hw.instance "arith_cmpi0" @arith_cmpi_in_ui64_ui64_out_ui1_slt(in0: %handshake_buffer49.out0: !esi.channel<i64>, in1: %handshake_buffer45.out0: !esi.channel<i64>) -> (out0: !esi.channel<i1>)
  %handshake_buffer50.out0 = hw.instance "handshake_buffer50" @handshake_buffer_in_ui1_out_ui1_2slots_seq(in0: %arith_cmpi0.out0: !esi.channel<i1>, clock: %clock: i1, reset: %reset: i1) -> (out0: !esi.channel<i1>)
  %handshake_fork7.out0, %handshake_fork7.out1, %handshake_fork7.out2, %handshake_fork7.out3, %handshake_fork7.out4 = hw.instance "handshake_fork7" @handshake_fork_in_ui1_out_ui1_ui1_ui1_ui1_ui1(in0: %handshake_buffer50.out0: !esi.channel<i1>, clock: %clock: i1, reset: %reset: i1) -> (out0: !esi.channel<i1>, out1: !esi.channel<i1>, out2: !esi.channel<i1>, out3: !esi.channel<i1>, out4: !esi.channel<i1>)
  %handshake_buffer51.out0 = hw.instance "handshake_buffer51" @handshake_buffer_in_ui1_out_ui1_2slots_seq(in0: %handshake_fork7.out4: !esi.channel<i1>, clock: %clock: i1, reset: %reset: i1) -> (out0: !esi.channel<i1>)
  %handshake_buffer52.out0 = hw.instance "handshake_buffer52" @handshake_buffer_in_ui1_out_ui1_2slots_seq(in0: %handshake_fork7.out3: !esi.channel<i1>, clock: %clock: i1, reset: %reset: i1) -> (out0: !esi.channel<i1>)
  %handshake_buffer53.out0 = hw.instance "handshake_buffer53" @handshake_buffer_in_ui1_out_ui1_2slots_seq(in0: %handshake_fork7.out2: !esi.channel<i1>, clock: %clock: i1, reset: %reset: i1) -> (out0: !esi.channel<i1>)
  %handshake_buffer54.out0 = hw.instance "handshake_buffer54" @handshake_buffer_in_ui1_out_ui1_2slots_seq(in0: %handshake_fork7.out1: !esi.channel<i1>, clock: %clock: i1, reset: %reset: i1) -> (out0: !esi.channel<i1>)
  %handshake_cond_br0.outTrue, %handshake_cond_br0.outFalse = hw.instance "handshake_cond_br0" @handshake_cond_br_in_ui1_ui64_out_ui64_ui64(cond: %handshake_buffer51.out0: !esi.channel<i1>, data: %handshake_buffer44.out0: !esi.channel<i64>) -> (outTrue: !esi.channel<i64>, outFalse: !esi.channel<i64>)
  %handshake_buffer55.out0 = hw.instance "handshake_buffer55" @handshake_buffer_in_ui64_out_ui64_2slots_seq(in0: %handshake_cond_br0.outTrue: !esi.channel<i64>, clock: %clock: i1, reset: %reset: i1) -> (out0: !esi.channel<i64>)
  hw.instance "handshake_sink0" @handshake_sink_in_ui64(in0: %handshake_cond_br0.outFalse: !esi.channel<i64>) -> ()
  %handshake_cond_br1.outTrue, %handshake_cond_br1.outFalse = hw.instance "handshake_cond_br1" @handshake_cond_br_in_ui1_ui64_out_ui64_ui64(cond: %handshake_buffer52.out0: !esi.channel<i1>, data: %handshake_buffer46.out0: !esi.channel<i64>) -> (outTrue: !esi.channel<i64>, outFalse: !esi.channel<i64>)
  %handshake_buffer56.out0 = hw.instance "handshake_buffer56" @handshake_buffer_in_ui64_out_ui64_2slots_seq(in0: %handshake_cond_br1.outTrue: !esi.channel<i64>, clock: %clock: i1, reset: %reset: i1) -> (out0: !esi.channel<i64>)
  hw.instance "handshake_sink1" @handshake_sink_in_ui64(in0: %handshake_cond_br1.outFalse: !esi.channel<i64>) -> ()
  %handshake_cond_br2.outTrue, %handshake_cond_br2.outFalse = hw.instance "handshake_cond_br2" @handshake_cond_br_in_ui1_2ins_2outs_ctrl(cond: %handshake_buffer53.out0: !esi.channel<i1>, data: %handshake_buffer42.out0: !esi.channel<i0>) -> (outTrue: !esi.channel<i0>, outFalse: !esi.channel<i0>)
  %handshake_buffer57.out0 = hw.instance "handshake_buffer57" @handshake_buffer_2slots_seq_1ins_1outs_ctrl(in0: %handshake_cond_br2.outFalse: !esi.channel<i0>, clock: %clock: i1, reset: %reset: i1) -> (out0: !esi.channel<i0>)
  %handshake_buffer58.out0 = hw.instance "handshake_buffer58" @handshake_buffer_2slots_seq_1ins_1outs_ctrl(in0: %handshake_cond_br2.outTrue: !esi.channel<i0>, clock: %clock: i1, reset: %reset: i1) -> (out0: !esi.channel<i0>)
  %handshake_cond_br3.outTrue, %handshake_cond_br3.outFalse = hw.instance "handshake_cond_br3" @handshake_cond_br_in_ui1_ui64_out_ui64_ui64(cond: %handshake_buffer54.out0: !esi.channel<i1>, data: %handshake_buffer48.out0: !esi.channel<i64>) -> (outTrue: !esi.channel<i64>, outFalse: !esi.channel<i64>)
  %handshake_buffer59.out0 = hw.instance "handshake_buffer59" @handshake_buffer_in_ui64_out_ui64_2slots_seq(in0: %handshake_cond_br3.outTrue: !esi.channel<i64>, clock: %clock: i1, reset: %reset: i1) -> (out0: !esi.channel<i64>)
  hw.instance "handshake_sink2" @handshake_sink_in_ui64(in0: %handshake_cond_br3.outFalse: !esi.channel<i64>) -> ()
  %handshake_fork8.out0, %handshake_fork8.out1, %handshake_fork8.out2 = hw.instance "handshake_fork8" @handshake_fork_in_ui64_out_ui64_ui64_ui64(in0: %handshake_buffer59.out0: !esi.channel<i64>, clock: %clock: i1, reset: %reset: i1) -> (out0: !esi.channel<i64>, out1: !esi.channel<i64>, out2: !esi.channel<i64>)
  %handshake_buffer60.out0 = hw.instance "handshake_buffer60" @handshake_buffer_in_ui64_out_ui64_2slots_seq(in0: %handshake_fork8.out2: !esi.channel<i64>, clock: %clock: i1, reset: %reset: i1) -> (out0: !esi.channel<i64>)
  %handshake_buffer61.out0 = hw.instance "handshake_buffer61" @handshake_buffer_in_ui64_out_ui64_2slots_seq(in0: %handshake_fork8.out1: !esi.channel<i64>, clock: %clock: i1, reset: %reset: i1) -> (out0: !esi.channel<i64>)
  %handshake_buffer62.out0 = hw.instance "handshake_buffer62" @handshake_buffer_in_ui64_out_ui64_2slots_seq(in0: %handshake_fork8.out0: !esi.channel<i64>, clock: %clock: i1, reset: %reset: i1) -> (out0: !esi.channel<i64>)
  %handshake_fork9.out0, %handshake_fork9.out1 = hw.instance "handshake_fork9" @handshake_fork_in_ui64_out_ui64_ui64(in0: %handshake_buffer56.out0: !esi.channel<i64>, clock: %clock: i1, reset: %reset: i1) -> (out0: !esi.channel<i64>, out1: !esi.channel<i64>)
  %handshake_buffer63.out0 = hw.instance "handshake_buffer63" @handshake_buffer_in_ui64_out_ui64_2slots_seq(in0: %handshake_fork9.out1: !esi.channel<i64>, clock: %clock: i1, reset: %reset: i1) -> (out0: !esi.channel<i64>)
  %handshake_buffer64.out0 = hw.instance "handshake_buffer64" @handshake_buffer_in_ui64_out_ui64_2slots_seq(in0: %handshake_fork9.out0: !esi.channel<i64>, clock: %clock: i1, reset: %reset: i1) -> (out0: !esi.channel<i64>)
  %handshake_fork10.out0, %handshake_fork10.out1, %handshake_fork10.out2, %handshake_fork10.out3, %handshake_fork10.out4, %handshake_fork10.out5, %handshake_fork10.out6 = hw.instance "handshake_fork10" @handshake_fork_1ins_7outs_ctrl(in0: %handshake_buffer58.out0: !esi.channel<i0>, clock: %clock: i1, reset: %reset: i1) -> (out0: !esi.channel<i0>, out1: !esi.channel<i0>, out2: !esi.channel<i0>, out3: !esi.channel<i0>, out4: !esi.channel<i0>, out5: !esi.channel<i0>, out6: !esi.channel<i0>)
  %handshake_buffer65.out0 = hw.instance "handshake_buffer65" @handshake_buffer_2slots_seq_1ins_1outs_ctrl(in0: %handshake_fork10.out6: !esi.channel<i0>, clock: %clock: i1, reset: %reset: i1) -> (out0: !esi.channel<i0>)
  %handshake_buffer66.out0 = hw.instance "handshake_buffer66" @handshake_buffer_2slots_seq_1ins_1outs_ctrl(in0: %handshake_fork10.out5: !esi.channel<i0>, clock: %clock: i1, reset: %reset: i1) -> (out0: !esi.channel<i0>)
  %handshake_buffer67.out0 = hw.instance "handshake_buffer67" @handshake_buffer_2slots_seq_1ins_1outs_ctrl(in0: %handshake_fork10.out4: !esi.channel<i0>, clock: %clock: i1, reset: %reset: i1) -> (out0: !esi.channel<i0>)
  %handshake_buffer68.out0 = hw.instance "handshake_buffer68" @handshake_buffer_2slots_seq_1ins_1outs_ctrl(in0: %handshake_fork10.out3: !esi.channel<i0>, clock: %clock: i1, reset: %reset: i1) -> (out0: !esi.channel<i0>)
  %handshake_buffer69.out0 = hw.instance "handshake_buffer69" @handshake_buffer_2slots_seq_1ins_1outs_ctrl(in0: %handshake_fork10.out2: !esi.channel<i0>, clock: %clock: i1, reset: %reset: i1) -> (out0: !esi.channel<i0>)
  %handshake_buffer70.out0 = hw.instance "handshake_buffer70" @handshake_buffer_2slots_seq_1ins_1outs_ctrl(in0: %handshake_fork10.out1: !esi.channel<i0>, clock: %clock: i1, reset: %reset: i1) -> (out0: !esi.channel<i0>)
  %handshake_buffer71.out0 = hw.instance "handshake_buffer71" @handshake_buffer_2slots_seq_1ins_1outs_ctrl(in0: %handshake_fork10.out0: !esi.channel<i0>, clock: %clock: i1, reset: %reset: i1) -> (out0: !esi.channel<i0>)
  %handshake_join3.out0 = hw.instance "handshake_join3" @handshake_join_5ins_1outs_ctrl(in0: %handshake_buffer65.out0: !esi.channel<i0>, in1: %handshake_buffer20.out0: !esi.channel<i0>, in2: %handshake_buffer16.out0: !esi.channel<i0>, in3: %handshake_buffer10.out0: !esi.channel<i0>, in4: %handshake_buffer6.out0: !esi.channel<i0>) -> (out0: !esi.channel<i0>)
  %handshake_buffer72.out0 = hw.instance "handshake_buffer72" @handshake_buffer_2slots_seq_1ins_1outs_ctrl(in0: %handshake_join3.out0: !esi.channel<i0>, clock: %clock: i1, reset: %reset: i1) -> (out0: !esi.channel<i0>)
  %handshake_load0.dataOut, %handshake_load0.addrOut0 = hw.instance "handshake_load0" @handshake_load_in_ui64_ui32_out_ui32_ui64(addrIn0: %handshake_buffer60.out0: !esi.channel<i64>, dataFromMem: %handshake_buffer19.out0: !esi.channel<i32>, ctrl: %handshake_buffer68.out0: !esi.channel<i0>) -> (dataOut: !esi.channel<i32>, addrOut0: !esi.channel<i64>)
  %handshake_buffer73.out0 = hw.instance "handshake_buffer73" @handshake_buffer_in_ui64_out_ui64_2slots_seq(in0: %handshake_load0.addrOut0: !esi.channel<i64>, clock: %clock: i1, reset: %reset: i1) -> (out0: !esi.channel<i64>)
  %handshake_buffer74.out0 = hw.instance "handshake_buffer74" @handshake_buffer_in_ui32_out_ui32_2slots_seq(in0: %handshake_load0.dataOut: !esi.channel<i32>, clock: %clock: i1, reset: %reset: i1) -> (out0: !esi.channel<i32>)
  %handshake_load1.dataOut, %handshake_load1.addrOut0 = hw.instance "handshake_load1" @handshake_load_in_ui64_ui32_out_ui32_ui64(addrIn0: %handshake_buffer61.out0: !esi.channel<i64>, dataFromMem: %handshake_buffer15.out0: !esi.channel<i32>, ctrl: %handshake_buffer69.out0: !esi.channel<i0>) -> (dataOut: !esi.channel<i32>, addrOut0: !esi.channel<i64>)
  %handshake_buffer75.out0 = hw.instance "handshake_buffer75" @handshake_buffer_in_ui64_out_ui64_2slots_seq(in0: %handshake_load1.addrOut0: !esi.channel<i64>, clock: %clock: i1, reset: %reset: i1) -> (out0: !esi.channel<i64>)
  %handshake_buffer76.out0 = hw.instance "handshake_buffer76" @handshake_buffer_in_ui32_out_ui32_2slots_seq(in0: %handshake_load1.dataOut: !esi.channel<i32>, clock: %clock: i1, reset: %reset: i1) -> (out0: !esi.channel<i32>)
  %handshake_constant5.out0 = hw.instance "handshake_constant5" @handshake_constant_c0_out_ui64(ctrl: %handshake_buffer66.out0: !esi.channel<i0>) -> (out0: !esi.channel<i64>)
  %handshake_buffer77.out0 = hw.instance "handshake_buffer77" @handshake_buffer_in_ui64_out_ui64_2slots_seq(in0: %handshake_constant5.out0: !esi.channel<i64>, clock: %clock: i1, reset: %reset: i1) -> (out0: !esi.channel<i64>)
  %handshake_load2.dataOut, %handshake_load2.addrOut0 = hw.instance "handshake_load2" @handshake_load_in_ui64_ui32_out_ui32_ui64(addrIn0: %handshake_buffer77.out0: !esi.channel<i64>, dataFromMem: %handshake_buffer9.out0: !esi.channel<i32>, ctrl: %handshake_buffer70.out0: !esi.channel<i0>) -> (dataOut: !esi.channel<i32>, addrOut0: !esi.channel<i64>)
  %handshake_buffer78.out0 = hw.instance "handshake_buffer78" @handshake_buffer_in_ui64_out_ui64_2slots_seq(in0: %handshake_load2.addrOut0: !esi.channel<i64>, clock: %clock: i1, reset: %reset: i1) -> (out0: !esi.channel<i64>)
  %handshake_buffer79.out0 = hw.instance "handshake_buffer79" @handshake_buffer_in_ui32_out_ui32_2slots_seq(in0: %handshake_load2.dataOut: !esi.channel<i32>, clock: %clock: i1, reset: %reset: i1) -> (out0: !esi.channel<i32>)
  %arith_muli0.out0 = hw.instance "arith_muli0" @arith_muli_in_ui32_ui32_out_ui32(in0: %handshake_buffer74.out0: !esi.channel<i32>, in1: %handshake_buffer76.out0: !esi.channel<i32>) -> (out0: !esi.channel<i32>)
  %handshake_buffer80.out0 = hw.instance "handshake_buffer80" @handshake_buffer_in_ui32_out_ui32_2slots_seq(in0: %arith_muli0.out0: !esi.channel<i32>, clock: %clock: i1, reset: %reset: i1) -> (out0: !esi.channel<i32>)
  %arith_addi0.out0 = hw.instance "arith_addi0" @arith_addi_in_ui32_ui32_out_ui32(in0: %handshake_buffer79.out0: !esi.channel<i32>, in1: %handshake_buffer80.out0: !esi.channel<i32>) -> (out0: !esi.channel<i32>)
  %handshake_buffer81.out0 = hw.instance "handshake_buffer81" @handshake_buffer_in_ui32_out_ui32_2slots_seq(in0: %arith_addi0.out0: !esi.channel<i32>, clock: %clock: i1, reset: %reset: i1) -> (out0: !esi.channel<i32>)
  %handshake_constant6.out0 = hw.instance "handshake_constant6" @handshake_constant_c0_out_ui64(ctrl: %handshake_buffer67.out0: !esi.channel<i0>) -> (out0: !esi.channel<i64>)
  %handshake_buffer82.out0 = hw.instance "handshake_buffer82" @handshake_buffer_in_ui64_out_ui64_2slots_seq(in0: %handshake_constant6.out0: !esi.channel<i64>, clock: %clock: i1, reset: %reset: i1) -> (out0: !esi.channel<i64>)
  %handshake_join4.out0 = hw.instance "handshake_join4" @handshake_join_2ins_1outs_ctrl(in0: %handshake_buffer71.out0: !esi.channel<i0>, in1: %handshake_buffer11.out0: !esi.channel<i0>) -> (out0: !esi.channel<i0>)
  %handshake_buffer83.out0 = hw.instance "handshake_buffer83" @handshake_buffer_2slots_seq_1ins_1outs_ctrl(in0: %handshake_join4.out0: !esi.channel<i0>, clock: %clock: i1, reset: %reset: i1) -> (out0: !esi.channel<i0>)
  %handshake_store1.dataToMem, %handshake_store1.addrOut0 = hw.instance "handshake_store1" @handshake_store_in_ui64_ui32_out_ui32_ui64(addrIn0: %handshake_buffer82.out0: !esi.channel<i64>, dataIn: %handshake_buffer81.out0: !esi.channel<i32>, ctrl: %handshake_buffer83.out0: !esi.channel<i0>) -> (dataToMem: !esi.channel<i32>, addrOut0: !esi.channel<i64>)
  %handshake_buffer84.out0 = hw.instance "handshake_buffer84" @handshake_buffer_in_ui64_out_ui64_2slots_seq(in0: %handshake_store1.addrOut0: !esi.channel<i64>, clock: %clock: i1, reset: %reset: i1) -> (out0: !esi.channel<i64>)
  %handshake_buffer85.out0 = hw.instance "handshake_buffer85" @handshake_buffer_in_ui32_out_ui32_2slots_seq(in0: %handshake_store1.dataToMem: !esi.channel<i32>, clock: %clock: i1, reset: %reset: i1) -> (out0: !esi.channel<i32>)
  %arith_addi1.out0 = hw.instance "arith_addi1" @arith_addi_in_ui64_ui64_out_ui64(in0: %handshake_buffer62.out0: !esi.channel<i64>, in1: %handshake_buffer63.out0: !esi.channel<i64>) -> (out0: !esi.channel<i64>)
  %handshake_buffer86.out0 = hw.instance "handshake_buffer86" @handshake_buffer_in_ui64_out_ui64_2slots_seq(in0: %arith_addi1.out0: !esi.channel<i64>, clock: %clock: i1, reset: %reset: i1) -> (out0: !esi.channel<i64>)
  %handshake_fork11.out0, %handshake_fork11.out1, %handshake_fork11.out2, %handshake_fork11.out3 = hw.instance "handshake_fork11" @handshake_fork_1ins_4outs_ctrl(in0: %handshake_buffer57.out0: !esi.channel<i0>, clock: %clock: i1, reset: %reset: i1) -> (out0: !esi.channel<i0>, out1: !esi.channel<i0>, out2: !esi.channel<i0>, out3: !esi.channel<i0>)
  %handshake_buffer87.out0 = hw.instance "handshake_buffer87" @handshake_buffer_2slots_seq_1ins_1outs_ctrl(in0: %handshake_fork11.out3: !esi.channel<i0>, clock: %clock: i1, reset: %reset: i1) -> (out0: !esi.channel<i0>)
  %handshake_buffer88.out0 = hw.instance "handshake_buffer88" @handshake_buffer_2slots_seq_1ins_1outs_ctrl(in0: %handshake_fork11.out2: !esi.channel<i0>, clock: %clock: i1, reset: %reset: i1) -> (out0: !esi.channel<i0>)
  %handshake_buffer89.out0 = hw.instance "handshake_buffer89" @handshake_buffer_2slots_seq_1ins_1outs_ctrl(in0: %handshake_fork11.out1: !esi.channel<i0>, clock: %clock: i1, reset: %reset: i1) -> (out0: !esi.channel<i0>)
  %handshake_buffer90.out0 = hw.instance "handshake_buffer90" @handshake_buffer_2slots_seq_1ins_1outs_ctrl(in0: %handshake_fork11.out0: !esi.channel<i0>, clock: %clock: i1, reset: %reset: i1) -> (out0: !esi.channel<i0>)
  %handshake_join5.out0 = hw.instance "handshake_join5" @handshake_join_3ins_1outs_ctrl(in0: %handshake_buffer87.out0: !esi.channel<i0>, in1: %handshake_buffer1.out0: !esi.channel<i0>, in2: %handshake_buffer4.out0: !esi.channel<i0>) -> (out0: !esi.channel<i0>)
  %handshake_buffer91.out0 = hw.instance "handshake_buffer91" @handshake_buffer_2slots_seq_1ins_1outs_ctrl(in0: %handshake_join5.out0: !esi.channel<i0>, clock: %clock: i1, reset: %reset: i1) -> (out0: !esi.channel<i0>)
  %handshake_constant7.out0 = hw.instance "handshake_constant7" @handshake_constant_c0_out_ui64(ctrl: %handshake_buffer88.out0: !esi.channel<i0>) -> (out0: !esi.channel<i64>)
  %handshake_buffer92.out0 = hw.instance "handshake_buffer92" @handshake_buffer_in_ui64_out_ui64_2slots_seq(in0: %handshake_constant7.out0: !esi.channel<i64>, clock: %clock: i1, reset: %reset: i1) -> (out0: !esi.channel<i64>)
  %handshake_fork12.out0, %handshake_fork12.out1 = hw.instance "handshake_fork12" @handshake_fork_in_ui64_out_ui64_ui64(in0: %handshake_buffer92.out0: !esi.channel<i64>, clock: %clock: i1, reset: %reset: i1) -> (out0: !esi.channel<i64>, out1: !esi.channel<i64>)
  %handshake_buffer93.out0 = hw.instance "handshake_buffer93" @handshake_buffer_in_ui64_out_ui64_2slots_seq(in0: %handshake_fork12.out1: !esi.channel<i64>, clock: %clock: i1, reset: %reset: i1) -> (out0: !esi.channel<i64>)
  %handshake_buffer94.out0 = hw.instance "handshake_buffer94" @handshake_buffer_in_ui64_out_ui64_2slots_seq(in0: %handshake_fork12.out0: !esi.channel<i64>, clock: %clock: i1, reset: %reset: i1) -> (out0: !esi.channel<i64>)
  %handshake_load3.dataOut, %handshake_load3.addrOut0 = hw.instance "handshake_load3" @handshake_load_in_ui64_ui32_out_ui32_ui64(addrIn0: %handshake_buffer94.out0: !esi.channel<i64>, dataFromMem: %handshake_buffer8.out0: !esi.channel<i32>, ctrl: %handshake_buffer90.out0: !esi.channel<i0>) -> (dataOut: !esi.channel<i32>, addrOut0: !esi.channel<i64>)
  %handshake_buffer95.out0 = hw.instance "handshake_buffer95" @handshake_buffer_in_ui64_out_ui64_2slots_seq(in0: %handshake_load3.addrOut0: !esi.channel<i64>, clock: %clock: i1, reset: %reset: i1) -> (out0: !esi.channel<i64>)
  %handshake_buffer96.out0 = hw.instance "handshake_buffer96" @handshake_buffer_in_ui32_out_ui32_2slots_seq(in0: %handshake_load3.dataOut: !esi.channel<i32>, clock: %clock: i1, reset: %reset: i1) -> (out0: !esi.channel<i32>)
  %handshake_store2.dataToMem, %handshake_store2.addrOut0 = hw.instance "handshake_store2" @handshake_store_in_ui64_ui32_out_ui32_ui64(addrIn0: %handshake_buffer93.out0: !esi.channel<i64>, dataIn: %handshake_buffer96.out0: !esi.channel<i32>, ctrl: %handshake_buffer89.out0: !esi.channel<i0>) -> (dataToMem: !esi.channel<i32>, addrOut0: !esi.channel<i64>)
  %handshake_buffer97.out0 = hw.instance "handshake_buffer97" @handshake_buffer_in_ui64_out_ui64_2slots_seq(in0: %handshake_store2.addrOut0: !esi.channel<i64>, clock: %clock: i1, reset: %reset: i1) -> (out0: !esi.channel<i64>)
  %handshake_buffer98.out0 = hw.instance "handshake_buffer98" @handshake_buffer_in_ui32_out_ui32_2slots_seq(in0: %handshake_store2.dataToMem: !esi.channel<i32>, clock: %clock: i1, reset: %reset: i1) -> (out0: !esi.channel<i32>)
  hw.output %handshake_buffer91.out0, %handshake_buffer21.out0, %handshake_buffer17.out0, %handshake_buffer13.out0 : !esi.channel<i0>, !esi.channel<i3>, !esi.channel<i3>, !esi.channel<!hw.struct<address: i0, data: i32>>
}
