module attributes {torch.debug_module_name = "DotModule"} {
  sv.verbatim "// Standard header to adapt well known macros to our needs." {symbols = []}
  sv.ifdef  "RANDOMIZE_REG_INIT" {
    sv.verbatim "`define RANDOMIZE" {symbols = []}
  }
  sv.verbatim "\0A// RANDOM may be set to an expression that produces a 32-bit random unsigned value." {symbols = []}
  sv.ifdef  "RANDOM" {
  } else {
    sv.verbatim "`define RANDOM $random" {symbols = []}
  }
  sv.verbatim "\0A// Users can define INIT_RANDOM as general code that gets injected into the\0A// initializer block for modules with registers." {symbols = []}
  sv.ifdef  "INIT_RANDOM" {
  } else {
    sv.verbatim "`define INIT_RANDOM" {symbols = []}
  }
  sv.verbatim "\0A// If using random initialization, you can also define RANDOMIZE_DELAY to\0A// customize the delay used, otherwise 0.002 is used." {symbols = []}
  sv.ifdef  "RANDOMIZE_DELAY" {
  } else {
    sv.verbatim "`define RANDOMIZE_DELAY 0.002" {symbols = []}
  }
  sv.verbatim "\0A// Define INIT_RANDOM_PROLOG_ for use in our modules below." {symbols = []}
  sv.ifdef  "RANDOMIZE" {
    sv.ifdef  "VERILATOR" {
      sv.verbatim "`define INIT_RANDOM_PROLOG_ `INIT_RANDOM" {symbols = []}
    } else {
      sv.verbatim "`define INIT_RANDOM_PROLOG_ `INIT_RANDOM #`RANDOMIZE_DELAY begin end" {symbols = []}
    }
  } else {
    sv.verbatim "`define INIT_RANDOM_PROLOG_" {symbols = []}
  }
  sv.verbatim "" {symbols = []}
  hw.module @handshake_buffer_2slots_seq_1ins_1outs_ctrl(%in0_valid: i1, %out0_ready: i1, %clock: i1, %reset: i1) -> (in0_ready: i1, out0_valid: i1) {
    %true = hw.constant true
    %false = hw.constant false
    %validReg0 = seq.firreg %2 clock %clock reset sync %reset, %false : i1
    %0 = comb.xor %validReg0, %true : i1
    %1 = comb.or %0, %4 : i1
    %2 = comb.mux %1, %in0_valid, %validReg0 : i1
    %readyReg = seq.firreg %9 clock %clock reset sync %reset, %false : i1
    %3 = comb.mux %readyReg, %readyReg, %validReg0 : i1
    %4 = comb.xor %readyReg, %true : i1
    %5 = comb.or %11, %readyReg : i1
    %6 = comb.mux %5, %readyReg, %validReg0 : i1
    %7 = comb.and %11, %readyReg : i1
    %8 = comb.xor %7, %true : i1
    %9 = comb.and %8, %6 : i1
    %validReg1 = seq.firreg %12 clock %clock reset sync %reset, %false : i1
    %10 = comb.xor %validReg1, %true : i1
    %11 = comb.or %10, %14 : i1
    %12 = comb.mux %11, %3, %validReg1 : i1
    %readyReg_0 = seq.firreg %19 clock %clock reset sync %reset, %false {name = "readyReg"} : i1
    %13 = comb.mux %readyReg_0, %readyReg_0, %validReg1 : i1
    %14 = comb.xor %readyReg_0, %true : i1
    %15 = comb.or %out0_ready, %readyReg_0 : i1
    %16 = comb.mux %15, %readyReg_0, %validReg1 : i1
    %17 = comb.and %out0_ready, %readyReg_0 : i1
    %18 = comb.xor %17, %true : i1
    %19 = comb.and %18, %16 : i1
    hw.output %1, %13 : i1, i1
  }
  hw.module @handshake_extmemory_in_ui64_out_ui32(%extmem_ldAddr0_ready: i1, %extmem_ldData0_valid: i1, %extmem_ldData0_data: i32, %extmem_ldDone0_valid: i1, %ldAddr0_valid: i1, %ldAddr0_data: i64, %ldData0_ready: i1, %ldDone0_ready: i1) -> (extmem_ldAddr0_valid: i1, extmem_ldAddr0_data: i64, extmem_ldData0_ready: i1, extmem_ldDone0_ready: i1, ldAddr0_ready: i1, ldData0_valid: i1, ldData0_data: i32, ldDone0_valid: i1) {
    hw.output %ldAddr0_valid, %ldAddr0_data, %ldData0_ready, %ldDone0_ready, %extmem_ldAddr0_ready, %extmem_ldData0_valid, %extmem_ldData0_data, %extmem_ldDone0_valid : i1, i64, i1, i1, i1, i1, i32, i1
  }
  hw.module @handshake_buffer_in_ui32_out_ui32_2slots_seq(%in0_valid: i1, %in0_data: i32, %out0_ready: i1, %clock: i1, %reset: i1) -> (in0_ready: i1, out0_valid: i1, out0_data: i32) {
    %true = hw.constant true
    %c0_i32 = hw.constant 0 : i32
    %false = hw.constant false
    %validReg0 = seq.firreg %2 clock %clock reset sync %reset, %false : i1
    %dataReg0 = seq.firreg %3 clock %clock reset sync %reset, %c0_i32 : i32
    %0 = comb.xor %validReg0, %true : i1
    %1 = comb.or %0, %5 : i1
    %2 = comb.mux %1, %in0_valid, %validReg0 : i1
    %3 = comb.mux %1, %in0_data, %dataReg0 : i32
    %readyReg = seq.firreg %11 clock %clock reset sync %reset, %false : i1
    %4 = comb.mux %readyReg, %readyReg, %validReg0 : i1
    %5 = comb.xor %readyReg, %true : i1
    %6 = comb.xor %16, %true : i1
    %7 = comb.and %6, %5 : i1
    %8 = comb.mux %7, %validReg0, %readyReg : i1
    %9 = comb.and %16, %readyReg : i1
    %10 = comb.xor %9, %true : i1
    %11 = comb.and %10, %8 : i1
    %ctrlDataReg = seq.firreg %14 clock %clock reset sync %reset, %c0_i32 : i32
    %12 = comb.mux %readyReg, %ctrlDataReg, %dataReg0 : i32
    %13 = comb.mux %7, %dataReg0, %ctrlDataReg : i32
    %14 = comb.mux %9, %c0_i32, %13 : i32
    %validReg1 = seq.firreg %17 clock %clock reset sync %reset, %false : i1
    %dataReg1 = seq.firreg %18 clock %clock reset sync %reset, %c0_i32 : i32
    %15 = comb.xor %validReg1, %true : i1
    %16 = comb.or %15, %20 : i1
    %17 = comb.mux %16, %4, %validReg1 : i1
    %18 = comb.mux %16, %12, %dataReg1 : i32
    %readyReg_0 = seq.firreg %26 clock %clock reset sync %reset, %false {name = "readyReg"} : i1
    %19 = comb.mux %readyReg_0, %readyReg_0, %validReg1 : i1
    %20 = comb.xor %readyReg_0, %true : i1
    %21 = comb.xor %out0_ready, %true : i1
    %22 = comb.and %21, %20 : i1
    %23 = comb.mux %22, %validReg1, %readyReg_0 : i1
    %24 = comb.and %out0_ready, %readyReg_0 : i1
    %25 = comb.xor %24, %true : i1
    %26 = comb.and %25, %23 : i1
    %ctrlDataReg_1 = seq.firreg %29 clock %clock reset sync %reset, %c0_i32 {name = "ctrlDataReg"} : i32
    %27 = comb.mux %readyReg_0, %ctrlDataReg_1, %dataReg1 : i32
    %28 = comb.mux %22, %dataReg1, %ctrlDataReg_1 : i32
    %29 = comb.mux %24, %c0_i32, %28 : i32
    hw.output %1, %19, %27 : i1, i1, i32
  }
  hw.module @handshake_fork_1ins_5outs_ctrl(%in0_valid: i1, %out0_ready: i1, %out1_ready: i1, %out2_ready: i1, %out3_ready: i1, %out4_ready: i1, %clock: i1, %reset: i1) -> (in0_ready: i1, out0_valid: i1, out1_valid: i1, out2_valid: i1, out3_valid: i1, out4_valid: i1) {
    %true = hw.constant true
    %false = hw.constant false
    %0 = comb.and %26, %21, %16, %11, %6 : i1
    %1 = comb.xor %0, %true : i1
    %emtd0 = seq.firreg %2 clock %clock reset sync %reset, %false : i1
    %2 = comb.and %6, %1 : i1
    %3 = comb.xor %emtd0, %true : i1
    %4 = comb.and %3, %in0_valid : i1
    %5 = comb.and %out0_ready, %4 : i1
    %6 = comb.or %5, %emtd0 : i1
    %emtd1 = seq.firreg %7 clock %clock reset sync %reset, %false : i1
    %7 = comb.and %11, %1 : i1
    %8 = comb.xor %emtd1, %true : i1
    %9 = comb.and %8, %in0_valid : i1
    %10 = comb.and %out1_ready, %9 : i1
    %11 = comb.or %10, %emtd1 : i1
    %emtd2 = seq.firreg %12 clock %clock reset sync %reset, %false : i1
    %12 = comb.and %16, %1 : i1
    %13 = comb.xor %emtd2, %true : i1
    %14 = comb.and %13, %in0_valid : i1
    %15 = comb.and %out2_ready, %14 : i1
    %16 = comb.or %15, %emtd2 : i1
    %emtd3 = seq.firreg %17 clock %clock reset sync %reset, %false : i1
    %17 = comb.and %21, %1 : i1
    %18 = comb.xor %emtd3, %true : i1
    %19 = comb.and %18, %in0_valid : i1
    %20 = comb.and %out3_ready, %19 : i1
    %21 = comb.or %20, %emtd3 : i1
    %emtd4 = seq.firreg %22 clock %clock reset sync %reset, %false : i1
    %22 = comb.and %26, %1 : i1
    %23 = comb.xor %emtd4, %true : i1
    %24 = comb.and %23, %in0_valid : i1
    %25 = comb.and %out4_ready, %24 : i1
    %26 = comb.or %25, %emtd4 : i1
    hw.output %0, %4, %9, %14, %19, %24 : i1, i1, i1, i1, i1, i1
  }
  hw.module @handshake_constant_c0_out_ui32(%ctrl_valid: i1, %out0_ready: i1) -> (ctrl_ready: i1, out0_valid: i1, out0_data: i32) {
    %c0_i32 = hw.constant 0 : i32
    hw.output %out0_ready, %ctrl_valid, %c0_i32 : i1, i1, i32
  }
  hw.module @handshake_constant_c0_out_ui64(%ctrl_valid: i1, %out0_ready: i1) -> (ctrl_ready: i1, out0_valid: i1, out0_data: i64) {
    %c0_i64 = hw.constant 0 : i64
    hw.output %out0_ready, %ctrl_valid, %c0_i64 : i1, i1, i64
  }
  hw.module @handshake_buffer_in_ui64_out_ui64_2slots_seq(%in0_valid: i1, %in0_data: i64, %out0_ready: i1, %clock: i1, %reset: i1) -> (in0_ready: i1, out0_valid: i1, out0_data: i64) {
    %true = hw.constant true
    %c0_i64 = hw.constant 0 : i64
    %false = hw.constant false
    %validReg0 = seq.firreg %2 clock %clock reset sync %reset, %false : i1
    %dataReg0 = seq.firreg %3 clock %clock reset sync %reset, %c0_i64 : i64
    %0 = comb.xor %validReg0, %true : i1
    %1 = comb.or %0, %5 : i1
    %2 = comb.mux %1, %in0_valid, %validReg0 : i1
    %3 = comb.mux %1, %in0_data, %dataReg0 : i64
    %readyReg = seq.firreg %11 clock %clock reset sync %reset, %false : i1
    %4 = comb.mux %readyReg, %readyReg, %validReg0 : i1
    %5 = comb.xor %readyReg, %true : i1
    %6 = comb.xor %16, %true : i1
    %7 = comb.and %6, %5 : i1
    %8 = comb.mux %7, %validReg0, %readyReg : i1
    %9 = comb.and %16, %readyReg : i1
    %10 = comb.xor %9, %true : i1
    %11 = comb.and %10, %8 : i1
    %ctrlDataReg = seq.firreg %14 clock %clock reset sync %reset, %c0_i64 : i64
    %12 = comb.mux %readyReg, %ctrlDataReg, %dataReg0 : i64
    %13 = comb.mux %7, %dataReg0, %ctrlDataReg : i64
    %14 = comb.mux %9, %c0_i64, %13 : i64
    %validReg1 = seq.firreg %17 clock %clock reset sync %reset, %false : i1
    %dataReg1 = seq.firreg %18 clock %clock reset sync %reset, %c0_i64 : i64
    %15 = comb.xor %validReg1, %true : i1
    %16 = comb.or %15, %20 : i1
    %17 = comb.mux %16, %4, %validReg1 : i1
    %18 = comb.mux %16, %12, %dataReg1 : i64
    %readyReg_0 = seq.firreg %26 clock %clock reset sync %reset, %false {name = "readyReg"} : i1
    %19 = comb.mux %readyReg_0, %readyReg_0, %validReg1 : i1
    %20 = comb.xor %readyReg_0, %true : i1
    %21 = comb.xor %out0_ready, %true : i1
    %22 = comb.and %21, %20 : i1
    %23 = comb.mux %22, %validReg1, %readyReg_0 : i1
    %24 = comb.and %out0_ready, %readyReg_0 : i1
    %25 = comb.xor %24, %true : i1
    %26 = comb.and %25, %23 : i1
    %ctrlDataReg_1 = seq.firreg %29 clock %clock reset sync %reset, %c0_i64 {name = "ctrlDataReg"} : i64
    %27 = comb.mux %readyReg_0, %ctrlDataReg_1, %dataReg1 : i64
    %28 = comb.mux %22, %dataReg1, %ctrlDataReg_1 : i64
    %29 = comb.mux %24, %c0_i64, %28 : i64
    hw.output %1, %19, %27 : i1, i1, i64
  }
  hw.module @handshake_constant_c5_out_ui64(%ctrl_valid: i1, %out0_ready: i1) -> (ctrl_ready: i1, out0_valid: i1, out0_data: i64) {
    %c5_i64 = hw.constant 5 : i64
    hw.output %out0_ready, %ctrl_valid, %c5_i64 : i1, i1, i64
  }
  hw.module @handshake_constant_c1_out_ui64(%ctrl_valid: i1, %out0_ready: i1) -> (ctrl_ready: i1, out0_valid: i1, out0_data: i64) {
    %c1_i64 = hw.constant 1 : i64
    hw.output %out0_ready, %ctrl_valid, %c1_i64 : i1, i1, i64
  }
  hw.module @handshake_buffer_in_ui1_out_ui1_1slots_seq_init_0(%in0_valid: i1, %in0_data: i1, %out0_ready: i1, %clock: i1, %reset: i1) -> (in0_ready: i1, out0_valid: i1, out0_data: i1) {
    %true = hw.constant true
    %false = hw.constant false
    %validReg0 = seq.firreg %2 clock %clock reset sync %reset, %true : i1
    %dataReg0 = seq.firreg %3 clock %clock reset sync %reset, %false : i1
    %0 = comb.xor %validReg0, %true : i1
    %1 = comb.or %0, %5 : i1
    %2 = comb.mux %1, %in0_valid, %validReg0 : i1
    %3 = comb.mux %1, %in0_data, %dataReg0 : i1
    %readyReg = seq.firreg %11 clock %clock reset sync %reset, %false : i1
    %4 = comb.mux %readyReg, %readyReg, %validReg0 : i1
    %5 = comb.xor %readyReg, %true : i1
    %6 = comb.xor %out0_ready, %true : i1
    %7 = comb.and %6, %5 : i1
    %8 = comb.mux %7, %validReg0, %readyReg : i1
    %9 = comb.and %out0_ready, %readyReg : i1
    %10 = comb.xor %9, %true : i1
    %11 = comb.and %10, %8 : i1
    %ctrlDataReg = seq.firreg %15 clock %clock reset sync %reset, %false : i1
    %12 = comb.mux %readyReg, %ctrlDataReg, %dataReg0 : i1
    %13 = comb.mux %7, %dataReg0, %ctrlDataReg : i1
    %14 = comb.xor %9, %true : i1
    %15 = comb.and %14, %13 : i1
    hw.output %1, %4, %12 : i1, i1, i1
  }
  hw.module @handshake_fork_in_ui1_out_ui1_ui1_ui1_ui1_ui1(%in0_valid: i1, %in0_data: i1, %out0_ready: i1, %out1_ready: i1, %out2_ready: i1, %out3_ready: i1, %out4_ready: i1, %clock: i1, %reset: i1) -> (in0_ready: i1, out0_valid: i1, out0_data: i1, out1_valid: i1, out1_data: i1, out2_valid: i1, out2_data: i1, out3_valid: i1, out3_data: i1, out4_valid: i1, out4_data: i1) {
    %true = hw.constant true
    %false = hw.constant false
    %0 = comb.and %26, %21, %16, %11, %6 : i1
    %1 = comb.xor %0, %true : i1
    %emtd0 = seq.firreg %2 clock %clock reset sync %reset, %false : i1
    %2 = comb.and %6, %1 : i1
    %3 = comb.xor %emtd0, %true : i1
    %4 = comb.and %3, %in0_valid : i1
    %5 = comb.and %out0_ready, %4 : i1
    %6 = comb.or %5, %emtd0 : i1
    %emtd1 = seq.firreg %7 clock %clock reset sync %reset, %false : i1
    %7 = comb.and %11, %1 : i1
    %8 = comb.xor %emtd1, %true : i1
    %9 = comb.and %8, %in0_valid : i1
    %10 = comb.and %out1_ready, %9 : i1
    %11 = comb.or %10, %emtd1 : i1
    %emtd2 = seq.firreg %12 clock %clock reset sync %reset, %false : i1
    %12 = comb.and %16, %1 : i1
    %13 = comb.xor %emtd2, %true : i1
    %14 = comb.and %13, %in0_valid : i1
    %15 = comb.and %out2_ready, %14 : i1
    %16 = comb.or %15, %emtd2 : i1
    %emtd3 = seq.firreg %17 clock %clock reset sync %reset, %false : i1
    %17 = comb.and %21, %1 : i1
    %18 = comb.xor %emtd3, %true : i1
    %19 = comb.and %18, %in0_valid : i1
    %20 = comb.and %out3_ready, %19 : i1
    %21 = comb.or %20, %emtd3 : i1
    %emtd4 = seq.firreg %22 clock %clock reset sync %reset, %false : i1
    %22 = comb.and %26, %1 : i1
    %23 = comb.xor %emtd4, %true : i1
    %24 = comb.and %23, %in0_valid : i1
    %25 = comb.and %out4_ready, %24 : i1
    %26 = comb.or %25, %emtd4 : i1
    hw.output %0, %4, %in0_data, %9, %in0_data, %14, %in0_data, %19, %in0_data, %24, %in0_data : i1, i1, i1, i1, i1, i1, i1, i1, i1, i1, i1
  }
  hw.module @handshake_buffer_in_ui1_out_ui1_2slots_seq(%in0_valid: i1, %in0_data: i1, %out0_ready: i1, %clock: i1, %reset: i1) -> (in0_ready: i1, out0_valid: i1, out0_data: i1) {
    %true = hw.constant true
    %false = hw.constant false
    %validReg0 = seq.firreg %2 clock %clock reset sync %reset, %false : i1
    %dataReg0 = seq.firreg %3 clock %clock reset sync %reset, %false : i1
    %0 = comb.xor %validReg0, %true : i1
    %1 = comb.or %0, %5 : i1
    %2 = comb.mux %1, %in0_valid, %validReg0 : i1
    %3 = comb.mux %1, %in0_data, %dataReg0 : i1
    %readyReg = seq.firreg %11 clock %clock reset sync %reset, %false : i1
    %4 = comb.mux %readyReg, %readyReg, %validReg0 : i1
    %5 = comb.xor %readyReg, %true : i1
    %6 = comb.xor %17, %true : i1
    %7 = comb.and %6, %5 : i1
    %8 = comb.mux %7, %validReg0, %readyReg : i1
    %9 = comb.and %17, %readyReg : i1
    %10 = comb.xor %9, %true : i1
    %11 = comb.and %10, %8 : i1
    %ctrlDataReg = seq.firreg %15 clock %clock reset sync %reset, %false : i1
    %12 = comb.mux %readyReg, %ctrlDataReg, %dataReg0 : i1
    %13 = comb.mux %7, %dataReg0, %ctrlDataReg : i1
    %14 = comb.xor %9, %true : i1
    %15 = comb.and %14, %13 : i1
    %validReg1 = seq.firreg %18 clock %clock reset sync %reset, %false : i1
    %dataReg1 = seq.firreg %19 clock %clock reset sync %reset, %false : i1
    %16 = comb.xor %validReg1, %true : i1
    %17 = comb.or %16, %21 : i1
    %18 = comb.mux %17, %4, %validReg1 : i1
    %19 = comb.mux %17, %12, %dataReg1 : i1
    %readyReg_0 = seq.firreg %27 clock %clock reset sync %reset, %false {name = "readyReg"} : i1
    %20 = comb.mux %readyReg_0, %readyReg_0, %validReg1 : i1
    %21 = comb.xor %readyReg_0, %true : i1
    %22 = comb.xor %out0_ready, %true : i1
    %23 = comb.and %22, %21 : i1
    %24 = comb.mux %23, %validReg1, %readyReg_0 : i1
    %25 = comb.and %out0_ready, %readyReg_0 : i1
    %26 = comb.xor %25, %true : i1
    %27 = comb.and %26, %24 : i1
    %ctrlDataReg_1 = seq.firreg %31 clock %clock reset sync %reset, %false {name = "ctrlDataReg"} : i1
    %28 = comb.mux %readyReg_0, %ctrlDataReg_1, %dataReg1 : i1
    %29 = comb.mux %23, %dataReg1, %ctrlDataReg_1 : i1
    %30 = comb.xor %25, %true : i1
    %31 = comb.and %30, %29 : i1
    hw.output %1, %20, %28 : i1, i1, i1
  }
  hw.module @handshake_mux_in_ui1_3ins_1outs_ctrl(%select_valid: i1, %select_data: i1, %in0_valid: i1, %in1_valid: i1, %out0_ready: i1) -> (select_ready: i1, in0_ready: i1, in1_ready: i1, out0_valid: i1) {
    %true = hw.constant true
    %0 = comb.mux %select_data, %in1_valid, %in0_valid : i1
    %1 = comb.and %0, %select_valid : i1
    %2 = comb.and %1, %out0_ready : i1
    %3 = comb.xor %select_data, %true : i1
    %4 = comb.and %3, %2 : i1
    %5 = comb.and %select_data, %2 : i1
    hw.output %2, %4, %5, %1 : i1, i1, i1, i1
  }
  hw.module @handshake_mux_in_ui1_ui64_ui64_out_ui64(%select_valid: i1, %select_data: i1, %in0_valid: i1, %in0_data: i64, %in1_valid: i1, %in1_data: i64, %out0_ready: i1) -> (select_ready: i1, in0_ready: i1, in1_ready: i1, out0_valid: i1, out0_data: i64) {
    %true = hw.constant true
    %0 = comb.mux %select_data, %in1_data, %in0_data : i64
    %1 = comb.mux %select_data, %in1_valid, %in0_valid : i1
    %2 = comb.and %1, %select_valid : i1
    %3 = comb.and %2, %out0_ready : i1
    %4 = comb.xor %select_data, %true : i1
    %5 = comb.and %4, %3 : i1
    %6 = comb.and %select_data, %3 : i1
    hw.output %3, %5, %6, %2, %0 : i1, i1, i1, i1, i64
  }
  hw.module @handshake_fork_in_ui64_out_ui64_ui64(%in0_valid: i1, %in0_data: i64, %out0_ready: i1, %out1_ready: i1, %clock: i1, %reset: i1) -> (in0_ready: i1, out0_valid: i1, out0_data: i64, out1_valid: i1, out1_data: i64) {
    %true = hw.constant true
    %false = hw.constant false
    %0 = comb.and %11, %6 : i1
    %1 = comb.xor %0, %true : i1
    %emtd0 = seq.firreg %2 clock %clock reset sync %reset, %false : i1
    %2 = comb.and %6, %1 : i1
    %3 = comb.xor %emtd0, %true : i1
    %4 = comb.and %3, %in0_valid : i1
    %5 = comb.and %out0_ready, %4 : i1
    %6 = comb.or %5, %emtd0 : i1
    %emtd1 = seq.firreg %7 clock %clock reset sync %reset, %false : i1
    %7 = comb.and %11, %1 : i1
    %8 = comb.xor %emtd1, %true : i1
    %9 = comb.and %8, %in0_valid : i1
    %10 = comb.and %out1_ready, %9 : i1
    %11 = comb.or %10, %emtd1 : i1
    hw.output %0, %4, %in0_data, %9, %in0_data : i1, i1, i64, i1, i64
  }
  hw.module @handshake_mux_in_ui1_ui32_ui32_out_ui32(%select_valid: i1, %select_data: i1, %in0_valid: i1, %in0_data: i32, %in1_valid: i1, %in1_data: i32, %out0_ready: i1) -> (select_ready: i1, in0_ready: i1, in1_ready: i1, out0_valid: i1, out0_data: i32) {
    %true = hw.constant true
    %0 = comb.mux %select_data, %in1_data, %in0_data : i32
    %1 = comb.mux %select_data, %in1_valid, %in0_valid : i1
    %2 = comb.and %1, %select_valid : i1
    %3 = comb.and %2, %out0_ready : i1
    %4 = comb.xor %select_data, %true : i1
    %5 = comb.and %4, %3 : i1
    %6 = comb.and %select_data, %3 : i1
    hw.output %3, %5, %6, %2, %0 : i1, i1, i1, i1, i32
  }
  hw.module @arith_cmpi_in_ui64_ui64_out_ui1_slt(%in0_valid: i1, %in0_data: i64, %in1_valid: i1, %in1_data: i64, %out0_ready: i1) -> (in0_ready: i1, in1_ready: i1, out0_valid: i1, out0_data: i1) {
    %0 = comb.icmp slt %in0_data, %in1_data : i64
    %1 = comb.and %in0_valid, %in1_valid : i1
    %2 = comb.and %out0_ready, %1 : i1
    hw.output %2, %2, %1, %0 : i1, i1, i1, i1
  }
  hw.module @handshake_fork_in_ui1_out_ui1_ui1_ui1_ui1_ui1_ui1(%in0_valid: i1, %in0_data: i1, %out0_ready: i1, %out1_ready: i1, %out2_ready: i1, %out3_ready: i1, %out4_ready: i1, %out5_ready: i1, %clock: i1, %reset: i1) -> (in0_ready: i1, out0_valid: i1, out0_data: i1, out1_valid: i1, out1_data: i1, out2_valid: i1, out2_data: i1, out3_valid: i1, out3_data: i1, out4_valid: i1, out4_data: i1, out5_valid: i1, out5_data: i1) {
    %true = hw.constant true
    %false = hw.constant false
    %0 = comb.and %31, %26, %21, %16, %11, %6 : i1
    %1 = comb.xor %0, %true : i1
    %emtd0 = seq.firreg %2 clock %clock reset sync %reset, %false : i1
    %2 = comb.and %6, %1 : i1
    %3 = comb.xor %emtd0, %true : i1
    %4 = comb.and %3, %in0_valid : i1
    %5 = comb.and %out0_ready, %4 : i1
    %6 = comb.or %5, %emtd0 : i1
    %emtd1 = seq.firreg %7 clock %clock reset sync %reset, %false : i1
    %7 = comb.and %11, %1 : i1
    %8 = comb.xor %emtd1, %true : i1
    %9 = comb.and %8, %in0_valid : i1
    %10 = comb.and %out1_ready, %9 : i1
    %11 = comb.or %10, %emtd1 : i1
    %emtd2 = seq.firreg %12 clock %clock reset sync %reset, %false : i1
    %12 = comb.and %16, %1 : i1
    %13 = comb.xor %emtd2, %true : i1
    %14 = comb.and %13, %in0_valid : i1
    %15 = comb.and %out2_ready, %14 : i1
    %16 = comb.or %15, %emtd2 : i1
    %emtd3 = seq.firreg %17 clock %clock reset sync %reset, %false : i1
    %17 = comb.and %21, %1 : i1
    %18 = comb.xor %emtd3, %true : i1
    %19 = comb.and %18, %in0_valid : i1
    %20 = comb.and %out3_ready, %19 : i1
    %21 = comb.or %20, %emtd3 : i1
    %emtd4 = seq.firreg %22 clock %clock reset sync %reset, %false : i1
    %22 = comb.and %26, %1 : i1
    %23 = comb.xor %emtd4, %true : i1
    %24 = comb.and %23, %in0_valid : i1
    %25 = comb.and %out4_ready, %24 : i1
    %26 = comb.or %25, %emtd4 : i1
    %emtd5 = seq.firreg %27 clock %clock reset sync %reset, %false : i1
    %27 = comb.and %31, %1 : i1
    %28 = comb.xor %emtd5, %true : i1
    %29 = comb.and %28, %in0_valid : i1
    %30 = comb.and %out5_ready, %29 : i1
    %31 = comb.or %30, %emtd5 : i1
    hw.output %0, %4, %in0_data, %9, %in0_data, %14, %in0_data, %19, %in0_data, %24, %in0_data, %29, %in0_data : i1, i1, i1, i1, i1, i1, i1, i1, i1, i1, i1, i1, i1
  }
  hw.module @handshake_cond_br_in_ui1_ui64_out_ui64_ui64(%cond_valid: i1, %cond_data: i1, %data_valid: i1, %data_data: i64, %outTrue_ready: i1, %outFalse_ready: i1) -> (cond_ready: i1, data_ready: i1, outTrue_valid: i1, outTrue_data: i64, outFalse_valid: i1, outFalse_data: i64) {
    %true = hw.constant true
    %0 = comb.and %cond_valid, %data_valid : i1
    %1 = comb.xor %cond_data, %true : i1
    %2 = comb.and %cond_data, %0 : i1
    %3 = comb.and %1, %0 : i1
    %4 = comb.mux %cond_data, %outTrue_ready, %outFalse_ready : i1
    %5 = comb.and %4, %0 : i1
    hw.output %5, %5, %2, %data_data, %3, %data_data : i1, i1, i1, i64, i1, i64
  }
  hw.module @handshake_sink_in_ui64(%in0_valid: i1, %in0_data: i64) -> (in0_ready: i1) {
    %true = hw.constant true
    hw.output %true : i1
  }
  hw.module @handshake_cond_br_in_ui1_2ins_2outs_ctrl(%cond_valid: i1, %cond_data: i1, %data_valid: i1, %outTrue_ready: i1, %outFalse_ready: i1) -> (cond_ready: i1, data_ready: i1, outTrue_valid: i1, outFalse_valid: i1) {
    %true = hw.constant true
    %0 = comb.and %cond_valid, %data_valid : i1
    %1 = comb.xor %cond_data, %true : i1
    %2 = comb.and %cond_data, %0 : i1
    %3 = comb.and %1, %0 : i1
    %4 = comb.mux %cond_data, %outTrue_ready, %outFalse_ready : i1
    %5 = comb.and %4, %0 : i1
    hw.output %5, %5, %2, %3 : i1, i1, i1, i1
  }
  hw.module @handshake_cond_br_in_ui1_ui32_out_ui32_ui32(%cond_valid: i1, %cond_data: i1, %data_valid: i1, %data_data: i32, %outTrue_ready: i1, %outFalse_ready: i1) -> (cond_ready: i1, data_ready: i1, outTrue_valid: i1, outTrue_data: i32, outFalse_valid: i1, outFalse_data: i32) {
    %true = hw.constant true
    %0 = comb.and %cond_valid, %data_valid : i1
    %1 = comb.xor %cond_data, %true : i1
    %2 = comb.and %cond_data, %0 : i1
    %3 = comb.and %1, %0 : i1
    %4 = comb.mux %cond_data, %outTrue_ready, %outFalse_ready : i1
    %5 = comb.and %4, %0 : i1
    hw.output %5, %5, %2, %data_data, %3, %data_data : i1, i1, i1, i32, i1, i32
  }
  hw.module @handshake_fork_in_ui64_out_ui64_ui64_ui64(%in0_valid: i1, %in0_data: i64, %out0_ready: i1, %out1_ready: i1, %out2_ready: i1, %clock: i1, %reset: i1) -> (in0_ready: i1, out0_valid: i1, out0_data: i64, out1_valid: i1, out1_data: i64, out2_valid: i1, out2_data: i64) {
    %true = hw.constant true
    %false = hw.constant false
    %0 = comb.and %16, %11, %6 : i1
    %1 = comb.xor %0, %true : i1
    %emtd0 = seq.firreg %2 clock %clock reset sync %reset, %false : i1
    %2 = comb.and %6, %1 : i1
    %3 = comb.xor %emtd0, %true : i1
    %4 = comb.and %3, %in0_valid : i1
    %5 = comb.and %out0_ready, %4 : i1
    %6 = comb.or %5, %emtd0 : i1
    %emtd1 = seq.firreg %7 clock %clock reset sync %reset, %false : i1
    %7 = comb.and %11, %1 : i1
    %8 = comb.xor %emtd1, %true : i1
    %9 = comb.and %8, %in0_valid : i1
    %10 = comb.and %out1_ready, %9 : i1
    %11 = comb.or %10, %emtd1 : i1
    %emtd2 = seq.firreg %12 clock %clock reset sync %reset, %false : i1
    %12 = comb.and %16, %1 : i1
    %13 = comb.xor %emtd2, %true : i1
    %14 = comb.and %13, %in0_valid : i1
    %15 = comb.and %out2_ready, %14 : i1
    %16 = comb.or %15, %emtd2 : i1
    hw.output %0, %4, %in0_data, %9, %in0_data, %14, %in0_data : i1, i1, i64, i1, i64, i1, i64
  }
  hw.module @handshake_fork_1ins_3outs_ctrl(%in0_valid: i1, %out0_ready: i1, %out1_ready: i1, %out2_ready: i1, %clock: i1, %reset: i1) -> (in0_ready: i1, out0_valid: i1, out1_valid: i1, out2_valid: i1) {
    %true = hw.constant true
    %false = hw.constant false
    %0 = comb.and %16, %11, %6 : i1
    %1 = comb.xor %0, %true : i1
    %emtd0 = seq.firreg %2 clock %clock reset sync %reset, %false : i1
    %2 = comb.and %6, %1 : i1
    %3 = comb.xor %emtd0, %true : i1
    %4 = comb.and %3, %in0_valid : i1
    %5 = comb.and %out0_ready, %4 : i1
    %6 = comb.or %5, %emtd0 : i1
    %emtd1 = seq.firreg %7 clock %clock reset sync %reset, %false : i1
    %7 = comb.and %11, %1 : i1
    %8 = comb.xor %emtd1, %true : i1
    %9 = comb.and %8, %in0_valid : i1
    %10 = comb.and %out1_ready, %9 : i1
    %11 = comb.or %10, %emtd1 : i1
    %emtd2 = seq.firreg %12 clock %clock reset sync %reset, %false : i1
    %12 = comb.and %16, %1 : i1
    %13 = comb.xor %emtd2, %true : i1
    %14 = comb.and %13, %in0_valid : i1
    %15 = comb.and %out2_ready, %14 : i1
    %16 = comb.or %15, %emtd2 : i1
    hw.output %0, %4, %9, %14 : i1, i1, i1, i1
  }
  hw.module @handshake_join_3ins_1outs_ctrl(%in0_valid: i1, %in1_valid: i1, %in2_valid: i1, %out0_ready: i1) -> (in0_ready: i1, in1_ready: i1, in2_ready: i1, out0_valid: i1) {
    %0 = comb.and %in2_valid, %in1_valid, %in0_valid : i1
    %1 = comb.and %out0_ready, %0 : i1
    hw.output %1, %1, %1, %0 : i1, i1, i1, i1
  }
  hw.module @handshake_load_in_ui64_ui32_out_ui32_ui64(%addrIn0_valid: i1, %addrIn0_data: i64, %dataFromMem_valid: i1, %dataFromMem_data: i32, %ctrl_valid: i1, %dataOut_ready: i1, %addrOut0_ready: i1) -> (addrIn0_ready: i1, dataFromMem_ready: i1, ctrl_ready: i1, dataOut_valid: i1, dataOut_data: i32, addrOut0_valid: i1, addrOut0_data: i64) {
    %0 = comb.and %addrIn0_valid, %ctrl_valid : i1
    %1 = comb.and %0, %addrOut0_ready : i1
    hw.output %1, %dataOut_ready, %1, %dataFromMem_valid, %dataFromMem_data, %0, %addrIn0_data : i1, i1, i1, i1, i32, i1, i64
  }
  hw.module @arith_muli_in_ui32_ui32_out_ui32(%in0_valid: i1, %in0_data: i32, %in1_valid: i1, %in1_data: i32, %out0_ready: i1) -> (in0_ready: i1, in1_ready: i1, out0_valid: i1, out0_data: i32) {
    %0 = comb.mul %in0_data, %in1_data : i32
    %1 = comb.and %in0_valid, %in1_valid : i1
    %2 = comb.and %out0_ready, %1 : i1
    hw.output %2, %2, %1, %0 : i1, i1, i1, i32
  }
  hw.module @arith_addi_in_ui32_ui32_out_ui32(%in0_valid: i1, %in0_data: i32, %in1_valid: i1, %in1_data: i32, %out0_ready: i1) -> (in0_ready: i1, in1_ready: i1, out0_valid: i1, out0_data: i32) {
    %0 = comb.add %in0_data, %in1_data : i32
    %1 = comb.and %in0_valid, %in1_valid : i1
    %2 = comb.and %out0_ready, %1 : i1
    hw.output %2, %2, %1, %0 : i1, i1, i1, i32
  }
  hw.module @arith_addi_in_ui64_ui64_out_ui64(%in0_valid: i1, %in0_data: i64, %in1_valid: i1, %in1_data: i64, %out0_ready: i1) -> (in0_ready: i1, in1_ready: i1, out0_valid: i1, out0_data: i64) {
    %0 = comb.add %in0_data, %in1_data : i64
    %1 = comb.and %in0_valid, %in1_valid : i1
    %2 = comb.and %out0_ready, %1 : i1
    hw.output %2, %2, %1, %0 : i1, i1, i1, i64
  }
  hw.module @forward(%in0_ldAddr0_ready: i1, %in0_ldData0_valid: i1, %in0_ldData0_data: i32, %in0_ldDone0_valid: i1, %in1_ldAddr0_ready: i1, %in1_ldData0_valid: i1, %in1_ldData0_data: i32, %in1_ldDone0_valid: i1, %inCtrl_valid: i1, %out0_ready: i1, %outCtrl_ready: i1, %clock: i1, %reset: i1) -> (in0_ldAddr0_valid: i1, in0_ldAddr0_data: i64, in0_ldData0_ready: i1, in0_ldDone0_ready: i1, in1_ldAddr0_valid: i1, in1_ldAddr0_data: i64, in1_ldData0_ready: i1, in1_ldDone0_ready: i1, inCtrl_ready: i1, out0_valid: i1, out0_data: i32, outCtrl_valid: i1) {
    %handshake_buffer0.in0_ready, %handshake_buffer0.out0_valid = hw.instance "handshake_buffer0" @handshake_buffer_2slots_seq_1ins_1outs_ctrl(in0_valid: %inCtrl_valid: i1, out0_ready: %handshake_fork0.in0_ready: i1, clock: %clock: i1, reset: %reset: i1) -> (in0_ready: i1, out0_valid: i1)
    %handshake_extmemory0.extmem_ldAddr0_valid, %handshake_extmemory0.extmem_ldAddr0_data, %handshake_extmemory0.extmem_ldData0_ready, %handshake_extmemory0.extmem_ldDone0_ready, %handshake_extmemory0.ldAddr0_ready, %handshake_extmemory0.ldData0_valid, %handshake_extmemory0.ldData0_data, %handshake_extmemory0.ldDone0_valid = hw.instance "handshake_extmemory0" @handshake_extmemory_in_ui64_out_ui32(extmem_ldAddr0_ready: %in1_ldAddr0_ready: i1, extmem_ldData0_valid: %in1_ldData0_valid: i1, extmem_ldData0_data: %in1_ldData0_data: i32, extmem_ldDone0_valid: %in1_ldDone0_valid: i1, ldAddr0_valid: %handshake_buffer56.out0_valid: i1, ldAddr0_data: %handshake_buffer56.out0_data: i64, ldData0_ready: %handshake_buffer2.in0_ready: i1, ldDone0_ready: %handshake_buffer1.in0_ready: i1) -> (extmem_ldAddr0_valid: i1, extmem_ldAddr0_data: i64, extmem_ldData0_ready: i1, extmem_ldDone0_ready: i1, ldAddr0_ready: i1, ldData0_valid: i1, ldData0_data: i32, ldDone0_valid: i1)
    %handshake_buffer1.in0_ready, %handshake_buffer1.out0_valid = hw.instance "handshake_buffer1" @handshake_buffer_2slots_seq_1ins_1outs_ctrl(in0_valid: %handshake_extmemory0.ldDone0_valid: i1, out0_ready: %handshake_join0.in2_ready: i1, clock: %clock: i1, reset: %reset: i1) -> (in0_ready: i1, out0_valid: i1)
    %handshake_buffer2.in0_ready, %handshake_buffer2.out0_valid, %handshake_buffer2.out0_data = hw.instance "handshake_buffer2" @handshake_buffer_in_ui32_out_ui32_2slots_seq(in0_valid: %handshake_extmemory0.ldData0_valid: i1, in0_data: %handshake_extmemory0.ldData0_data: i32, out0_ready: %handshake_load1.dataFromMem_ready: i1, clock: %clock: i1, reset: %reset: i1) -> (in0_ready: i1, out0_valid: i1, out0_data: i32)
    %handshake_extmemory1.extmem_ldAddr0_valid, %handshake_extmemory1.extmem_ldAddr0_data, %handshake_extmemory1.extmem_ldData0_ready, %handshake_extmemory1.extmem_ldDone0_ready, %handshake_extmemory1.ldAddr0_ready, %handshake_extmemory1.ldData0_valid, %handshake_extmemory1.ldData0_data, %handshake_extmemory1.ldDone0_valid = hw.instance "handshake_extmemory1" @handshake_extmemory_in_ui64_out_ui32(extmem_ldAddr0_ready: %in0_ldAddr0_ready: i1, extmem_ldData0_valid: %in0_ldData0_valid: i1, extmem_ldData0_data: %in0_ldData0_data: i32, extmem_ldDone0_valid: %in0_ldDone0_valid: i1, ldAddr0_valid: %handshake_buffer54.out0_valid: i1, ldAddr0_data: %handshake_buffer54.out0_data: i64, ldData0_ready: %handshake_buffer4.in0_ready: i1, ldDone0_ready: %handshake_buffer3.in0_ready: i1) -> (extmem_ldAddr0_valid: i1, extmem_ldAddr0_data: i64, extmem_ldData0_ready: i1, extmem_ldDone0_ready: i1, ldAddr0_ready: i1, ldData0_valid: i1, ldData0_data: i32, ldDone0_valid: i1)
    %handshake_buffer3.in0_ready, %handshake_buffer3.out0_valid = hw.instance "handshake_buffer3" @handshake_buffer_2slots_seq_1ins_1outs_ctrl(in0_valid: %handshake_extmemory1.ldDone0_valid: i1, out0_ready: %handshake_join0.in1_ready: i1, clock: %clock: i1, reset: %reset: i1) -> (in0_ready: i1, out0_valid: i1)
    %handshake_buffer4.in0_ready, %handshake_buffer4.out0_valid, %handshake_buffer4.out0_data = hw.instance "handshake_buffer4" @handshake_buffer_in_ui32_out_ui32_2slots_seq(in0_valid: %handshake_extmemory1.ldData0_valid: i1, in0_data: %handshake_extmemory1.ldData0_data: i32, out0_ready: %handshake_load0.dataFromMem_ready: i1, clock: %clock: i1, reset: %reset: i1) -> (in0_ready: i1, out0_valid: i1, out0_data: i32)
    %handshake_fork0.in0_ready, %handshake_fork0.out0_valid, %handshake_fork0.out1_valid, %handshake_fork0.out2_valid, %handshake_fork0.out3_valid, %handshake_fork0.out4_valid = hw.instance "handshake_fork0" @handshake_fork_1ins_5outs_ctrl(in0_valid: %handshake_buffer0.out0_valid: i1, out0_ready: %handshake_buffer9.in0_ready: i1, out1_ready: %handshake_buffer8.in0_ready: i1, out2_ready: %handshake_buffer7.in0_ready: i1, out3_ready: %handshake_buffer6.in0_ready: i1, out4_ready: %handshake_buffer5.in0_ready: i1, clock: %clock: i1, reset: %reset: i1) -> (in0_ready: i1, out0_valid: i1, out1_valid: i1, out2_valid: i1, out3_valid: i1, out4_valid: i1)
    %handshake_buffer5.in0_ready, %handshake_buffer5.out0_valid = hw.instance "handshake_buffer5" @handshake_buffer_2slots_seq_1ins_1outs_ctrl(in0_valid: %handshake_fork0.out4_valid: i1, out0_ready: %handshake_mux0.in0_ready: i1, clock: %clock: i1, reset: %reset: i1) -> (in0_ready: i1, out0_valid: i1)
    %handshake_buffer6.in0_ready, %handshake_buffer6.out0_valid = hw.instance "handshake_buffer6" @handshake_buffer_2slots_seq_1ins_1outs_ctrl(in0_valid: %handshake_fork0.out3_valid: i1, out0_ready: %handshake_constant0.ctrl_ready: i1, clock: %clock: i1, reset: %reset: i1) -> (in0_ready: i1, out0_valid: i1)
    %handshake_buffer7.in0_ready, %handshake_buffer7.out0_valid = hw.instance "handshake_buffer7" @handshake_buffer_2slots_seq_1ins_1outs_ctrl(in0_valid: %handshake_fork0.out2_valid: i1, out0_ready: %handshake_constant1.ctrl_ready: i1, clock: %clock: i1, reset: %reset: i1) -> (in0_ready: i1, out0_valid: i1)
    %handshake_buffer8.in0_ready, %handshake_buffer8.out0_valid = hw.instance "handshake_buffer8" @handshake_buffer_2slots_seq_1ins_1outs_ctrl(in0_valid: %handshake_fork0.out1_valid: i1, out0_ready: %handshake_constant2.ctrl_ready: i1, clock: %clock: i1, reset: %reset: i1) -> (in0_ready: i1, out0_valid: i1)
    %handshake_buffer9.in0_ready, %handshake_buffer9.out0_valid = hw.instance "handshake_buffer9" @handshake_buffer_2slots_seq_1ins_1outs_ctrl(in0_valid: %handshake_fork0.out0_valid: i1, out0_ready: %handshake_constant3.ctrl_ready: i1, clock: %clock: i1, reset: %reset: i1) -> (in0_ready: i1, out0_valid: i1)
    %handshake_constant0.ctrl_ready, %handshake_constant0.out0_valid, %handshake_constant0.out0_data = hw.instance "handshake_constant0" @handshake_constant_c0_out_ui32(ctrl_valid: %handshake_buffer6.out0_valid: i1, out0_ready: %handshake_buffer10.in0_ready: i1) -> (ctrl_ready: i1, out0_valid: i1, out0_data: i32)
    %handshake_buffer10.in0_ready, %handshake_buffer10.out0_valid, %handshake_buffer10.out0_data = hw.instance "handshake_buffer10" @handshake_buffer_in_ui32_out_ui32_2slots_seq(in0_valid: %handshake_constant0.out0_valid: i1, in0_data: %handshake_constant0.out0_data: i32, out0_ready: %handshake_mux4.in0_ready: i1, clock: %clock: i1, reset: %reset: i1) -> (in0_ready: i1, out0_valid: i1, out0_data: i32)
    %handshake_constant1.ctrl_ready, %handshake_constant1.out0_valid, %handshake_constant1.out0_data = hw.instance "handshake_constant1" @handshake_constant_c0_out_ui64(ctrl_valid: %handshake_buffer7.out0_valid: i1, out0_ready: %handshake_buffer11.in0_ready: i1) -> (ctrl_ready: i1, out0_valid: i1, out0_data: i64)
    %handshake_buffer11.in0_ready, %handshake_buffer11.out0_valid, %handshake_buffer11.out0_data = hw.instance "handshake_buffer11" @handshake_buffer_in_ui64_out_ui64_2slots_seq(in0_valid: %handshake_constant1.out0_valid: i1, in0_data: %handshake_constant1.out0_data: i64, out0_ready: %handshake_mux3.in0_ready: i1, clock: %clock: i1, reset: %reset: i1) -> (in0_ready: i1, out0_valid: i1, out0_data: i64)
    %handshake_constant2.ctrl_ready, %handshake_constant2.out0_valid, %handshake_constant2.out0_data = hw.instance "handshake_constant2" @handshake_constant_c5_out_ui64(ctrl_valid: %handshake_buffer8.out0_valid: i1, out0_ready: %handshake_buffer12.in0_ready: i1) -> (ctrl_ready: i1, out0_valid: i1, out0_data: i64)
    %handshake_buffer12.in0_ready, %handshake_buffer12.out0_valid, %handshake_buffer12.out0_data = hw.instance "handshake_buffer12" @handshake_buffer_in_ui64_out_ui64_2slots_seq(in0_valid: %handshake_constant2.out0_valid: i1, in0_data: %handshake_constant2.out0_data: i64, out0_ready: %handshake_mux1.in0_ready: i1, clock: %clock: i1, reset: %reset: i1) -> (in0_ready: i1, out0_valid: i1, out0_data: i64)
    %handshake_constant3.ctrl_ready, %handshake_constant3.out0_valid, %handshake_constant3.out0_data = hw.instance "handshake_constant3" @handshake_constant_c1_out_ui64(ctrl_valid: %handshake_buffer9.out0_valid: i1, out0_ready: %handshake_buffer13.in0_ready: i1) -> (ctrl_ready: i1, out0_valid: i1, out0_data: i64)
    %handshake_buffer13.in0_ready, %handshake_buffer13.out0_valid, %handshake_buffer13.out0_data = hw.instance "handshake_buffer13" @handshake_buffer_in_ui64_out_ui64_2slots_seq(in0_valid: %handshake_constant3.out0_valid: i1, in0_data: %handshake_constant3.out0_data: i64, out0_ready: %handshake_mux2.in0_ready: i1, clock: %clock: i1, reset: %reset: i1) -> (in0_ready: i1, out0_valid: i1, out0_data: i64)
    %handshake_buffer14.in0_ready, %handshake_buffer14.out0_valid, %handshake_buffer14.out0_data = hw.instance "handshake_buffer14" @handshake_buffer_in_ui1_out_ui1_1slots_seq_init_0(in0_valid: %handshake_fork4.out0_valid: i1, in0_data: %handshake_fork4.out0_data: i1, out0_ready: %handshake_fork1.in0_ready: i1, clock: %clock: i1, reset: %reset: i1) -> (in0_ready: i1, out0_valid: i1, out0_data: i1)
    %handshake_fork1.in0_ready, %handshake_fork1.out0_valid, %handshake_fork1.out0_data, %handshake_fork1.out1_valid, %handshake_fork1.out1_data, %handshake_fork1.out2_valid, %handshake_fork1.out2_data, %handshake_fork1.out3_valid, %handshake_fork1.out3_data, %handshake_fork1.out4_valid, %handshake_fork1.out4_data = hw.instance "handshake_fork1" @handshake_fork_in_ui1_out_ui1_ui1_ui1_ui1_ui1(in0_valid: %handshake_buffer14.out0_valid: i1, in0_data: %handshake_buffer14.out0_data: i1, out0_ready: %handshake_buffer19.in0_ready: i1, out1_ready: %handshake_buffer18.in0_ready: i1, out2_ready: %handshake_buffer17.in0_ready: i1, out3_ready: %handshake_buffer16.in0_ready: i1, out4_ready: %handshake_buffer15.in0_ready: i1, clock: %clock: i1, reset: %reset: i1) -> (in0_ready: i1, out0_valid: i1, out0_data: i1, out1_valid: i1, out1_data: i1, out2_valid: i1, out2_data: i1, out3_valid: i1, out3_data: i1, out4_valid: i1, out4_data: i1)
    %handshake_buffer15.in0_ready, %handshake_buffer15.out0_valid, %handshake_buffer15.out0_data = hw.instance "handshake_buffer15" @handshake_buffer_in_ui1_out_ui1_2slots_seq(in0_valid: %handshake_fork1.out4_valid: i1, in0_data: %handshake_fork1.out4_data: i1, out0_ready: %handshake_mux0.select_ready: i1, clock: %clock: i1, reset: %reset: i1) -> (in0_ready: i1, out0_valid: i1, out0_data: i1)
    %handshake_buffer16.in0_ready, %handshake_buffer16.out0_valid, %handshake_buffer16.out0_data = hw.instance "handshake_buffer16" @handshake_buffer_in_ui1_out_ui1_2slots_seq(in0_valid: %handshake_fork1.out3_valid: i1, in0_data: %handshake_fork1.out3_data: i1, out0_ready: %handshake_mux1.select_ready: i1, clock: %clock: i1, reset: %reset: i1) -> (in0_ready: i1, out0_valid: i1, out0_data: i1)
    %handshake_buffer17.in0_ready, %handshake_buffer17.out0_valid, %handshake_buffer17.out0_data = hw.instance "handshake_buffer17" @handshake_buffer_in_ui1_out_ui1_2slots_seq(in0_valid: %handshake_fork1.out2_valid: i1, in0_data: %handshake_fork1.out2_data: i1, out0_ready: %handshake_mux2.select_ready: i1, clock: %clock: i1, reset: %reset: i1) -> (in0_ready: i1, out0_valid: i1, out0_data: i1)
    %handshake_buffer18.in0_ready, %handshake_buffer18.out0_valid, %handshake_buffer18.out0_data = hw.instance "handshake_buffer18" @handshake_buffer_in_ui1_out_ui1_2slots_seq(in0_valid: %handshake_fork1.out1_valid: i1, in0_data: %handshake_fork1.out1_data: i1, out0_ready: %handshake_mux3.select_ready: i1, clock: %clock: i1, reset: %reset: i1) -> (in0_ready: i1, out0_valid: i1, out0_data: i1)
    %handshake_buffer19.in0_ready, %handshake_buffer19.out0_valid, %handshake_buffer19.out0_data = hw.instance "handshake_buffer19" @handshake_buffer_in_ui1_out_ui1_2slots_seq(in0_valid: %handshake_fork1.out0_valid: i1, in0_data: %handshake_fork1.out0_data: i1, out0_ready: %handshake_mux4.select_ready: i1, clock: %clock: i1, reset: %reset: i1) -> (in0_ready: i1, out0_valid: i1, out0_data: i1)
    %handshake_mux0.select_ready, %handshake_mux0.in0_ready, %handshake_mux0.in1_ready, %handshake_mux0.out0_valid = hw.instance "handshake_mux0" @handshake_mux_in_ui1_3ins_1outs_ctrl(select_valid: %handshake_buffer15.out0_valid: i1, select_data: %handshake_buffer15.out0_data: i1, in0_valid: %handshake_buffer5.out0_valid: i1, in1_valid: %handshake_buffer53.out0_valid: i1, out0_ready: %handshake_buffer20.in0_ready: i1) -> (select_ready: i1, in0_ready: i1, in1_ready: i1, out0_valid: i1)
    %handshake_buffer20.in0_ready, %handshake_buffer20.out0_valid = hw.instance "handshake_buffer20" @handshake_buffer_2slots_seq_1ins_1outs_ctrl(in0_valid: %handshake_mux0.out0_valid: i1, out0_ready: %handshake_cond_br2.data_ready: i1, clock: %clock: i1, reset: %reset: i1) -> (in0_ready: i1, out0_valid: i1)
    %handshake_mux1.select_ready, %handshake_mux1.in0_ready, %handshake_mux1.in1_ready, %handshake_mux1.out0_valid, %handshake_mux1.out0_data = hw.instance "handshake_mux1" @handshake_mux_in_ui1_ui64_ui64_out_ui64(select_valid: %handshake_buffer16.out0_valid: i1, select_data: %handshake_buffer16.out0_data: i1, in0_valid: %handshake_buffer12.out0_valid: i1, in0_data: %handshake_buffer12.out0_data: i64, in1_valid: %handshake_buffer36.out0_valid: i1, in1_data: %handshake_buffer36.out0_data: i64, out0_ready: %handshake_buffer21.in0_ready: i1) -> (select_ready: i1, in0_ready: i1, in1_ready: i1, out0_valid: i1, out0_data: i64)
    %handshake_buffer21.in0_ready, %handshake_buffer21.out0_valid, %handshake_buffer21.out0_data = hw.instance "handshake_buffer21" @handshake_buffer_in_ui64_out_ui64_2slots_seq(in0_valid: %handshake_mux1.out0_valid: i1, in0_data: %handshake_mux1.out0_data: i64, out0_ready: %handshake_fork2.in0_ready: i1, clock: %clock: i1, reset: %reset: i1) -> (in0_ready: i1, out0_valid: i1, out0_data: i64)
    %handshake_fork2.in0_ready, %handshake_fork2.out0_valid, %handshake_fork2.out0_data, %handshake_fork2.out1_valid, %handshake_fork2.out1_data = hw.instance "handshake_fork2" @handshake_fork_in_ui64_out_ui64_ui64(in0_valid: %handshake_buffer21.out0_valid: i1, in0_data: %handshake_buffer21.out0_data: i64, out0_ready: %handshake_buffer23.in0_ready: i1, out1_ready: %handshake_buffer22.in0_ready: i1, clock: %clock: i1, reset: %reset: i1) -> (in0_ready: i1, out0_valid: i1, out0_data: i64, out1_valid: i1, out1_data: i64)
    %handshake_buffer22.in0_ready, %handshake_buffer22.out0_valid, %handshake_buffer22.out0_data = hw.instance "handshake_buffer22" @handshake_buffer_in_ui64_out_ui64_2slots_seq(in0_valid: %handshake_fork2.out1_valid: i1, in0_data: %handshake_fork2.out1_data: i64, out0_ready: %handshake_cond_br0.data_ready: i1, clock: %clock: i1, reset: %reset: i1) -> (in0_ready: i1, out0_valid: i1, out0_data: i64)
    %handshake_buffer23.in0_ready, %handshake_buffer23.out0_valid, %handshake_buffer23.out0_data = hw.instance "handshake_buffer23" @handshake_buffer_in_ui64_out_ui64_2slots_seq(in0_valid: %handshake_fork2.out0_valid: i1, in0_data: %handshake_fork2.out0_data: i64, out0_ready: %arith_cmpi0.in1_ready: i1, clock: %clock: i1, reset: %reset: i1) -> (in0_ready: i1, out0_valid: i1, out0_data: i64)
    %handshake_mux2.select_ready, %handshake_mux2.in0_ready, %handshake_mux2.in1_ready, %handshake_mux2.out0_valid, %handshake_mux2.out0_data = hw.instance "handshake_mux2" @handshake_mux_in_ui1_ui64_ui64_out_ui64(select_valid: %handshake_buffer17.out0_valid: i1, select_data: %handshake_buffer17.out0_data: i1, in0_valid: %handshake_buffer13.out0_valid: i1, in0_data: %handshake_buffer13.out0_data: i64, in1_valid: %handshake_buffer49.out0_valid: i1, in1_data: %handshake_buffer49.out0_data: i64, out0_ready: %handshake_buffer24.in0_ready: i1) -> (select_ready: i1, in0_ready: i1, in1_ready: i1, out0_valid: i1, out0_data: i64)
    %handshake_buffer24.in0_ready, %handshake_buffer24.out0_valid, %handshake_buffer24.out0_data = hw.instance "handshake_buffer24" @handshake_buffer_in_ui64_out_ui64_2slots_seq(in0_valid: %handshake_mux2.out0_valid: i1, in0_data: %handshake_mux2.out0_data: i64, out0_ready: %handshake_cond_br1.data_ready: i1, clock: %clock: i1, reset: %reset: i1) -> (in0_ready: i1, out0_valid: i1, out0_data: i64)
    %handshake_mux3.select_ready, %handshake_mux3.in0_ready, %handshake_mux3.in1_ready, %handshake_mux3.out0_valid, %handshake_mux3.out0_data = hw.instance "handshake_mux3" @handshake_mux_in_ui1_ui64_ui64_out_ui64(select_valid: %handshake_buffer18.out0_valid: i1, select_data: %handshake_buffer18.out0_data: i1, in0_valid: %handshake_buffer11.out0_valid: i1, in0_data: %handshake_buffer11.out0_data: i64, in1_valid: %handshake_buffer60.out0_valid: i1, in1_data: %handshake_buffer60.out0_data: i64, out0_ready: %handshake_buffer25.in0_ready: i1) -> (select_ready: i1, in0_ready: i1, in1_ready: i1, out0_valid: i1, out0_data: i64)
    %handshake_buffer25.in0_ready, %handshake_buffer25.out0_valid, %handshake_buffer25.out0_data = hw.instance "handshake_buffer25" @handshake_buffer_in_ui64_out_ui64_2slots_seq(in0_valid: %handshake_mux3.out0_valid: i1, in0_data: %handshake_mux3.out0_data: i64, out0_ready: %handshake_fork3.in0_ready: i1, clock: %clock: i1, reset: %reset: i1) -> (in0_ready: i1, out0_valid: i1, out0_data: i64)
    %handshake_fork3.in0_ready, %handshake_fork3.out0_valid, %handshake_fork3.out0_data, %handshake_fork3.out1_valid, %handshake_fork3.out1_data = hw.instance "handshake_fork3" @handshake_fork_in_ui64_out_ui64_ui64(in0_valid: %handshake_buffer25.out0_valid: i1, in0_data: %handshake_buffer25.out0_data: i64, out0_ready: %handshake_buffer27.in0_ready: i1, out1_ready: %handshake_buffer26.in0_ready: i1, clock: %clock: i1, reset: %reset: i1) -> (in0_ready: i1, out0_valid: i1, out0_data: i64, out1_valid: i1, out1_data: i64)
    %handshake_buffer26.in0_ready, %handshake_buffer26.out0_valid, %handshake_buffer26.out0_data = hw.instance "handshake_buffer26" @handshake_buffer_in_ui64_out_ui64_2slots_seq(in0_valid: %handshake_fork3.out1_valid: i1, in0_data: %handshake_fork3.out1_data: i64, out0_ready: %handshake_cond_br3.data_ready: i1, clock: %clock: i1, reset: %reset: i1) -> (in0_ready: i1, out0_valid: i1, out0_data: i64)
    %handshake_buffer27.in0_ready, %handshake_buffer27.out0_valid, %handshake_buffer27.out0_data = hw.instance "handshake_buffer27" @handshake_buffer_in_ui64_out_ui64_2slots_seq(in0_valid: %handshake_fork3.out0_valid: i1, in0_data: %handshake_fork3.out0_data: i64, out0_ready: %arith_cmpi0.in0_ready: i1, clock: %clock: i1, reset: %reset: i1) -> (in0_ready: i1, out0_valid: i1, out0_data: i64)
    %handshake_mux4.select_ready, %handshake_mux4.in0_ready, %handshake_mux4.in1_ready, %handshake_mux4.out0_valid, %handshake_mux4.out0_data = hw.instance "handshake_mux4" @handshake_mux_in_ui1_ui32_ui32_out_ui32(select_valid: %handshake_buffer19.out0_valid: i1, select_data: %handshake_buffer19.out0_data: i1, in0_valid: %handshake_buffer10.out0_valid: i1, in0_data: %handshake_buffer10.out0_data: i32, in1_valid: %handshake_buffer59.out0_valid: i1, in1_data: %handshake_buffer59.out0_data: i32, out0_ready: %handshake_buffer28.in0_ready: i1) -> (select_ready: i1, in0_ready: i1, in1_ready: i1, out0_valid: i1, out0_data: i32)
    %handshake_buffer28.in0_ready, %handshake_buffer28.out0_valid, %handshake_buffer28.out0_data = hw.instance "handshake_buffer28" @handshake_buffer_in_ui32_out_ui32_2slots_seq(in0_valid: %handshake_mux4.out0_valid: i1, in0_data: %handshake_mux4.out0_data: i32, out0_ready: %handshake_cond_br4.data_ready: i1, clock: %clock: i1, reset: %reset: i1) -> (in0_ready: i1, out0_valid: i1, out0_data: i32)
    %arith_cmpi0.in0_ready, %arith_cmpi0.in1_ready, %arith_cmpi0.out0_valid, %arith_cmpi0.out0_data = hw.instance "arith_cmpi0" @arith_cmpi_in_ui64_ui64_out_ui1_slt(in0_valid: %handshake_buffer27.out0_valid: i1, in0_data: %handshake_buffer27.out0_data: i64, in1_valid: %handshake_buffer23.out0_valid: i1, in1_data: %handshake_buffer23.out0_data: i64, out0_ready: %handshake_buffer29.in0_ready: i1) -> (in0_ready: i1, in1_ready: i1, out0_valid: i1, out0_data: i1)
    %handshake_buffer29.in0_ready, %handshake_buffer29.out0_valid, %handshake_buffer29.out0_data = hw.instance "handshake_buffer29" @handshake_buffer_in_ui1_out_ui1_2slots_seq(in0_valid: %arith_cmpi0.out0_valid: i1, in0_data: %arith_cmpi0.out0_data: i1, out0_ready: %handshake_fork4.in0_ready: i1, clock: %clock: i1, reset: %reset: i1) -> (in0_ready: i1, out0_valid: i1, out0_data: i1)
    %handshake_fork4.in0_ready, %handshake_fork4.out0_valid, %handshake_fork4.out0_data, %handshake_fork4.out1_valid, %handshake_fork4.out1_data, %handshake_fork4.out2_valid, %handshake_fork4.out2_data, %handshake_fork4.out3_valid, %handshake_fork4.out3_data, %handshake_fork4.out4_valid, %handshake_fork4.out4_data, %handshake_fork4.out5_valid, %handshake_fork4.out5_data = hw.instance "handshake_fork4" @handshake_fork_in_ui1_out_ui1_ui1_ui1_ui1_ui1_ui1(in0_valid: %handshake_buffer29.out0_valid: i1, in0_data: %handshake_buffer29.out0_data: i1, out0_ready: %handshake_buffer14.in0_ready: i1, out1_ready: %handshake_buffer34.in0_ready: i1, out2_ready: %handshake_buffer33.in0_ready: i1, out3_ready: %handshake_buffer32.in0_ready: i1, out4_ready: %handshake_buffer31.in0_ready: i1, out5_ready: %handshake_buffer30.in0_ready: i1, clock: %clock: i1, reset: %reset: i1) -> (in0_ready: i1, out0_valid: i1, out0_data: i1, out1_valid: i1, out1_data: i1, out2_valid: i1, out2_data: i1, out3_valid: i1, out3_data: i1, out4_valid: i1, out4_data: i1, out5_valid: i1, out5_data: i1)
    %handshake_buffer30.in0_ready, %handshake_buffer30.out0_valid, %handshake_buffer30.out0_data = hw.instance "handshake_buffer30" @handshake_buffer_in_ui1_out_ui1_2slots_seq(in0_valid: %handshake_fork4.out5_valid: i1, in0_data: %handshake_fork4.out5_data: i1, out0_ready: %handshake_cond_br0.cond_ready: i1, clock: %clock: i1, reset: %reset: i1) -> (in0_ready: i1, out0_valid: i1, out0_data: i1)
    %handshake_buffer31.in0_ready, %handshake_buffer31.out0_valid, %handshake_buffer31.out0_data = hw.instance "handshake_buffer31" @handshake_buffer_in_ui1_out_ui1_2slots_seq(in0_valid: %handshake_fork4.out4_valid: i1, in0_data: %handshake_fork4.out4_data: i1, out0_ready: %handshake_cond_br1.cond_ready: i1, clock: %clock: i1, reset: %reset: i1) -> (in0_ready: i1, out0_valid: i1, out0_data: i1)
    %handshake_buffer32.in0_ready, %handshake_buffer32.out0_valid, %handshake_buffer32.out0_data = hw.instance "handshake_buffer32" @handshake_buffer_in_ui1_out_ui1_2slots_seq(in0_valid: %handshake_fork4.out3_valid: i1, in0_data: %handshake_fork4.out3_data: i1, out0_ready: %handshake_cond_br2.cond_ready: i1, clock: %clock: i1, reset: %reset: i1) -> (in0_ready: i1, out0_valid: i1, out0_data: i1)
    %handshake_buffer33.in0_ready, %handshake_buffer33.out0_valid, %handshake_buffer33.out0_data = hw.instance "handshake_buffer33" @handshake_buffer_in_ui1_out_ui1_2slots_seq(in0_valid: %handshake_fork4.out2_valid: i1, in0_data: %handshake_fork4.out2_data: i1, out0_ready: %handshake_cond_br3.cond_ready: i1, clock: %clock: i1, reset: %reset: i1) -> (in0_ready: i1, out0_valid: i1, out0_data: i1)
    %handshake_buffer34.in0_ready, %handshake_buffer34.out0_valid, %handshake_buffer34.out0_data = hw.instance "handshake_buffer34" @handshake_buffer_in_ui1_out_ui1_2slots_seq(in0_valid: %handshake_fork4.out1_valid: i1, in0_data: %handshake_fork4.out1_data: i1, out0_ready: %handshake_cond_br4.cond_ready: i1, clock: %clock: i1, reset: %reset: i1) -> (in0_ready: i1, out0_valid: i1, out0_data: i1)
    %handshake_cond_br0.cond_ready, %handshake_cond_br0.data_ready, %handshake_cond_br0.outTrue_valid, %handshake_cond_br0.outTrue_data, %handshake_cond_br0.outFalse_valid, %handshake_cond_br0.outFalse_data = hw.instance "handshake_cond_br0" @handshake_cond_br_in_ui1_ui64_out_ui64_ui64(cond_valid: %handshake_buffer30.out0_valid: i1, cond_data: %handshake_buffer30.out0_data: i1, data_valid: %handshake_buffer22.out0_valid: i1, data_data: %handshake_buffer22.out0_data: i64, outTrue_ready: %handshake_buffer36.in0_ready: i1, outFalse_ready: %handshake_buffer35.in0_ready: i1) -> (cond_ready: i1, data_ready: i1, outTrue_valid: i1, outTrue_data: i64, outFalse_valid: i1, outFalse_data: i64)
    %handshake_buffer35.in0_ready, %handshake_buffer35.out0_valid, %handshake_buffer35.out0_data = hw.instance "handshake_buffer35" @handshake_buffer_in_ui64_out_ui64_2slots_seq(in0_valid: %handshake_cond_br0.outFalse_valid: i1, in0_data: %handshake_cond_br0.outFalse_data: i64, out0_ready: %handshake_sink0.in0_ready: i1, clock: %clock: i1, reset: %reset: i1) -> (in0_ready: i1, out0_valid: i1, out0_data: i64)
    %handshake_buffer36.in0_ready, %handshake_buffer36.out0_valid, %handshake_buffer36.out0_data = hw.instance "handshake_buffer36" @handshake_buffer_in_ui64_out_ui64_2slots_seq(in0_valid: %handshake_cond_br0.outTrue_valid: i1, in0_data: %handshake_cond_br0.outTrue_data: i64, out0_ready: %handshake_mux1.in1_ready: i1, clock: %clock: i1, reset: %reset: i1) -> (in0_ready: i1, out0_valid: i1, out0_data: i64)
    %handshake_sink0.in0_ready = hw.instance "handshake_sink0" @handshake_sink_in_ui64(in0_valid: %handshake_buffer35.out0_valid: i1, in0_data: %handshake_buffer35.out0_data: i64) -> (in0_ready: i1)
    %handshake_cond_br1.cond_ready, %handshake_cond_br1.data_ready, %handshake_cond_br1.outTrue_valid, %handshake_cond_br1.outTrue_data, %handshake_cond_br1.outFalse_valid, %handshake_cond_br1.outFalse_data = hw.instance "handshake_cond_br1" @handshake_cond_br_in_ui1_ui64_out_ui64_ui64(cond_valid: %handshake_buffer31.out0_valid: i1, cond_data: %handshake_buffer31.out0_data: i1, data_valid: %handshake_buffer24.out0_valid: i1, data_data: %handshake_buffer24.out0_data: i64, outTrue_ready: %handshake_buffer38.in0_ready: i1, outFalse_ready: %handshake_buffer37.in0_ready: i1) -> (cond_ready: i1, data_ready: i1, outTrue_valid: i1, outTrue_data: i64, outFalse_valid: i1, outFalse_data: i64)
    %handshake_buffer37.in0_ready, %handshake_buffer37.out0_valid, %handshake_buffer37.out0_data = hw.instance "handshake_buffer37" @handshake_buffer_in_ui64_out_ui64_2slots_seq(in0_valid: %handshake_cond_br1.outFalse_valid: i1, in0_data: %handshake_cond_br1.outFalse_data: i64, out0_ready: %handshake_sink1.in0_ready: i1, clock: %clock: i1, reset: %reset: i1) -> (in0_ready: i1, out0_valid: i1, out0_data: i64)
    %handshake_buffer38.in0_ready, %handshake_buffer38.out0_valid, %handshake_buffer38.out0_data = hw.instance "handshake_buffer38" @handshake_buffer_in_ui64_out_ui64_2slots_seq(in0_valid: %handshake_cond_br1.outTrue_valid: i1, in0_data: %handshake_cond_br1.outTrue_data: i64, out0_ready: %handshake_fork6.in0_ready: i1, clock: %clock: i1, reset: %reset: i1) -> (in0_ready: i1, out0_valid: i1, out0_data: i64)
    %handshake_sink1.in0_ready = hw.instance "handshake_sink1" @handshake_sink_in_ui64(in0_valid: %handshake_buffer37.out0_valid: i1, in0_data: %handshake_buffer37.out0_data: i64) -> (in0_ready: i1)
    %handshake_cond_br2.cond_ready, %handshake_cond_br2.data_ready, %handshake_cond_br2.outTrue_valid, %handshake_cond_br2.outFalse_valid = hw.instance "handshake_cond_br2" @handshake_cond_br_in_ui1_2ins_2outs_ctrl(cond_valid: %handshake_buffer32.out0_valid: i1, cond_data: %handshake_buffer32.out0_data: i1, data_valid: %handshake_buffer20.out0_valid: i1, outTrue_ready: %handshake_buffer40.in0_ready: i1, outFalse_ready: %handshake_buffer39.in0_ready: i1) -> (cond_ready: i1, data_ready: i1, outTrue_valid: i1, outFalse_valid: i1)
    %handshake_buffer39.in0_ready, %handshake_buffer39.out0_valid = hw.instance "handshake_buffer39" @handshake_buffer_2slots_seq_1ins_1outs_ctrl(in0_valid: %handshake_cond_br2.outFalse_valid: i1, out0_ready: %outCtrl_ready: i1, clock: %clock: i1, reset: %reset: i1) -> (in0_ready: i1, out0_valid: i1)
    %handshake_buffer40.in0_ready, %handshake_buffer40.out0_valid = hw.instance "handshake_buffer40" @handshake_buffer_2slots_seq_1ins_1outs_ctrl(in0_valid: %handshake_cond_br2.outTrue_valid: i1, out0_ready: %handshake_fork7.in0_ready: i1, clock: %clock: i1, reset: %reset: i1) -> (in0_ready: i1, out0_valid: i1)
    %handshake_cond_br3.cond_ready, %handshake_cond_br3.data_ready, %handshake_cond_br3.outTrue_valid, %handshake_cond_br3.outTrue_data, %handshake_cond_br3.outFalse_valid, %handshake_cond_br3.outFalse_data = hw.instance "handshake_cond_br3" @handshake_cond_br_in_ui1_ui64_out_ui64_ui64(cond_valid: %handshake_buffer33.out0_valid: i1, cond_data: %handshake_buffer33.out0_data: i1, data_valid: %handshake_buffer26.out0_valid: i1, data_data: %handshake_buffer26.out0_data: i64, outTrue_ready: %handshake_buffer42.in0_ready: i1, outFalse_ready: %handshake_buffer41.in0_ready: i1) -> (cond_ready: i1, data_ready: i1, outTrue_valid: i1, outTrue_data: i64, outFalse_valid: i1, outFalse_data: i64)
    %handshake_buffer41.in0_ready, %handshake_buffer41.out0_valid, %handshake_buffer41.out0_data = hw.instance "handshake_buffer41" @handshake_buffer_in_ui64_out_ui64_2slots_seq(in0_valid: %handshake_cond_br3.outFalse_valid: i1, in0_data: %handshake_cond_br3.outFalse_data: i64, out0_ready: %handshake_sink2.in0_ready: i1, clock: %clock: i1, reset: %reset: i1) -> (in0_ready: i1, out0_valid: i1, out0_data: i64)
    %handshake_buffer42.in0_ready, %handshake_buffer42.out0_valid, %handshake_buffer42.out0_data = hw.instance "handshake_buffer42" @handshake_buffer_in_ui64_out_ui64_2slots_seq(in0_valid: %handshake_cond_br3.outTrue_valid: i1, in0_data: %handshake_cond_br3.outTrue_data: i64, out0_ready: %handshake_fork5.in0_ready: i1, clock: %clock: i1, reset: %reset: i1) -> (in0_ready: i1, out0_valid: i1, out0_data: i64)
    %handshake_sink2.in0_ready = hw.instance "handshake_sink2" @handshake_sink_in_ui64(in0_valid: %handshake_buffer41.out0_valid: i1, in0_data: %handshake_buffer41.out0_data: i64) -> (in0_ready: i1)
    %handshake_cond_br4.cond_ready, %handshake_cond_br4.data_ready, %handshake_cond_br4.outTrue_valid, %handshake_cond_br4.outTrue_data, %handshake_cond_br4.outFalse_valid, %handshake_cond_br4.outFalse_data = hw.instance "handshake_cond_br4" @handshake_cond_br_in_ui1_ui32_out_ui32_ui32(cond_valid: %handshake_buffer34.out0_valid: i1, cond_data: %handshake_buffer34.out0_data: i1, data_valid: %handshake_buffer28.out0_valid: i1, data_data: %handshake_buffer28.out0_data: i32, outTrue_ready: %handshake_buffer44.in0_ready: i1, outFalse_ready: %handshake_buffer43.in0_ready: i1) -> (cond_ready: i1, data_ready: i1, outTrue_valid: i1, outTrue_data: i32, outFalse_valid: i1, outFalse_data: i32)
    %handshake_buffer43.in0_ready, %handshake_buffer43.out0_valid, %handshake_buffer43.out0_data = hw.instance "handshake_buffer43" @handshake_buffer_in_ui32_out_ui32_2slots_seq(in0_valid: %handshake_cond_br4.outFalse_valid: i1, in0_data: %handshake_cond_br4.outFalse_data: i32, out0_ready: %out0_ready: i1, clock: %clock: i1, reset: %reset: i1) -> (in0_ready: i1, out0_valid: i1, out0_data: i32)
    %handshake_buffer44.in0_ready, %handshake_buffer44.out0_valid, %handshake_buffer44.out0_data = hw.instance "handshake_buffer44" @handshake_buffer_in_ui32_out_ui32_2slots_seq(in0_valid: %handshake_cond_br4.outTrue_valid: i1, in0_data: %handshake_cond_br4.outTrue_data: i32, out0_ready: %arith_addi0.in0_ready: i1, clock: %clock: i1, reset: %reset: i1) -> (in0_ready: i1, out0_valid: i1, out0_data: i32)
    %handshake_fork5.in0_ready, %handshake_fork5.out0_valid, %handshake_fork5.out0_data, %handshake_fork5.out1_valid, %handshake_fork5.out1_data, %handshake_fork5.out2_valid, %handshake_fork5.out2_data = hw.instance "handshake_fork5" @handshake_fork_in_ui64_out_ui64_ui64_ui64(in0_valid: %handshake_buffer42.out0_valid: i1, in0_data: %handshake_buffer42.out0_data: i64, out0_ready: %handshake_buffer47.in0_ready: i1, out1_ready: %handshake_buffer46.in0_ready: i1, out2_ready: %handshake_buffer45.in0_ready: i1, clock: %clock: i1, reset: %reset: i1) -> (in0_ready: i1, out0_valid: i1, out0_data: i64, out1_valid: i1, out1_data: i64, out2_valid: i1, out2_data: i64)
    %handshake_buffer45.in0_ready, %handshake_buffer45.out0_valid, %handshake_buffer45.out0_data = hw.instance "handshake_buffer45" @handshake_buffer_in_ui64_out_ui64_2slots_seq(in0_valid: %handshake_fork5.out2_valid: i1, in0_data: %handshake_fork5.out2_data: i64, out0_ready: %handshake_load0.addrIn0_ready: i1, clock: %clock: i1, reset: %reset: i1) -> (in0_ready: i1, out0_valid: i1, out0_data: i64)
    %handshake_buffer46.in0_ready, %handshake_buffer46.out0_valid, %handshake_buffer46.out0_data = hw.instance "handshake_buffer46" @handshake_buffer_in_ui64_out_ui64_2slots_seq(in0_valid: %handshake_fork5.out1_valid: i1, in0_data: %handshake_fork5.out1_data: i64, out0_ready: %handshake_load1.addrIn0_ready: i1, clock: %clock: i1, reset: %reset: i1) -> (in0_ready: i1, out0_valid: i1, out0_data: i64)
    %handshake_buffer47.in0_ready, %handshake_buffer47.out0_valid, %handshake_buffer47.out0_data = hw.instance "handshake_buffer47" @handshake_buffer_in_ui64_out_ui64_2slots_seq(in0_valid: %handshake_fork5.out0_valid: i1, in0_data: %handshake_fork5.out0_data: i64, out0_ready: %arith_addi1.in0_ready: i1, clock: %clock: i1, reset: %reset: i1) -> (in0_ready: i1, out0_valid: i1, out0_data: i64)
    %handshake_fork6.in0_ready, %handshake_fork6.out0_valid, %handshake_fork6.out0_data, %handshake_fork6.out1_valid, %handshake_fork6.out1_data = hw.instance "handshake_fork6" @handshake_fork_in_ui64_out_ui64_ui64(in0_valid: %handshake_buffer38.out0_valid: i1, in0_data: %handshake_buffer38.out0_data: i64, out0_ready: %handshake_buffer49.in0_ready: i1, out1_ready: %handshake_buffer48.in0_ready: i1, clock: %clock: i1, reset: %reset: i1) -> (in0_ready: i1, out0_valid: i1, out0_data: i64, out1_valid: i1, out1_data: i64)
    %handshake_buffer48.in0_ready, %handshake_buffer48.out0_valid, %handshake_buffer48.out0_data = hw.instance "handshake_buffer48" @handshake_buffer_in_ui64_out_ui64_2slots_seq(in0_valid: %handshake_fork6.out1_valid: i1, in0_data: %handshake_fork6.out1_data: i64, out0_ready: %arith_addi1.in1_ready: i1, clock: %clock: i1, reset: %reset: i1) -> (in0_ready: i1, out0_valid: i1, out0_data: i64)
    %handshake_buffer49.in0_ready, %handshake_buffer49.out0_valid, %handshake_buffer49.out0_data = hw.instance "handshake_buffer49" @handshake_buffer_in_ui64_out_ui64_2slots_seq(in0_valid: %handshake_fork6.out0_valid: i1, in0_data: %handshake_fork6.out0_data: i64, out0_ready: %handshake_mux2.in1_ready: i1, clock: %clock: i1, reset: %reset: i1) -> (in0_ready: i1, out0_valid: i1, out0_data: i64)
    %handshake_fork7.in0_ready, %handshake_fork7.out0_valid, %handshake_fork7.out1_valid, %handshake_fork7.out2_valid = hw.instance "handshake_fork7" @handshake_fork_1ins_3outs_ctrl(in0_valid: %handshake_buffer40.out0_valid: i1, out0_ready: %handshake_buffer52.in0_ready: i1, out1_ready: %handshake_buffer51.in0_ready: i1, out2_ready: %handshake_buffer50.in0_ready: i1, clock: %clock: i1, reset: %reset: i1) -> (in0_ready: i1, out0_valid: i1, out1_valid: i1, out2_valid: i1)
    %handshake_buffer50.in0_ready, %handshake_buffer50.out0_valid = hw.instance "handshake_buffer50" @handshake_buffer_2slots_seq_1ins_1outs_ctrl(in0_valid: %handshake_fork7.out2_valid: i1, out0_ready: %handshake_load0.ctrl_ready: i1, clock: %clock: i1, reset: %reset: i1) -> (in0_ready: i1, out0_valid: i1)
    %handshake_buffer51.in0_ready, %handshake_buffer51.out0_valid = hw.instance "handshake_buffer51" @handshake_buffer_2slots_seq_1ins_1outs_ctrl(in0_valid: %handshake_fork7.out1_valid: i1, out0_ready: %handshake_join0.in0_ready: i1, clock: %clock: i1, reset: %reset: i1) -> (in0_ready: i1, out0_valid: i1)
    %handshake_buffer52.in0_ready, %handshake_buffer52.out0_valid = hw.instance "handshake_buffer52" @handshake_buffer_2slots_seq_1ins_1outs_ctrl(in0_valid: %handshake_fork7.out0_valid: i1, out0_ready: %handshake_load1.ctrl_ready: i1, clock: %clock: i1, reset: %reset: i1) -> (in0_ready: i1, out0_valid: i1)
    %handshake_join0.in0_ready, %handshake_join0.in1_ready, %handshake_join0.in2_ready, %handshake_join0.out0_valid = hw.instance "handshake_join0" @handshake_join_3ins_1outs_ctrl(in0_valid: %handshake_buffer51.out0_valid: i1, in1_valid: %handshake_buffer3.out0_valid: i1, in2_valid: %handshake_buffer1.out0_valid: i1, out0_ready: %handshake_buffer53.in0_ready: i1) -> (in0_ready: i1, in1_ready: i1, in2_ready: i1, out0_valid: i1)
    %handshake_buffer53.in0_ready, %handshake_buffer53.out0_valid = hw.instance "handshake_buffer53" @handshake_buffer_2slots_seq_1ins_1outs_ctrl(in0_valid: %handshake_join0.out0_valid: i1, out0_ready: %handshake_mux0.in1_ready: i1, clock: %clock: i1, reset: %reset: i1) -> (in0_ready: i1, out0_valid: i1)
    %handshake_load0.addrIn0_ready, %handshake_load0.dataFromMem_ready, %handshake_load0.ctrl_ready, %handshake_load0.dataOut_valid, %handshake_load0.dataOut_data, %handshake_load0.addrOut0_valid, %handshake_load0.addrOut0_data = hw.instance "handshake_load0" @handshake_load_in_ui64_ui32_out_ui32_ui64(addrIn0_valid: %handshake_buffer45.out0_valid: i1, addrIn0_data: %handshake_buffer45.out0_data: i64, dataFromMem_valid: %handshake_buffer4.out0_valid: i1, dataFromMem_data: %handshake_buffer4.out0_data: i32, ctrl_valid: %handshake_buffer50.out0_valid: i1, dataOut_ready: %handshake_buffer55.in0_ready: i1, addrOut0_ready: %handshake_buffer54.in0_ready: i1) -> (addrIn0_ready: i1, dataFromMem_ready: i1, ctrl_ready: i1, dataOut_valid: i1, dataOut_data: i32, addrOut0_valid: i1, addrOut0_data: i64)
    %handshake_buffer54.in0_ready, %handshake_buffer54.out0_valid, %handshake_buffer54.out0_data = hw.instance "handshake_buffer54" @handshake_buffer_in_ui64_out_ui64_2slots_seq(in0_valid: %handshake_load0.addrOut0_valid: i1, in0_data: %handshake_load0.addrOut0_data: i64, out0_ready: %handshake_extmemory1.ldAddr0_ready: i1, clock: %clock: i1, reset: %reset: i1) -> (in0_ready: i1, out0_valid: i1, out0_data: i64)
    %handshake_buffer55.in0_ready, %handshake_buffer55.out0_valid, %handshake_buffer55.out0_data = hw.instance "handshake_buffer55" @handshake_buffer_in_ui32_out_ui32_2slots_seq(in0_valid: %handshake_load0.dataOut_valid: i1, in0_data: %handshake_load0.dataOut_data: i32, out0_ready: %arith_muli0.in0_ready: i1, clock: %clock: i1, reset: %reset: i1) -> (in0_ready: i1, out0_valid: i1, out0_data: i32)
    %handshake_load1.addrIn0_ready, %handshake_load1.dataFromMem_ready, %handshake_load1.ctrl_ready, %handshake_load1.dataOut_valid, %handshake_load1.dataOut_data, %handshake_load1.addrOut0_valid, %handshake_load1.addrOut0_data = hw.instance "handshake_load1" @handshake_load_in_ui64_ui32_out_ui32_ui64(addrIn0_valid: %handshake_buffer46.out0_valid: i1, addrIn0_data: %handshake_buffer46.out0_data: i64, dataFromMem_valid: %handshake_buffer2.out0_valid: i1, dataFromMem_data: %handshake_buffer2.out0_data: i32, ctrl_valid: %handshake_buffer52.out0_valid: i1, dataOut_ready: %handshake_buffer57.in0_ready: i1, addrOut0_ready: %handshake_buffer56.in0_ready: i1) -> (addrIn0_ready: i1, dataFromMem_ready: i1, ctrl_ready: i1, dataOut_valid: i1, dataOut_data: i32, addrOut0_valid: i1, addrOut0_data: i64)
    %handshake_buffer56.in0_ready, %handshake_buffer56.out0_valid, %handshake_buffer56.out0_data = hw.instance "handshake_buffer56" @handshake_buffer_in_ui64_out_ui64_2slots_seq(in0_valid: %handshake_load1.addrOut0_valid: i1, in0_data: %handshake_load1.addrOut0_data: i64, out0_ready: %handshake_extmemory0.ldAddr0_ready: i1, clock: %clock: i1, reset: %reset: i1) -> (in0_ready: i1, out0_valid: i1, out0_data: i64)
    %handshake_buffer57.in0_ready, %handshake_buffer57.out0_valid, %handshake_buffer57.out0_data = hw.instance "handshake_buffer57" @handshake_buffer_in_ui32_out_ui32_2slots_seq(in0_valid: %handshake_load1.dataOut_valid: i1, in0_data: %handshake_load1.dataOut_data: i32, out0_ready: %arith_muli0.in1_ready: i1, clock: %clock: i1, reset: %reset: i1) -> (in0_ready: i1, out0_valid: i1, out0_data: i32)
    %arith_muli0.in0_ready, %arith_muli0.in1_ready, %arith_muli0.out0_valid, %arith_muli0.out0_data = hw.instance "arith_muli0" @arith_muli_in_ui32_ui32_out_ui32(in0_valid: %handshake_buffer55.out0_valid: i1, in0_data: %handshake_buffer55.out0_data: i32, in1_valid: %handshake_buffer57.out0_valid: i1, in1_data: %handshake_buffer57.out0_data: i32, out0_ready: %handshake_buffer58.in0_ready: i1) -> (in0_ready: i1, in1_ready: i1, out0_valid: i1, out0_data: i32)
    %handshake_buffer58.in0_ready, %handshake_buffer58.out0_valid, %handshake_buffer58.out0_data = hw.instance "handshake_buffer58" @handshake_buffer_in_ui32_out_ui32_2slots_seq(in0_valid: %arith_muli0.out0_valid: i1, in0_data: %arith_muli0.out0_data: i32, out0_ready: %arith_addi0.in1_ready: i1, clock: %clock: i1, reset: %reset: i1) -> (in0_ready: i1, out0_valid: i1, out0_data: i32)
    %arith_addi0.in0_ready, %arith_addi0.in1_ready, %arith_addi0.out0_valid, %arith_addi0.out0_data = hw.instance "arith_addi0" @arith_addi_in_ui32_ui32_out_ui32(in0_valid: %handshake_buffer44.out0_valid: i1, in0_data: %handshake_buffer44.out0_data: i32, in1_valid: %handshake_buffer58.out0_valid: i1, in1_data: %handshake_buffer58.out0_data: i32, out0_ready: %handshake_buffer59.in0_ready: i1) -> (in0_ready: i1, in1_ready: i1, out0_valid: i1, out0_data: i32)
    %handshake_buffer59.in0_ready, %handshake_buffer59.out0_valid, %handshake_buffer59.out0_data = hw.instance "handshake_buffer59" @handshake_buffer_in_ui32_out_ui32_2slots_seq(in0_valid: %arith_addi0.out0_valid: i1, in0_data: %arith_addi0.out0_data: i32, out0_ready: %handshake_mux4.in1_ready: i1, clock: %clock: i1, reset: %reset: i1) -> (in0_ready: i1, out0_valid: i1, out0_data: i32)
    %arith_addi1.in0_ready, %arith_addi1.in1_ready, %arith_addi1.out0_valid, %arith_addi1.out0_data = hw.instance "arith_addi1" @arith_addi_in_ui64_ui64_out_ui64(in0_valid: %handshake_buffer47.out0_valid: i1, in0_data: %handshake_buffer47.out0_data: i64, in1_valid: %handshake_buffer48.out0_valid: i1, in1_data: %handshake_buffer48.out0_data: i64, out0_ready: %handshake_buffer60.in0_ready: i1) -> (in0_ready: i1, in1_ready: i1, out0_valid: i1, out0_data: i64)
    %handshake_buffer60.in0_ready, %handshake_buffer60.out0_valid, %handshake_buffer60.out0_data = hw.instance "handshake_buffer60" @handshake_buffer_in_ui64_out_ui64_2slots_seq(in0_valid: %arith_addi1.out0_valid: i1, in0_data: %arith_addi1.out0_data: i64, out0_ready: %handshake_mux3.in1_ready: i1, clock: %clock: i1, reset: %reset: i1) -> (in0_ready: i1, out0_valid: i1, out0_data: i64)
    hw.output %handshake_extmemory1.extmem_ldAddr0_valid, %handshake_extmemory1.extmem_ldAddr0_data, %handshake_extmemory1.extmem_ldData0_ready, %handshake_extmemory1.extmem_ldDone0_ready, %handshake_extmemory0.extmem_ldAddr0_valid, %handshake_extmemory0.extmem_ldAddr0_data, %handshake_extmemory0.extmem_ldData0_ready, %handshake_extmemory0.extmem_ldDone0_ready, %handshake_buffer0.in0_ready, %handshake_buffer43.out0_valid, %handshake_buffer43.out0_data, %handshake_buffer39.out0_valid : i1, i64, i1, i1, i1, i64, i1, i1, i1, i1, i32, i1
  }
}
