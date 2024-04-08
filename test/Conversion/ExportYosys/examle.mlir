module {
  sv.verbatim "// Standard header to adapt well known macros to our needs." {symbols = []}
  sv.ifdef  "RANDOMIZE_REG_INIT" {
    sv.verbatim "`define RANDOMIZE" {symbols = []}
  }
  sv.verbatim "\0A// RANDOM may be set to an expression that produces a 32-bit random unsigned value." {symbols = []}
  sv.ifdef  "RANDOM" {
  } else {
    sv.verbatim "`define RANDOM $random" {symbols = []}
  }
  sv.verbatim "\0A// Users can define 'PRINTF_COND' to add an extra gate to prints." {symbols = []}
  sv.ifdef  "PRINTF_COND" {
    sv.verbatim "`define PRINTF_COND_ (`PRINTF_COND)" {symbols = []}
  } else {
    sv.verbatim "`define PRINTF_COND_ 1" {symbols = []}
  }
  sv.verbatim "\0A// Users can define 'STOP_COND' to add an extra gate to stop conditions." {symbols = []}
  sv.ifdef  "STOP_COND" {
    sv.verbatim "`define STOP_COND_ (`STOP_COND)" {symbols = []}
  } else {
    sv.verbatim "`define STOP_COND_ 1" {symbols = []}
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
  hw.module @RocketCore(%clock: i1, %reset: i1, %io_hartid: i64, %io_imem_resp_valid: i1, %io_imem_resp_bits_btb_valid: i1, %io_imem_resp_bits_btb_bits_taken: i1, %io_imem_resp_bits_btb_bits_bridx: i1, %io_imem_resp_bits_pc: i40, %io_imem_resp_bits_data: i32, %io_imem_resp_bits_xcpt_if: i1, %io_imem_resp_bits_replay: i1, %io_dmem_req_ready: i1, %io_dmem_resp_valid: i1, %io_dmem_resp_bits_tag: i7, %io_dmem_resp_bits_data: i64, %io_dmem_resp_bits_replay: i1, %io_dmem_resp_bits_has_data: i1, %io_dmem_ordered: i1, %io_fpu_fcsr_rdy: i1, %io_fpu_dec_wen: i1, %io_fpu_dec_ren1: i1, %io_fpu_dec_ren2: i1, %io_fpu_dec_ren3: i1) -> (io_dmem_req_valid: i1, io_fpu_valid: i1) {
    %_T_4428 = sv.wire  : !hw.inout<i1>
    %c0_i2 = hw.constant 0 : i2
    %c0_i3 = hw.constant 0 : i3
    %c5_i4 = hw.constant 5 : i4
    %c9_i5 = hw.constant 9 : i5
    %c7_i5 = hw.constant 7 : i5
    %c-1_i4 = hw.constant -1 : i4
    %c-3_i3 = hw.constant -3 : i3
    %c-1_i3 = hw.constant -1 : i3
    %c-1_i2 = hw.constant -1 : i2
    %c3_i3 = hw.constant 3 : i3
    %c1_i2 = hw.constant 1 : i2
    %c-10_i5 = hw.constant -10 : i5
    %c143_i10 = hw.constant 143 : i10
    %c3_i8 = hw.constant 3 : i8
    %c3_i7 = hw.constant 3 : i7
    %c-6_i5 = hw.constant -6 : i5
    %c-4_i3 = hw.constant -4 : i3
    %c2_i3 = hw.constant 2 : i3
    %c18_i6 = hw.constant 18 : i6
    %c-2_i2 = hw.constant -2 : i2
    %c1_i4 = hw.constant 1 : i4
    %c32943_i19 = hw.constant 32943 : i19
    %c687_i13 = hw.constant 687 : i13
    %c175_i11 = hw.constant 175 : i11
    %c35_i7 = hw.constant 35 : i7
    %c0_i4 = hw.constant 0 : i4
    %c1_i5 = hw.constant 1 : i5
    %true = hw.constant true
    %c19_i8 = hw.constant 19 : i8
    %c35_i10 = hw.constant 35 : i10
    %c18_i8 = hw.constant 18 : i8
    %c35_i9 = hw.constant 35 : i9
    %c39_i10 = hw.constant 39 : i10
    %c115_i15 = hw.constant 115 : i15
    %c231_i15 = hw.constant 231 : i15
    %c31_i15 = hw.constant 31 : i15
    %c0_i8 = hw.constant 0 : i8
    %c103_i15 = hw.constant 103 : i15
    %c19_i11 = hw.constant 19 : i11
    %c18_i10 = hw.constant 18 : i10
    %c1_i7 = hw.constant 1 : i7
    %c111_i12 = hw.constant 111 : i12
    %false = hw.constant false
    %c1_i3 = hw.constant 1 : i3
    %c-3_i5 = hw.constant -3 : i5
    %c-11_i5 = hw.constant -11 : i5
    %c3_i4 = hw.constant 3 : i4
    %c47_i9 = hw.constant 47 : i9
    %c13_i5 = hw.constant 13 : i5
    %c65_i12 = hw.constant 65 : i12
    %c0_i27 = hw.constant 0 : i27
    %c0_i6 = hw.constant 0 : i6
    %c0_i185 = hw.constant 0 : i185
    %c0_i63 = hw.constant 0 : i63
    %c0_i7 = hw.constant 0 : i7
    %c2_i5 = hw.constant 2 : i5
    %c-29_i7 = hw.constant -29 : i7
    %c51_i7 = hw.constant 51 : i7
    %c0_i12 = hw.constant 0 : i12
    %c7_i7 = hw.constant 7 : i7
    %c31_i7 = hw.constant 31 : i7
    %c19_i7 = hw.constant 19 : i7
    %c0_i5 = hw.constant 0 : i5
    %c0_i32 = hw.constant 0 : i32
    %0 = comb.extract %io_imem_resp_bits_pc from 1 {sv.namehint = "pcWordBits"} : (i40) -> i1
    %1 = comb.and bin %io_imem_resp_bits_btb_valid, %io_imem_resp_bits_btb_bits_taken {sv.namehint = "_T_375"} : i1
    %2 = comb.concat %false, %io_imem_resp_bits_btb_bits_bridx : i1, i1
    %3 = sv.wire  {hw.verilogName = "_GEN"} : !hw.inout<i2>
    sv.assign %3, %2 : i2
    %4 = sv.read_inout %3 : !hw.inout<i2>
    %5 = comb.add bin %4, %c1_i2 {sv.namehint = "_T_377"} : i2
    %6 = comb.mux bin %1, %5, %c-2_i2 {sv.namehint = "_T_379"} : i2
    %7 = comb.concat %false, %0 : i1, i1
    %8 = sv.wire  {hw.verilogName = "_GEN_0"} : !hw.inout<i2>
    sv.assign %8, %7 : i2
    %9 = sv.read_inout %8 : !hw.inout<i2>
    %10 = comb.sub %6, %9 {sv.namehint = "nIC"} : i2
    %11 = comb.mux bin %io_imem_resp_valid, %10, %c0_i2 {sv.namehint = "_T_385"} : i2
    %12 = sv.read_inout %8 : !hw.inout<i2>
    %13 = comb.sub %c-2_i2, %12 {sv.namehint = "icShiftAmt"} : i2
    %14 = comb.extract %io_imem_resp_bits_data from 0 {sv.namehint = "_T_439"} : (i32) -> i16
    %15 = comb.replicate %14 {sv.namehint = "_T_440"} : (i16) -> i32
    %16 = comb.extract %io_imem_resp_bits_data from 16 {sv.namehint = "_T_442"} : (i32) -> i16
    %17 = comb.replicate %16 {sv.namehint = "_T_443"} : (i16) -> i32
    %18 = comb.replicate %17 {sv.namehint = "_T_444"} : (i32) -> i64
    %19 = comb.concat %c0_i63, %18, %io_imem_resp_bits_data, %15 : i63, i64, i32, i32
    %20 = comb.concat %c0_i185, %13, %c0_i4 : i185, i2, i4
    %21 = comb.shl bin %19, %20 : i191
    %_T_447 = sv.wire  : !hw.inout<i191>
    sv.assign %_T_447, %21 : i191
    %22 = sv.read_inout %_T_447 : !hw.inout<i191>
    %23 = comb.extract %22 from 64 {sv.namehint = "icData"} : (i191) -> i32
    %24 = comb.concat %c0_i2, %11 : i2, i2
    %25 = comb.shl bin %c1_i4, %24 : i4
    %_T_456 = sv.wire  : !hw.inout<i4>
    sv.assign %_T_456, %25 : i4
    %26 = sv.read_inout %_T_456 : !hw.inout<i4>
    %27 = comb.extract %26 from 0 : (i4) -> i2
    %c1_i2_0 = hw.constant 1 : i2
    %28 = comb.sub %27, %c1_i2_0 : i2
    %valid = sv.wire  : !hw.inout<i2>
    sv.assign %valid, %28 : i2
    %29 = sv.read_inout %valid : !hw.inout<i2>
    %30 = comb.extract %29 from 1 : (i2) -> i1
    %31 = comb.and %30, %io_imem_resp_bits_xcpt_if {sv.namehint = "_T_540"} : i1
    %32 = comb.replicate %io_imem_resp_bits_replay {sv.namehint = "_T_476"} : (i1) -> i2
    %33 = sv.read_inout %valid : !hw.inout<i2>
    %34 = comb.and bin %33, %32 : i2
    %ic_replay = sv.wire  : !hw.inout<i2>
    sv.assign %ic_replay, %34 : i2
    %35 = sv.read_inout %3 : !hw.inout<i2>
    %36 = sv.read_inout %8 : !hw.inout<i2>
    %37 = comb.sub %35, %36 {sv.namehint = "_T_491"} : i2
    %38 = comb.icmp eq %37, %c0_i2 : i2
    %39 = comb.and %io_imem_resp_bits_btb_valid, %38 {sv.namehint = "_T_519"} : i1
    %40 = sv.read_inout %_T_447 : !hw.inout<i191>
    %41 = comb.extract %40 from 64 {sv.namehint = "_T_1552"} : (i191) -> i2
    %42 = comb.icmp bin eq %41, %c-1_i2 {sv.namehint = "_T_557"} : i2
    %43 = comb.xor bin %42, %true {sv.namehint = "_T_14"} : i1
    %44 = sv.read_inout %_T_447 : !hw.inout<i191>
    %45 = comb.extract %44 from 69 {sv.namehint = "_T_15"} : (i191) -> i8
    %46 = comb.icmp bin ne %45, %c0_i8 {sv.namehint = "_T_17"} : i8
    %47 = comb.mux bin %46, %c19_i7, %c31_i7 {sv.namehint = "_T_20"} : i7
    %48 = sv.read_inout %_T_447 : !hw.inout<i191>
    %49 = comb.extract %48 from 71 {sv.namehint = "_T_21"} : (i191) -> i4
    %50 = sv.read_inout %_T_447 : !hw.inout<i191>
    %51 = comb.extract %50 from 75 {sv.namehint = "_T_22"} : (i191) -> i2
    %52 = sv.read_inout %_T_447 : !hw.inout<i191>
    %53 = comb.extract %52 from 69 {sv.namehint = "_T_546"} : (i191) -> i1
    %54 = sv.read_inout %_T_447 : !hw.inout<i191>
    %55 = comb.extract %54 from 70 {sv.namehint = "_T_818"} : (i191) -> i1
    %56 = sv.read_inout %_T_447 : !hw.inout<i191>
    %57 = comb.extract %56 from 66 {sv.namehint = "_T_1141"} : (i191) -> i3
    %58 = sv.read_inout %_T_447 : !hw.inout<i191>
    %59 = comb.extract %58 from 91 {sv.namehint = "_T_1539"} : (i191) -> i5
    %60 = comb.concat %c0_i2, %49, %51, %53, %55, %c65_i12, %57, %47 : i2, i4, i2, i1, i1, i12, i3, i7
    %61 = sv.read_inout %_T_447 : !hw.inout<i191>
    %62 = comb.extract %61 from 69 {sv.namehint = "_T_1143"} : (i191) -> i2
    %63 = sv.read_inout %_T_447 : !hw.inout<i191>
    %64 = comb.extract %63 from 74 {sv.namehint = "_T_1397"} : (i191) -> i3
    %65 = sv.read_inout %_T_447 : !hw.inout<i191>
    %66 = comb.extract %65 from 71 {sv.namehint = "_T_1396"} : (i191) -> i3
    %67 = comb.concat %c1_i2, %66 {sv.namehint = "_T_81"} : i2, i3
    %68 = comb.concat %c0_i4, %62, %64, %c1_i5, %66, %c13_i5, %57, %c7_i7 : i4, i2, i3, i5, i3, i5, i3, i7
    %69 = comb.concat %c1_i2, %66 {sv.namehint = "_T_161"} : i2, i3
    %70 = sv.read_inout %_T_447 : !hw.inout<i191>
    %71 = comb.extract %70 from 76 : (i191) -> i1
    %72 = sv.read_inout %_T_447 : !hw.inout<i191>
    %73 = comb.extract %72 from 74 : (i191) -> i2
    %74 = comb.concat %c0_i5, %53, %71, %c1_i2, %57, %c1_i2, %66, %c2_i3, %73, %55, %c47_i9 : i5, i1, i1, i2, i3, i2, i3, i3, i2, i1, i9
    %75 = comb.concat %c0_i4, %62, %71, %c1_i2, %57, %c1_i2, %66, %c3_i3, %73, %c39_i10 : i4, i2, i1, i2, i3, i2, i3, i3, i2, i10
    %76 = comb.concat %c0_i5, %53, %71, %c1_i2, %57, %c1_i2, %66, %c2_i3, %73, %55, %c35_i9 : i5, i1, i1, i2, i3, i2, i3, i3, i2, i1, i9
    %77 = comb.concat %c1_i2, %57 {sv.namehint = "_T_354"} : i2, i3
    %78 = comb.concat %c1_i2, %66 {sv.namehint = "_T_357"} : i2, i3
    %79 = comb.concat %c1_i2, %57 {sv.namehint = "_T_360"} : i2, i3
    %80 = comb.concat %c0_i4, %62, %71, %c1_i2, %57, %c1_i2, %66, %c3_i3, %73, %c35_i10 : i4, i2, i1, i2, i3, i2, i3, i3, i2, i10
    %81 = sv.read_inout %_T_447 : !hw.inout<i191>
    %82 = comb.extract %81 from 76 {sv.namehint = "_T_1306"} : (i191) -> i1
    %83 = sv.read_inout %_T_447 : !hw.inout<i191>
    %84 = comb.extract %83 from 66 {sv.namehint = "_T_1410"} : (i191) -> i5
    %85 = sv.read_inout %_T_447 : !hw.inout<i191>
    %86 = comb.extract %85 from 71 {sv.namehint = "_T_1536"} : (i191) -> i5
    %87 = comb.icmp eq %86, %c0_i5 : i5
    %88 = comb.replicate %82 {sv.namehint = "_T_482"} : (i1) -> i7
    %89 = comb.concat %88, %84 {sv.namehint = "_T_484"} : i7, i5
    %90 = comb.icmp eq %89, %c0_i12 : i12
    %91 = comb.icmp eq %86, %c0_i5 {sv.namehint = "_T_523"} : i5
    %92 = comb.icmp bin eq %86, %c2_i5 {sv.namehint = "_T_526"} : i5
    %93 = comb.or bin %91, %92 {sv.namehint = "_T_527"} : i1
    %94 = comb.replicate %82 {sv.namehint = "_T_532"} : (i1) -> i7
    %95 = comb.concat %94, %84 {sv.namehint = "_T_534"} : i7, i5
    %96 = comb.icmp bin ne %95, %c0_i12 {sv.namehint = "_T_536"} : i12
    %97 = comb.mux bin %96, %c19_i7, %c31_i7 {sv.namehint = "_T_539"} : i7
    %98 = sv.read_inout %_T_447 : !hw.inout<i191>
    %99 = comb.extract %98 from 67 {sv.namehint = "_T_1018"} : (i191) -> i2
    %100 = sv.read_inout %_T_447 : !hw.inout<i191>
    %101 = comb.extract %100 from 66 {sv.namehint = "_T_1016"} : (i191) -> i1
    %102 = comb.concat %c1_i2, %57 {sv.namehint = "_T_566"} : i2, i3
    %103 = comb.concat %99, %53, %101, %55, %c0_i4, %86, %c0_i3, %86, %97 : i2, i1, i1, i1, i4, i5, i3, i5, i7
    %104 = comb.replicate %82 : (i1) -> i12
    %105 = comb.concat %104, %84, %86, %c3_i3, %90, %c-1_i3 : i12, i5, i5, i3, i1, i3
    %106 = comb.mux bin %93, %103, %105 : i29
    %107 = comb.replicate %82 {sv.namehint = "_T_622"} : (i1) -> i7
    %108 = comb.concat %107, %84, %c1_i2, %66, %c-3_i5, %66, %c19_i7 {sv.namehint = "_T_636"} : i7, i5, i2, i3, i5, i3, i7
    %109 = sv.read_inout %_T_447 : !hw.inout<i191>
    %110 = comb.extract %109 from 70 {sv.namehint = "_T_670"} : (i191) -> i1
    %111 = sv.read_inout %_T_447 : !hw.inout<i191>
    %112 = comb.extract %111 from 69 : (i191) -> i1
    %113 = comb.concat %true, %112 {sv.namehint = "_T_660"} : i1, i1
    %114 = comb.mux bin %110, %113, %c0_i2 {sv.namehint = "_T_666"} : i2
    %115 = comb.concat %c-1_i2, %112 {sv.namehint = "_T_675"} : i2, i1
    %116 = comb.concat %112, %c0_i2 {sv.namehint = "_T_680"} : i1, i2
    %117 = comb.mux bin %110, %115, %116 {sv.namehint = "_T_681"} : i3
    %118 = comb.concat %false, %114 : i1, i2
    %119 = comb.mux bin %82, %118, %117 {sv.namehint = "_T_682"} : i3
    %120 = comb.icmp eq %62, %c0_i2 {sv.namehint = "_T_685"} : i2
    %121 = sv.read_inout %_T_447 : !hw.inout<i191>
    %122 = comb.extract %121 from 74 {sv.namehint = "_T_1017"} : (i191) -> i2
    %123 = sv.read_inout %_T_447 : !hw.inout<i191>
    %124 = comb.extract %123 from 74 {sv.namehint = "_T_720"} : (i191) -> i1
    %125 = sv.read_inout %_T_447 : !hw.inout<i191>
    %126 = comb.extract %125 from 75 {sv.namehint = "_T_711"} : (i191) -> i1
    %127 = comb.concat %false, %120, %c1_i7, %57, %c1_i2, %66, %119, %c1_i2, %66, %c3_i3, %82, %c3_i3 : i1, i1, i7, i3, i2, i3, i3, i2, i3, i3, i1, i3
    %128 = comb.mux bin %124, %127, %108 {sv.namehint = "_T_716"} : i32
    %129 = comb.concat %false, %124, %c0_i4, %82, %84, %c1_i2, %66, %c-11_i5, %66, %c19_i7 : i1, i1, i4, i1, i5, i2, i3, i5, i3, i7
    %130 = comb.mux bin %126, %128, %129 {sv.namehint = "_T_722"} : i32
    %131 = comb.concat %c1_i2, %66 {sv.namehint = "_T_725"} : i2, i3
    %132 = sv.read_inout %_T_447 : !hw.inout<i191>
    %133 = comb.extract %132 from 72 {sv.namehint = "_T_816"} : (i191) -> i1
    %134 = sv.read_inout %_T_447 : !hw.inout<i191>
    %135 = comb.extract %134 from 73 {sv.namehint = "_T_817"} : (i191) -> i2
    %136 = sv.read_inout %_T_447 : !hw.inout<i191>
    %137 = comb.extract %136 from 71 {sv.namehint = "_T_819"} : (i191) -> i1
    %138 = sv.read_inout %_T_447 : !hw.inout<i191>
    %139 = comb.extract %138 from 75 {sv.namehint = "_T_821"} : (i191) -> i1
    %140 = sv.read_inout %_T_447 : !hw.inout<i191>
    %141 = comb.extract %140 from 67 {sv.namehint = "_T_822"} : (i191) -> i3
    %142 = comb.replicate %82 : (i1) -> i9
    %143 = comb.concat %82, %133, %135, %55, %137, %101, %139, %141, %142, %c111_i12 {sv.namehint = "_T_839"} : i1, i1, i2, i1, i1, i1, i1, i3, i9, i12
    %144 = comb.concat %c1_i2, %57 {sv.namehint = "_T_846"} : i2, i3
    %145 = comb.concat %c1_i2, %66 {sv.namehint = "_T_939"} : i2, i3
    %146 = comb.concat %c1_i2, %66 {sv.namehint = "_T_1037"} : i2, i3
    %147 = comb.concat %c0_i6, %82, %84, %86, %c1_i3, %86, %c19_i7 : i6, i1, i5, i5, i3, i5, i7
    %148 = comb.concat %c0_i3, %57, %82, %62, %c19_i11, %86, %c7_i7 : i3, i3, i1, i2, i11, i5, i7
    %149 = sv.read_inout %_T_447 : !hw.inout<i191>
    %150 = comb.extract %149 from 66 {sv.namehint = "_T_1110"} : (i191) -> i2
    %151 = sv.read_inout %_T_447 : !hw.inout<i191>
    %152 = comb.extract %151 from 68 {sv.namehint = "_T_1112"} : (i191) -> i3
    %153 = comb.icmp bin ne %86, %c0_i5 {sv.namehint = "_T_1236"} : i5
    %154 = comb.mux bin %153, %c103_i15, %c31_i15 : i15
    %155 = comb.concat %c0_i8, %86, %c51_i7 : i8, i5, i7
    %156 = comb.concat %86, %154 : i5, i15
    %157 = comb.icmp bin ne %84, %c0_i5 {sv.namehint = "_T_1256"} : i5
    %158 = comb.mux bin %157, %155, %156 : i20
    %159 = comb.icmp bin ne %84, %c0_i5 {sv.namehint = "_T_1256"} : i5
    %160 = comb.mux bin %159, %86, %c0_i5 {sv.namehint = "_T_1257_rd"} : i5
    %161 = comb.concat %84, %86, %c231_i15 {sv.namehint = "_T_1271"} : i5, i5, i15
    %162 = comb.or %84, %c1_i5 : i5
    %163 = comb.concat %162, %86, %c115_i15 {sv.namehint = "_T_1276"} : i5, i5, i15
    %164 = comb.icmp bin ne %86, %c0_i5 {sv.namehint = "_T_1279"} : i5
    %165 = comb.mux bin %164, %161, %163 {sv.namehint = "_T_1280"} : i25
    %166 = comb.concat %84, %86, %c0_i3, %86, %c51_i7 : i5, i5, i3, i5, i7
    %167 = comb.icmp bin ne %84, %c0_i5 {sv.namehint = "_T_1299"} : i5
    %168 = comb.mux bin %167, %166, %165 : i25
    %169 = comb.icmp bin ne %84, %c0_i5 {sv.namehint = "_T_1299"} : i5
    %170 = comb.mux bin %169, %86, %c1_i5 {sv.namehint = "_T_1300_rd"} : i5
    %171 = comb.concat %84, %158 : i5, i20
    %172 = comb.mux bin %82, %168, %171 : i25
    %173 = comb.concat %c0_i7, %172 {sv.namehint = "_T_1307_bits"} : i7, i25
    %174 = comb.mux bin %82, %170, %160 {sv.namehint = "_T_1307_rd"} : i5
    %175 = comb.icmp bin ne %84, %c0_i5 {sv.namehint = "_T_1256"} : i5
    %176 = comb.xor %175, %true : i1
    %177 = comb.or %82, %176 : i1
    %178 = comb.mux bin %177, %86, %c0_i5 {sv.namehint = "_T_1307_rs1"} : i5
    %179 = comb.concat %c0_i3, %66, %71, %84, %c19_i8, %73, %c39_i10 : i3, i3, i1, i5, i8, i2, i10
    %180 = sv.read_inout %_T_447 : !hw.inout<i191>
    %181 = comb.extract %180 from 71 {sv.namehint = "_T_1359"} : (i191) -> i2
    %182 = sv.read_inout %_T_447 : !hw.inout<i191>
    %183 = comb.extract %182 from 73 : (i191) -> i3
    %184 = comb.concat %c0_i4, %181, %71, %84, %c18_i8, %183, %c35_i9 : i4, i2, i1, i5, i8, i3, i9
    %185 = comb.concat %c0_i3, %66, %71, %84, %c19_i8, %73, %c35_i10 : i3, i3, i1, i5, i8, i2, i10
    %186 = sv.read_inout %_T_447 : !hw.inout<i191>
    %187 = comb.extract %186 from 79 {sv.namehint = "_T_1537"} : (i191) -> i5
    %188 = sv.read_inout %_T_447 : !hw.inout<i191>
    %189 = comb.extract %188 from 84 {sv.namehint = "_T_1538"} : (i191) -> i5
    %190 = sv.read_inout %_T_447 : !hw.inout<i191>
    %191 = comb.extract %190 from 65 {sv.namehint = "_T_1558"} : (i191) -> i1
    %192 = sv.read_inout %_T_447 : !hw.inout<i191>
    %193 = comb.extract %192 from 64 {sv.namehint = "_T_1712"} : (i191) -> i1
    %194 = sv.read_inout %_T_447 : !hw.inout<i191>
    %195 = comb.extract %194 from 79 {sv.namehint = "_T_1636"} : (i191) -> i1
    %196 = sv.read_inout %_T_447 : !hw.inout<i191>
    %197 = comb.extract %196 from 78 {sv.namehint = "_T_1670"} : (i191) -> i1
    %198 = sv.read_inout %_T_447 : !hw.inout<i191>
    %199 = comb.extract %198 from 77 : (i191) -> i1
    %200 = comb.mux bin %199, %185, %184 {sv.namehint = "_T_1645_bits"} : i32
    %201 = comb.mux bin %199, %179, %173 {sv.namehint = "_T_1655_bits"} : i32
    %202 = comb.mux bin %197, %200, %201 {sv.namehint = "_T_1661_bits"} : i32
    %203 = comb.or %197, %199 : i1
    %204 = sv.wire  {hw.verilogName = "_GEN_1"} : !hw.inout<i1>
    sv.assign %204, %203 : i1
    %205 = sv.read_inout %204 : !hw.inout<i1>
    %206 = comb.mux bin %205, %c2_i5, %178 {sv.namehint = "_T_1661_rs1"} : i5
    %207 = comb.concat %c0_i3, %57, %82, %62, %c19_i11 : i3, i3, i1, i2, i11
    %208 = comb.concat %c0_i4, %150, %82, %152, %c18_i10 : i4, i2, i1, i3, i10
    %209 = comb.mux bin %199, %207, %208 : i20
    %210 = comb.concat %209, %86, %c3_i7 {sv.namehint = "_T_1675_bits"} : i20, i5, i7
    %211 = comb.mux bin %199, %148, %147 {sv.namehint = "_T_1685_bits"} : i32
    %212 = comb.mux bin %197, %210, %211 {sv.namehint = "_T_1691_bits"} : i32
    %213 = sv.read_inout %204 : !hw.inout<i1>
    %214 = comb.mux bin %213, %c2_i5, %86 {sv.namehint = "_T_1691_rs1"} : i5
    %215 = comb.mux bin %195, %202, %212 {sv.namehint = "_T_1697_bits"} : i32
    %216 = comb.xor %195, %true : i1
    %217 = comb.mux bin %195, %206, %214 {sv.namehint = "_T_1697_rs1"} : i5
    %218 = comb.mux bin %193, %23, %215 {sv.namehint = "_T_1703_bits"} : i32
    %219 = comb.or %193, %216 : i1
    %220 = comb.or %197, %199 : i1
    %221 = comb.or %219, %220 : i1
    %222 = comb.mux bin %221, %86, %174 {sv.namehint = "_T_1703_rd"} : i5
    %223 = comb.mux bin %193, %187, %217 {sv.namehint = "_T_1703_rs1"} : i5
    %224 = comb.mux bin %193, %189, %84 {sv.namehint = "_T_1703_rs2"} : i5
    %225 = sv.read_inout %_T_447 : !hw.inout<i191>
    %226 = comb.extract %225 from 79 {sv.namehint = "_T_1786"} : (i191) -> i1
    %227 = sv.read_inout %_T_447 : !hw.inout<i191>
    %228 = comb.extract %227 from 78 {sv.namehint = "_T_1750"} : (i191) -> i1
    %229 = comb.replicate %82 : (i1) -> i4
    %230 = sv.wire  {hw.verilogName = "_GEN_2"} : !hw.inout<i4>
    sv.assign %230, %229 : i4
    %231 = sv.read_inout %230 : !hw.inout<i4>
    %232 = comb.concat %231, %62, %101, %c1_i7, %66, %c0_i2, %199, %122, %99, %82, %c-29_i7 {sv.namehint = "_T_1725_bits"} : i4, i2, i1, i7, i3, i2, i1, i2, i2, i1, i7
    %233 = comb.mux bin %199, %c0_i5, %145 {sv.namehint = "_T_1725_rd"} : i5
    %234 = comb.mux bin %199, %143, %130 {sv.namehint = "_T_1735_bits"} : i32
    %235 = comb.mux bin %199, %c0_i5, %131 {sv.namehint = "_T_1735_rd"} : i5
    %236 = comb.mux bin %228, %232, %234 {sv.namehint = "_T_1741_bits"} : i32
    %237 = comb.mux bin %228, %233, %235 {sv.namehint = "_T_1741_rd"} : i5
    %238 = comb.mux bin %228, %c0_i5, %144 {sv.namehint = "_T_1741_rs2"} : i5
    %239 = sv.read_inout %230 : !hw.inout<i4>
    %240 = comb.concat %239, %84, %c0_i8, %86, %c19_i7 : i4, i5, i8, i5, i7
    %241 = comb.mux bin %199, %106, %240 : i29
    %242 = comb.concat %c3_i4, %87, %c-1_i2 : i4, i1, i2
    %243 = comb.mux bin %199, %242, %c19_i7 : i7
    %244 = comb.replicate %82 : (i1) -> i3
    %245 = sv.read_inout %230 : !hw.inout<i4>
    %246 = comb.concat %245, %84, %86, %c0_i3, %86, %243 : i4, i5, i5, i3, i5, i7
    %247 = comb.mux bin %228, %241, %246 : i29
    %248 = comb.concat %244, %247 {sv.namehint = "_T_1771_bits"} : i3, i29
    %249 = comb.xor %228, %true : i1
    %250 = comb.or %249, %199 : i1
    %251 = comb.mux bin %250, %86, %c0_i5 {sv.namehint = "_T_1771_rs1"} : i5
    %252 = comb.mux bin %226, %236, %248 {sv.namehint = "_T_1777_bits"} : i32
    %253 = comb.mux bin %226, %237, %86 {sv.namehint = "_T_1777_rd"} : i5
    %254 = comb.mux bin %226, %146, %251 {sv.namehint = "_T_1777_rs1"} : i5
    %255 = comb.mux bin %226, %238, %102 {sv.namehint = "_T_1777_rs2"} : i5
    %256 = sv.read_inout %_T_447 : !hw.inout<i191>
    %257 = comb.extract %256 from 78 {sv.namehint = "_T_1820"} : (i191) -> i1
    %258 = comb.mux bin %199, %80, %76 {sv.namehint = "_T_1795_bits"} : i32
    %259 = comb.mux bin %199, %75, %74 {sv.namehint = "_T_1805_bits"} : i32
    %260 = comb.mux bin %257, %258, %259 {sv.namehint = "_T_1811_bits"} : i32
    %261 = comb.concat %c0_i4, %62, %64, %c1_i5, %66, %c13_i5 : i4, i2, i3, i5, i3, i5
    %262 = comb.concat %c0_i5, %53, %64, %55, %c1_i4, %66, %c9_i5 : i5, i1, i3, i1, i4, i3, i5
    %263 = comb.mux bin %199, %261, %262 : i22
    %264 = comb.concat %263, %57, %c3_i7 {sv.namehint = "_T_1825_bits"} : i22, i3, i7
    %265 = comb.mux bin %199, %68, %60 {sv.namehint = "_T_1835_bits"} : i32
    %266 = comb.mux bin %199, %67, %c2_i5 {sv.namehint = "_T_1835_rs1"} : i5
    %267 = comb.mux bin %257, %264, %265 {sv.namehint = "_T_1841_bits"} : i32
    %268 = comb.mux bin %257, %69, %266 {sv.namehint = "_T_1841_rs1"} : i5
    %269 = comb.mux bin %226, %260, %267 {sv.namehint = "_T_1847_bits"} : i32
    %270 = comb.mux bin %226, %78, %268 {sv.namehint = "_T_1847_rs1"} : i5
    %271 = comb.mux bin %193, %252, %269 {sv.namehint = "_T_1853_bits"} : i32
    %272 = comb.mux bin %193, %253, %77 {sv.namehint = "_T_1853_rd"} : i5
    %273 = comb.mux bin %193, %254, %270 {sv.namehint = "_T_1853_rs1"} : i5
    %274 = comb.mux bin %193, %255, %79 {sv.namehint = "_T_1853_rs2"} : i5
    %275 = comb.mux bin %191, %218, %271 : i32
    %_T_1859_bits = sv.wire  : !hw.inout<i32>
    sv.assign %_T_1859_bits, %275 : i32
    %276 = comb.mux bin %191, %222, %272 : i5
    %_T_1859_rd = sv.wire  : !hw.inout<i5>
    sv.assign %_T_1859_rd, %276 : i5
    %277 = comb.mux bin %191, %223, %273 : i5
    %_T_1859_rs1 = sv.wire  : !hw.inout<i5>
    sv.assign %_T_1859_rs1, %277 : i5
    %278 = comb.mux bin %191, %224, %274 : i5
    %_T_1859_rs2 = sv.wire  : !hw.inout<i5>
    sv.assign %_T_1859_rs2, %278 : i5
    %279 = sv.read_inout %ic_replay : !hw.inout<i2>
    %280 = comb.extract %279 from 0 {sv.namehint = "_T_515"} : (i2) -> i1
    %281 = sv.read_inout %ic_replay : !hw.inout<i2>
    %282 = comb.extract %281 from 1 {sv.namehint = "_T_524"} : (i2) -> i1
    %283 = comb.or bin %39, %282 {sv.namehint = "_T_525"} : i1
    %284 = comb.icmp bin eq %41, %c-1_i2 {sv.namehint = "_T_557"} : i2
    %285 = comb.and bin %284, %283 {sv.namehint = "_T_526"} : i1
    %286 = comb.or bin %280, %285 : i1
    %_T_527 = sv.wire  : !hw.inout<i1>
    sv.assign %_T_527, %286 : i1
    %287 = sv.read_inout %valid : !hw.inout<i2>
    %288 = comb.extract %287 from 0 {sv.namehint = "_T_529"} : (i2) -> i1
    %289 = sv.read_inout %valid : !hw.inout<i2>
    %290 = comb.extract %289 from 1 {sv.namehint = "_T_534"} : (i2) -> i1
    %291 = sv.read_inout %_T_527 : !hw.inout<i1>
    %292 = comb.or %43, %290 : i1
    %293 = comb.or %31, %291 : i1
    %294 = comb.or %292, %293 {sv.namehint = "_T_542"} : i1
    %295 = comb.and bin %288, %294 {sv.namehint = "_T_543"} : i1
    %296 = sv.read_inout %_T_1859_bits : !hw.inout<i32>
    %297 = comb.extract %296 from 13 : (i32) -> i1
    %298 = sv.read_inout %_T_1859_bits : !hw.inout<i32>
    %299 = comb.extract %298 from 0 : (i32) -> i7
    %300 = comb.concat %297, %299 : i1, i7
    %301 = comb.icmp eq %300, %c3_i8 {sv.namehint = "_T_2763"} : i8
    %302 = sv.read_inout %_T_1859_bits : !hw.inout<i32>
    %303 = comb.extract %302 from 13 : (i32) -> i2
    %304 = sv.read_inout %_T_1859_bits : !hw.inout<i32>
    %305 = comb.extract %304 from 6 : (i32) -> i1
    %306 = sv.read_inout %_T_1859_bits : !hw.inout<i32>
    %307 = comb.extract %306 from 3 : (i32) -> i2
    %308 = sv.read_inout %_T_1859_bits : !hw.inout<i32>
    %309 = comb.extract %308 from 0 : (i32) -> i2
    %310 = comb.concat %303, %305, %307, %309 : i2, i1, i2, i2
    %311 = comb.icmp eq %310, %c35_i7 {sv.namehint = "_T_2661"} : i7
    %312 = sv.read_inout %_T_1859_bits : !hw.inout<i32>
    %313 = comb.extract %312 from 27 : (i32) -> i2
    %314 = comb.concat %313, %303, %299 : i2, i2, i7
    %315 = comb.icmp eq %314, %c175_i11 {sv.namehint = "_T_2669"} : i11
    %316 = sv.read_inout %_T_1859_bits : !hw.inout<i32>
    %317 = comb.extract %316 from 29 : (i32) -> i3
    %318 = sv.read_inout %_T_1859_bits : !hw.inout<i32>
    %319 = comb.extract %318 from 27 : (i32) -> i1
    %320 = comb.concat %317, %319, %303, %299 : i3, i1, i2, i7
    %321 = comb.icmp eq %320, %c687_i13 {sv.namehint = "_T_2693"} : i13
    %322 = sv.read_inout %_T_1859_bits : !hw.inout<i32>
    %323 = comb.extract %322 from 27 : (i32) -> i5
    %324 = sv.read_inout %_T_1859_bits : !hw.inout<i32>
    %325 = comb.extract %324 from 20 : (i32) -> i5
    %326 = comb.concat %323, %325, %303, %299 : i5, i5, i2, i7
    %327 = comb.icmp eq %326, %c32943_i19 {sv.namehint = "_T_2697"} : i19
    %328 = sv.read_inout %_T_1859_bits : !hw.inout<i32>
    %329 = comb.extract %328 from 2 : (i32) -> i3
    %330 = comb.concat %305, %329 : i1, i3
    %331 = comb.icmp eq %330, %c1_i4 {sv.namehint = "_T_2813"} : i4
    %332 = sv.read_inout %_T_1859_bits : !hw.inout<i32>
    %333 = comb.extract %332 from 5 : (i32) -> i2
    %334 = comb.icmp eq %333, %c-2_i2 {sv.namehint = "_T_2811"} : i2
    %335 = comb.or bin %331, %334 : i1
    %_T_2814 = sv.wire  : !hw.inout<i1>
    sv.assign %_T_2814, %335 : i1
    %336 = sv.read_inout %_T_1859_bits : !hw.inout<i32>
    %337 = comb.extract %336 from 2 : (i32) -> i1
    %338 = comb.concat %333, %337 : i2, i1
    %339 = sv.wire  {hw.verilogName = "_GEN_3"} : !hw.inout<i3>
    sv.assign %339, %338 : i3
    %340 = sv.read_inout %339 : !hw.inout<i3>
    %341 = comb.icmp eq %340, %c2_i3 {sv.namehint = "_T_2851"} : i3
    %342 = sv.read_inout %_T_1859_bits : !hw.inout<i32>
    %343 = comb.extract %342 from 4 : (i32) -> i2
    %344 = comb.concat %343, %337 : i2, i1
    %345 = comb.icmp eq %344, %c-4_i3 {sv.namehint = "_T_2841"} : i3
    %346 = sv.read_inout %_T_1859_bits : !hw.inout<i32>
    %347 = comb.extract %346 from 3 : (i32) -> i1
    %348 = comb.concat %297, %305, %347 : i1, i1, i1
    %349 = comb.icmp eq %348, %c-3_i3 {sv.namehint = "_T_2845"} : i3
    %350 = sv.read_inout %_T_1859_bits : !hw.inout<i32>
    %351 = comb.extract %350 from 30 : (i32) -> i1
    %352 = sv.read_inout %_T_1859_bits : !hw.inout<i32>
    %353 = comb.extract %352 from 25 : (i32) -> i1
    %354 = sv.read_inout %_T_1859_bits : !hw.inout<i32>
    %355 = comb.extract %354 from 12 : (i32) -> i2
    %356 = sv.read_inout %_T_1859_bits : !hw.inout<i32>
    %357 = comb.extract %356 from 5 : (i32) -> i1
    %358 = comb.concat %351, %353, %355, %357, %337 : i1, i1, i2, i1, i1
    %359 = comb.icmp eq %358, %c18_i6 {sv.namehint = "_T_2849"} : i6
    %360 = comb.or %341, %345 : i1
    %361 = comb.or %349, %359 : i1
    %362 = comb.or %360, %361 {sv.namehint = "_T_2854"} : i1
    %363 = comb.concat %305, %337 : i1, i1
    %364 = comb.icmp eq %363, %c0_i2 {sv.namehint = "_T_2876"} : i2
    %365 = sv.read_inout %_T_1859_bits : !hw.inout<i32>
    %366 = comb.extract %365 from 14 : (i32) -> i1
    %367 = comb.concat %366, %357, %337 : i1, i1, i1
    %368 = comb.icmp eq %367, %c2_i3 {sv.namehint = "_T_2862"} : i3
    %369 = sv.read_inout %_T_1859_bits : !hw.inout<i32>
    %370 = comb.extract %369 from 3 : (i32) -> i3
    %371 = comb.icmp eq %370, %c-4_i3 {sv.namehint = "_T_2866"} : i3
    %372 = sv.read_inout %_T_1859_bits : !hw.inout<i32>
    %373 = comb.extract %372 from 4 : (i32) -> i1
    %374 = comb.concat %297, %305, %373 : i1, i1, i1
    %375 = comb.icmp eq %374, %c-4_i3 {sv.namehint = "_T_2870"} : i3
    %376 = sv.read_inout %_T_1859_bits : !hw.inout<i32>
    %377 = comb.extract %376 from 31 : (i32) -> i1
    %378 = sv.read_inout %_T_1859_bits : !hw.inout<i32>
    %379 = comb.extract %378 from 28 : (i32) -> i1
    %380 = comb.concat %377, %379, %343, %337 : i1, i1, i2, i1
    %381 = comb.icmp eq %380, %c-6_i5 {sv.namehint = "_T_2874"} : i5
    %382 = comb.or %364, %368 : i1
    %383 = comb.or %375, %381 : i1
    %384 = comb.or %371, %383 : i1
    %385 = comb.or %382, %384 {sv.namehint = "_T_2880"} : i1
    %386 = sv.read_inout %_T_1859_bits : !hw.inout<i32>
    %387 = comb.extract %386 from 0 : (i32) -> i5
    %388 = comb.concat %366, %305, %387 : i1, i1, i5
    %389 = comb.icmp eq %388, %c3_i7 {sv.namehint = "_T_3095"} : i7
    %390 = sv.read_inout %_T_1859_bits : !hw.inout<i32>
    %391 = comb.extract %390 from 12 : (i32) -> i1
    %392 = comb.concat %391, %299 : i1, i7
    %393 = comb.icmp eq %392, %c3_i8 {sv.namehint = "_T_3089"} : i8
    %394 = sv.read_inout %_T_1859_bits : !hw.inout<i32>
    %395 = comb.extract %394 from 12 : (i32) -> i3
    %396 = comb.concat %395, %299 : i3, i7
    %397 = comb.icmp eq %396, %c143_i10 {sv.namehint = "_T_3093"} : i10
    %398 = comb.or %389, %301 : i1
    %399 = comb.or %393, %397 : i1
    %400 = comb.or %398, %399 : i1
    %401 = comb.or %311, %315 : i1
    %402 = comb.or %321, %327 : i1
    %403 = comb.or %401, %402 : i1
    %404 = comb.or %400, %403 {sv.namehint = "_T_3102"} : i1
    %405 = sv.read_inout %_T_1859_bits : !hw.inout<i32>
    %406 = comb.extract %405 from 4 : (i32) -> i3
    %407 = comb.concat %353, %406, %337 : i1, i3, i1
    %408 = comb.icmp eq %407, %c-10_i5 {sv.namehint = "_T_3236"} : i5
    %409 = sv.read_inout %339 : !hw.inout<i3>
    %410 = comb.icmp eq %409, %c0_i3 {sv.namehint = "_T_3266"} : i3
    %411 = comb.concat %305, %373 : i1, i1
    %412 = comb.icmp eq %411, %c1_i2 {sv.namehint = "_T_3244"} : i2
    %413 = comb.concat %297, %357, %337 : i1, i1, i1
    %414 = comb.icmp eq %413, %c3_i3 {sv.namehint = "_T_3248"} : i3
    %415 = comb.concat %357, %347 : i1, i1
    %416 = comb.icmp eq %415, %c-1_i2 {sv.namehint = "_T_3252"} : i2
    %417 = comb.concat %391, %343 : i1, i2
    %418 = comb.icmp eq %417, %c-1_i3 {sv.namehint = "_T_3256"} : i3
    %419 = comb.concat %297, %343 : i1, i2
    %420 = comb.icmp eq %419, %c-1_i3 {sv.namehint = "_T_3260"} : i3
    %421 = comb.concat %377, %379, %373 : i1, i1, i1
    %422 = comb.icmp eq %421, %c-3_i3 {sv.namehint = "_T_3264"} : i3
    %423 = comb.or %412, %414 : i1
    %424 = comb.or %410, %423 : i1
    %425 = comb.or %416, %418 : i1
    %426 = comb.or %420, %422 : i1
    %427 = comb.or %425, %426 : i1
    %428 = comb.or %424, %427 {sv.namehint = "_T_3272"} : i1
    %429 = comb.concat %391, %406 : i1, i3
    %430 = comb.icmp eq %429, %c-1_i4 {sv.namehint = "_T_3278"} : i4
    %431 = comb.concat %297, %406 : i1, i3
    %432 = comb.icmp eq %431, %c-1_i4 {sv.namehint = "_T_3284"} : i4
    %433 = comb.concat %355, %406 : i2, i3
    %434 = comb.icmp eq %433, %c7_i5 {sv.namehint = "_T_3290"} : i5
    %435 = comb.concat %434, %432, %430 : i1, i1, i1
    %_T_3292 = sv.wire  : !hw.inout<i3>
    sv.assign %_T_3292, %435 : i3
    %436 = comb.concat %355, %305, %307 : i2, i1, i2
    %437 = comb.icmp eq %436, %c9_i5 {sv.namehint = "_T_3298"} : i5
    %438 = comb.concat %303, %305, %347 : i2, i1, i1
    %439 = comb.icmp eq %438, %c5_i4 {sv.namehint = "_T_3310"} : i4
    %440 = sv.read_inout %_T_4428 : !hw.inout<i1>
    %441 = comb.xor bin %440, %true {sv.namehint = "_T_4490"} : i1
    %442 = sv.read_inout %_T_3292 : !hw.inout<i3>
    %443 = comb.icmp bin eq %442, %c2_i3 {sv.namehint = "_T_3457"} : i3
    %444 = sv.read_inout %_T_3292 : !hw.inout<i3>
    %445 = comb.icmp bin eq %444, %c3_i3 {sv.namehint = "_T_3458"} : i3
    %446 = sv.read_inout %_T_3292 : !hw.inout<i3>
    %447 = comb.icmp bin eq %446, %c1_i3 {sv.namehint = "_T_3452"} : i3
    %448 = comb.or %445, %447 : i1
    %449 = comb.or %443, %448 {sv.namehint = "id_csr_en"} : i1
    %450 = sv.read_inout %_T_1859_bits : !hw.inout<i32>
    %451 = comb.extract %450 from 26 {sv.namehint = "id_amo_aq"} : (i32) -> i1
    %452 = comb.xor bin %io_dmem_ordered, %true {sv.namehint = "_T_3503"} : i1
    %453 = comb.and bin %439, %451 {sv.namehint = "_T_3514"} : i1
    %454 = comb.or bin %453, %437 {sv.namehint = "_T_3515"} : i1
    %455 = comb.and bin %452, %454 {sv.namehint = "id_do_fence"} : i1
    %456 = sv.read_inout %_T_1859_rs1 : !hw.inout<i5>
    %457 = comb.icmp bin ne %456, %c0_i5 {sv.namehint = "_T_4232"} : i5
    %458 = sv.read_inout %_T_1859_rs2 : !hw.inout<i5>
    %459 = comb.icmp bin ne %458, %c0_i5 {sv.namehint = "_T_4235"} : i5
    %460 = sv.read_inout %_T_1859_rd : !hw.inout<i5>
    %461 = comb.icmp bin ne %460, %c0_i5 {sv.namehint = "_T_4238"} : i5
    %462 = sv.read_inout %_T_1859_rs1 : !hw.inout<i5>
    %463 = comb.concat %c0_i27, %462 : i27, i5
    %464 = sv.wire  {hw.verilogName = "_GEN_4"} : !hw.inout<i32>
    sv.assign %464, %463 : i32
    %465 = sv.read_inout %464 : !hw.inout<i32>
    %466 = comb.shru bin %c0_i32, %465 : i32
    %_T_4252 = sv.wire  : !hw.inout<i32>
    sv.assign %_T_4252, %466 : i32
    %467 = sv.read_inout %_T_4252 : !hw.inout<i32>
    %468 = comb.extract %467 from 0 {sv.namehint = "_T_4253"} : (i32) -> i1
    %469 = comb.and %457, %468 : i1
    %470 = comb.and %385, %469 {sv.namehint = "_T_4254"} : i1
    %471 = sv.read_inout %_T_1859_rs2 : !hw.inout<i5>
    %472 = comb.concat %c0_i27, %471 : i27, i5
    %473 = sv.wire  {hw.verilogName = "_GEN_5"} : !hw.inout<i32>
    sv.assign %473, %472 : i32
    %474 = sv.read_inout %473 : !hw.inout<i32>
    %475 = comb.shru bin %c0_i32, %474 : i32
    %_T_4255 = sv.wire  : !hw.inout<i32>
    sv.assign %_T_4255, %475 : i32
    %476 = sv.read_inout %_T_4255 : !hw.inout<i32>
    %477 = comb.extract %476 from 0 {sv.namehint = "_T_4256"} : (i32) -> i1
    %478 = comb.and %459, %477 : i1
    %479 = comb.and %362, %478 {sv.namehint = "_T_4257"} : i1
    %480 = sv.read_inout %_T_1859_rd : !hw.inout<i5>
    %481 = comb.concat %c0_i27, %480 : i27, i5
    %482 = sv.wire  {hw.verilogName = "_GEN_6"} : !hw.inout<i32>
    sv.assign %482, %481 : i32
    %483 = sv.read_inout %482 : !hw.inout<i32>
    %484 = comb.shru bin %c0_i32, %483 : i32
    %_T_4258 = sv.wire  : !hw.inout<i32>
    sv.assign %_T_4258, %484 : i32
    %485 = sv.read_inout %_T_4258 : !hw.inout<i32>
    %486 = comb.extract %485 from 0 {sv.namehint = "_T_4259"} : (i32) -> i1
    %487 = comb.and %461, %486 : i1
    %488 = comb.and %428, %487 {sv.namehint = "_T_4260"} : i1
    %489 = comb.xor bin %io_fpu_fcsr_rdy, %true {sv.namehint = "_T_4375"} : i1
    %490 = comb.and bin %449, %489 {sv.namehint = "_T_4376"} : i1
    %491 = sv.read_inout %464 : !hw.inout<i32>
    %492 = comb.shru bin %c0_i32, %491 : i32
    %_T_4377 = sv.wire  : !hw.inout<i32>
    sv.assign %_T_4377, %492 : i32
    %493 = sv.read_inout %_T_4377 : !hw.inout<i32>
    %494 = comb.extract %493 from 0 {sv.namehint = "_T_4378"} : (i32) -> i1
    %495 = comb.and bin %io_fpu_dec_ren1, %494 {sv.namehint = "_T_4379"} : i1
    %496 = sv.read_inout %473 : !hw.inout<i32>
    %497 = comb.shru bin %c0_i32, %496 : i32
    %_T_4380 = sv.wire  : !hw.inout<i32>
    sv.assign %_T_4380, %497 : i32
    %498 = sv.read_inout %_T_4380 : !hw.inout<i32>
    %499 = comb.extract %498 from 0 {sv.namehint = "_T_4381"} : (i32) -> i1
    %500 = comb.and bin %io_fpu_dec_ren2, %499 {sv.namehint = "_T_4382"} : i1
    %501 = comb.concat %c0_i27, %59 : i27, i5
    %502 = comb.shru bin %c0_i32, %501 : i32
    %_T_4383 = sv.wire  : !hw.inout<i32>
    sv.assign %_T_4383, %502 : i32
    %503 = sv.read_inout %_T_4383 : !hw.inout<i32>
    %504 = comb.extract %503 from 0 {sv.namehint = "_T_4384"} : (i32) -> i1
    %505 = comb.and bin %io_fpu_dec_ren3, %504 {sv.namehint = "_T_4385"} : i1
    %506 = sv.read_inout %482 : !hw.inout<i32>
    %507 = comb.shru bin %c0_i32, %506 : i32
    %_T_4386 = sv.wire  : !hw.inout<i32>
    sv.assign %_T_4386, %507 : i32
    %508 = sv.read_inout %_T_4386 : !hw.inout<i32>
    %509 = comb.extract %508 from 0 {sv.namehint = "_T_4387"} : (i32) -> i1
    %510 = comb.and bin %io_fpu_dec_wen, %509 {sv.namehint = "_T_4388"} : i1
    %511 = comb.or %490, %495 : i1
    %512 = comb.or %505, %510 : i1
    %513 = comb.or %500, %512 : i1
    %514 = comb.or %511, %513 {sv.namehint = "id_stall_fpu"} : i1
    %dcache_blocked = sv.reg  : !hw.inout<i1>
    %515 = sv.read_inout %_T_2814 : !hw.inout<i1>
    %516 = comb.and bin %515, %514 {sv.namehint = "_T_4408"} : i1
    %517 = sv.read_inout %dcache_blocked : !hw.inout<i1>
    %518 = comb.and bin %404, %517 {sv.namehint = "_T_4410"} : i1
    %519 = comb.xor bin %295, %true {sv.namehint = "_T_4425"} : i1
    %520 = sv.read_inout %_T_527 : !hw.inout<i1>
    %521 = comb.or %519, %520 : i1
    %522 = comb.or %470, %479 : i1
    %523 = comb.or %521, %522 : i1
    %524 = comb.or %488, %516 : i1
    %525 = comb.or %408, %455 : i1
    %526 = comb.or %518, %525 : i1
    %527 = comb.or %524, %526 : i1
    %528 = comb.or %523, %527 : i1
    sv.assign %_T_4428, %528 : i1
    %529 = sv.read_inout %_T_2814 : !hw.inout<i1>
    %530 = comb.and bin %441, %529 {sv.namehint = "_T_4491"} : i1
    sv.always posedge %clock {
      %true_1 = hw.constant true
      %531 = comb.xor bin %io_dmem_req_ready, %true_1 {sv.namehint = "_T_4394"} : i1
      %532 = sv.read_inout %dcache_blocked : !hw.inout<i1>
      %533 = comb.and bin %531, %532 {sv.namehint = "_T_4396"} : i1
      sv.passign %dcache_blocked, %533 : i1
    }
    sv.ifdef  "SYNTHESIS" {
    } else {
      sv.always posedge %clock {
        %_T_485 = sv.logic  : !hw.inout<i1>
        %true_1 = hw.constant true
        %531 = comb.icmp bin uge %io_imem_resp_bits_btb_bits_bridx, %0 {sv.namehint = "_T_483"} : i1
        %532 = comb.xor bin %io_imem_resp_bits_btb_valid, %true_1 {sv.namehint = "_T_482"} : i1
        %533 = comb.or %531, %reset : i1
        %534 = comb.or %532, %533 : i1
        sv.bpassign %_T_485, %534 : i1
        %PRINTF_COND_ = sv.macro.ref< "PRINTF_COND_"> : i1
        %true_2 = hw.constant true
        %535 = sv.read_inout %_T_485 : !hw.inout<i1>
        %536 = comb.xor bin %535, %true_2 {sv.namehint = "_T_487"} : i1
        %537 = comb.and bin %PRINTF_COND_, %536 : i1
        sv.if %537 {
          %c-2147483646_i32 = hw.constant -2147483646 : i32
          sv.fwrite %c-2147483646_i32, "Assertion failed\0A    at IBuf.scala:84 assert(!io.imem.bits.btb.valid || io.imem.bits.btb.bits.bridx >= pcWordBits)\0A"
        }
        %STOP_COND_ = sv.macro.ref< "STOP_COND_"> : i1
        %true_3 = hw.constant true
        %538 = sv.read_inout %_T_485 : !hw.inout<i1>
        %539 = comb.xor bin %538, %true_3 {sv.namehint = "_T_487"} : i1
        %540 = comb.and bin %STOP_COND_, %539 : i1
        sv.if %540 {
          sv.fatal 1
        }
        %541 = comb.xor bin %reset, %true_1 {sv.namehint = "_T_4587"} : i1
        %542 = comb.and bin %PRINTF_COND_, %541 : i1
        sv.if %542 {
          %dmem_resp_valid = sv.logic  : !hw.inout<i1>
          %_T_4183 = sv.logic  : !hw.inout<i1>
          %false_4 = hw.constant false
          %c-2147483646_i32 = hw.constant -2147483646 : i32
          %c0_i40 = hw.constant 0 : i40
          %c0_i5_5 = hw.constant 0 : i5
          %c0_i64 = hw.constant 0 : i64
          %c0_i32_6 = hw.constant 0 : i32
          %543 = comb.extract %io_dmem_resp_bits_tag from 0 {sv.namehint = "dmem_resp_fpu"} : (i7) -> i1
          %544 = comb.and bin %io_dmem_resp_valid, %io_dmem_resp_bits_has_data : i1
          sv.bpassign %dmem_resp_valid, %544 : i1
          %true_7 = hw.constant true
          %545 = comb.xor bin %543, %true_7 {sv.namehint = "dmem_resp_xpu"} : i1
          %546 = sv.read_inout %dmem_resp_valid : !hw.inout<i1>
          %547 = comb.and %io_dmem_resp_bits_replay, %545 : i1
          %548 = comb.and %546, %547 : i1
          sv.bpassign %_T_4183, %548 : i1
          %549 = comb.extract %io_dmem_resp_bits_tag from 1 {sv.namehint = "dmem_resp_waddr"} : (i7) -> i5
          %550 = sv.read_inout %_T_4183 : !hw.inout<i1>
          %551 = comb.mux bin %550, %549, %c0_i5_5 : i5
          %true_8 = hw.constant true
          %552 = comb.xor bin %543, %true_8 {sv.namehint = "dmem_resp_xpu"} : i1
          %553 = sv.read_inout %dmem_resp_valid : !hw.inout<i1>
          %554 = comb.and bin %553, %552 {sv.namehint = "_T_4191"} : i1
          %555 = comb.mux bin %554, %io_dmem_resp_bits_data, %c0_i64 {sv.namehint = "rf_wdata"} : i64
          %556 = sv.read_inout %_T_4183 : !hw.inout<i1>
          sv.fwrite %c-2147483646_i32, "C%d: %d [%d] pc=[%x] W[r%d=%x][%d] R[r%d=%x] R[r%d=%x] inst=[%x] DASM(%x)\0A"(%io_hartid, %c0_i32_6, %false_4, %c0_i40, %551, %555, %556, %c0_i5_5, %c0_i64, %c0_i5_5, %c0_i64, %c0_i32_6, %c0_i32_6) : i64, i32, i1, i40, i5, i64, i1, i5, i64, i5, i64, i32, i32
        }
      }
      sv.ordered {
        sv.ifdef  "FIRRTL_BEFORE_INITIAL" {
          sv.verbatim "`FIRRTL_BEFORE_INITIAL" {symbols = []}
        }
        sv.initial {
          %_RANDOM_0 = sv.logic  : !hw.inout<i32>
          sv.ifdef.procedural  "INIT_RANDOM_PROLOG_" {
            sv.verbatim "`INIT_RANDOM_PROLOG_" {symbols = []}
          }
          sv.ifdef.procedural  "RANDOMIZE_REG_INIT" {
            %RANDOM = sv.macro.ref.se< "RANDOM"> : i32
            sv.bpassign %_RANDOM_0, %RANDOM : i32
            %531 = sv.read_inout %_RANDOM_0 : !hw.inout<i32>
            %532 = comb.extract %531 from 0 : (i32) -> i1
            sv.bpassign %dcache_blocked, %532 : i1
          }
        }
        sv.ifdef  "FIRRTL_AFTER_INITIAL" {
          sv.verbatim "`FIRRTL_AFTER_INITIAL" {symbols = []}
        }
      }
    }
    hw.output %false, %530 : i1, i1
  }
}
