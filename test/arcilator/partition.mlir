// RUN: arcilator %s --partition-chunks=2

module {
  hw.module @gcd(in %clk : !seq.clock, in %rst : i1, in %in_a : i64, in %in_b : i64, in %in_valid : i1, out in_ready : i1, out out : i64, out out_valid : i64) {
    %c0_i63 = hw.constant 0 : i63
    %true = hw.constant true
    %c0_i64 = hw.constant 0 : i64
    %false = hw.constant false
    %smaller = seq.firreg %13 clock %clk {firrtl.random_init_start = 0 : ui64} : i64
    %larger = seq.firreg %11 clock %clk {firrtl.random_init_start = 64 : ui64} : i64
    %working = seq.firreg %15 clock %clk reset sync %rst, %false {firrtl.random_init_start = 128 : ui64} : i1
    %0 = comb.icmp bin eq %smaller, %c0_i64 : i64
    %1 = comb.and bin %0, %working {sv.namehint = "done"} : i1
    %2 = comb.concat %false, %larger : i1, i64
    %3 = comb.concat %false, %smaller : i1, i64
    %4 = comb.sub bin %2, %3 : i65
    %5 = comb.concat %false, %in_a : i1, i64
    %6 = comb.mux bin %working, %4, %5 {sv.namehint = "na"} : i65
    %7 = comb.mux bin %working, %smaller, %in_b {sv.namehint = "nb"} : i64
    %8 = comb.concat %false, %7 : i1, i64
    %9 = comb.icmp bin ugt %6, %8 : i65
    %10 = comb.extract %6 from 0 : (i65) -> i64
    %11 = comb.mux %9, %10, %7 {sv.namehint = "nlarger"} : i64
    %12 = comb.extract %6 from 0 : (i65) -> i64
    %13 = comb.mux %9, %7, %12 {sv.namehint = "nsmaller"} : i64
    %14 = comb.xor bin %1, %true : i1
    %15 = comb.mux bin %working, %14, %in_valid {sv.namehint = "nworking"} : i1
    %16 = comb.concat %c0_i63, %1 : i63, i1
    %17 = comb.xor bin %working, %true : i1
    hw.output %17, %larger, %16 : i1, i64, i64
  }
}
