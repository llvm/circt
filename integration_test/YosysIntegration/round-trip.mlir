// REQUIRES: yosys-integration

// RUN: circt-translate --export-rtlil %s | circt-translate --import-rtlil | circt-opt -canonicalize | FileCheck %s

// CHECK-LABEL: hw.module @Arith(in %in1 : i2, in %in2 : i2, in %in3 : i2, out add : i2, out sub : i2, out mul : i2, out and : i2, out or : i2, out xor : i2)
hw.module @Arith(in %in1 : i2, in %in2 : i2, in %in3: i2, out add : i2, out sub: i2, out mul: i2, out and: i2, out or: i2, out xor: i2 ) {
    %0 = comb.add %in1, %in2, %in3: i2
    %1 = comb.sub %in1, %in2: i2
    %2 = comb.mul %in1, %in2, %in3: i2
    %3 = comb.and %in1, %in2, %in3: i2
    %4 = comb.or %in1, %in2, %in3: i2
    %5 = comb.xor %in1, %in2, %in3: i2
    // CHECK-NEXT: %0 = comb.add %in1, %in2, %in3 : i2
    // CHECK-NEXT: %1 = comb.sub %in1, %in2 : i2
    // CHECK-NEXT: %2 = comb.mul %in1, %in2, %in3 : i2
    // CHECK-NEXT: %3 = comb.and %in1, %in2, %in3 : i2
    // CHECK-NEXT: %4 = comb.or %in1, %in2, %in3 : i2
    // CHECK-NEXT: %5 = comb.xor %in1, %in2, %in3 : i2
    // CHECK-NEXT: hw.output %0, %1, %2, %3, %4, %5 : i2, i2, i2, i2, i2, i2
    hw.output %0, %1, %2, %3, %4, %5 : i2, i2, i2, i2, i2, i2
}

// CHECK-LABEL: hw.module @ICmp(in %a : i2, in %b : i2, out eq : i1, out ne : i1, out slt : i1, out sle : i1, out sgt : i1, out sge : i1, out ult : i1, out ule : i1, out ugt : i1, out uge : i1)
hw.module @ICmp(in %a : i2, in %b : i2, out eq : i1, 
                out ne: i1, out slt: i1, out sle: i1, out sgt: i1, out sge: i1,
                out ult: i1, out ule: i1, out ugt: i1, out uge: i1
                ) {
  %eq = comb.icmp eq %a, %b : i2
  %ne = comb.icmp ne %a, %b : i2
  %slt = comb.icmp slt %a, %b : i2
  %sle = comb.icmp sle %a, %b : i2
  %sgt = comb.icmp sgt %a, %b : i2
  %sge = comb.icmp sge %a, %b : i2
  %ult = comb.icmp ult %a, %b : i2
  %ule = comb.icmp ule %a, %b : i2
  %ugt = comb.icmp ugt %a, %b : i2
  %uge = comb.icmp uge %a, %b : i2
  // CHECK-NEXT: %0 = comb.icmp eq %a, %b : i2
  // CHECK-NEXT: %1 = comb.icmp ne %a, %b : i2
  // CHECK-NEXT: %2 = comb.icmp slt %a, %b : i2
  // CHECK-NEXT: %3 = comb.icmp sle %a, %b : i2
  // CHECK-NEXT: %4 = comb.icmp sgt %a, %b : i2
  // CHECK-NEXT: %5 = comb.icmp sge %a, %b : i2
  // CHECK-NEXT: %6 = comb.icmp ult %a, %b : i2
  // CHECK-NEXT: %7 = comb.icmp ule %a, %b : i2
  // CHECK-NEXT: %8 = comb.icmp ugt %a, %b : i2
  // CHECK-NEXT: %9 = comb.icmp uge %a, %b : i2
  // CHECK-NEXT: hw.output %0, %1, %2, %3, %4, %5, %6, %7, %8, %9 : i1, i1, i1, i1, i1, i1, i1, i1, i1, i1
  hw.output %eq, %ne, %slt, %sle, %sgt, %sge, %ult, %ule, %ugt, %uge : i1, i1, i1, i1, i1, i1, i1, i1, i1, i1
}


// CHECK-LABEL: hw.module @counter(in %clk : i1, out o : i8) 
hw.module @counter(in %clk: i1, out o: i8) {
  // CHECK-NEXT: %c1_i8 = hw.constant 1 : i8
  // CHECK-NEXT: %0 = seq.to_clock %clk
  // CHECK-NEXT: %reg = seq.compreg %1, %0 : i8  
  // CHECK-NEXT: %1 = comb.add %reg, %c1_i8 : i8
  // CHECK-NEXT: hw.output %reg : i8
  %seq_clk = seq.to_clock %clk
  %reg = seq.compreg %added, %seq_clk : i8
  %one = hw.constant 1 : i8
  %added = comb.add %reg, %one : i8
  hw.output %reg : i8
}

// CHECK-LABEL:  hw.module @misc(in %cond : i1, in %in1 : i2, in %in2 : i2, in %in3 : i5, out mux : i2, out extract : i2, out concat : i9, out replicate : i6, out shl : i5, out parity : i1)
hw.module @misc(in %cond:i1, in %in1 : i2, in %in2 : i2, in %in3: i5,
                out mux : i2, out extract: i2, out concat: i9, out replicate: i6, out shl: i5, out parity: i1 ) {
  // CHECK-NEXT: %0 = comb.extract %in3 from 3 : (i5) -> i2
  // CHECK-NEXT: %1 = comb.concat %in1, %in2, %in3 : i2, i2, i5
  // CHECK-NEXT: %2 = comb.replicate %in1 : (i2) -> i6
  // CHECK-NEXT: %3 = comb.mux %cond, %in1, %in2 : i2
  // CHECK-NEXT: %4 = comb.shl %in3, %in3 : i5
  // CHECK-NEXT: %5 = comb.parity %in3 : i5
  // CHECK-NEXT: hw.output %3, %0, %1, %2, %4, %5 : i2, i2, i9, i6, i5, i1
  %mux = comb.mux %cond, %in1, %in2 : i2
  %extract = comb.extract %in3 from 3 : (i5) -> i2
  %concat = comb.concat %in1, %in2, %in3 : i2, i2, i5
  %replicate = comb.replicate %in1 : (i2) -> i6
  %shl = comb.shl %in3, %in3 : i5
  %partiy = comb.parity %in3 : i5
  hw.output %mux, %extract, %concat, %replicate, %shl, %partiy:  i2,  i2,  i9,  i6,  i5,  i1
}
