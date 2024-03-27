// REQUIRES: verilator
// RUN: circt-opt %s -export-split-verilog='dir-name=%t.dir'
// RUN: verilator %driver --cc --sv --exe --build -I%t.dir -F %t.dir%{fs-sep}filelist.f -o %t.exe --top-module top
// RUN: %t.exe --cycles 10 2>&1 | FileCheck %s --check-prefix=DEFAULT

hw.module private @TargetA(in %a: i32, out b: i32) {
  %cst = hw.constant 5 : i32
  %out = comb.add %cst, %out : i32
  hw.output %out : i32
}

hw.module private @TargetB(in %a: i32, out b: i32) {
  %cst = hw.constant 15 : i32
  %out = comb.add %cst, %out : i32
  hw.output %out : i32
}

hw.module private @TargetDefault(in %a: i32, out b: i32) {
  hw.output %a : i32
}

hw.module public @top(in %clk : i1, in %rst : i1) {
  %reg = sv.reg : !hw.inout<i32>

  %a = sv.read_inout %reg : !hw.inout<i32>
  %b = hw.instance_choice "inst1" sym @inst1 option "Perf" @TargetDefault or @TargetA if "A" or @TargetB if "B"(a: %a: i32) -> (b: i32)

  sv.alwaysff(posedge %clk) {
    sv.if %rst {
      %zero = hw.constant 0 : i32
      sv.passign %reg, %zero : i32
    } else {
      %one = hw.constant 1 : i32

      %a_read = sv.read_inout %reg : !hw.inout<i32>
      %next = comb.add %a_read, %one : i32
      sv.passign %reg, %next : i32

      %fd = hw.constant 0x80000002 : i32
      sv.fwrite %fd, "b: %d\n" (%b) : i32
    }
  }
}

// DEFAULT: b:          0
// DEFAULT: b:          1
// DEFAULT: b:          2
// DEFAULT: b:          3
// DEFAULT: b:          4
// DEFAULT: b:          5
// DEFAULT: b:          6
// DEFAULT: b:          7
// DEFAULT: b:          8
// DEFAULT: b:          9
