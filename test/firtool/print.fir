; RUN: firtool %s --format=fir --ir-sv | FileCheck %s

FIRRTL version 4.0.0
circuit PrintTest:
  ; CHECK-LABEL: sv.func private @"__circt_lib_logging::FileDescriptor::get"
  ; CHECK-SAME: (in %name : !hw.string, out fd : i32 {sv.func.explicitly_returned}) attributes {verilogName = "__circt_lib_logging::FileDescriptor::get"}
  ; CHECK-LABEL: @PrintTest
  public module PrintTest :
    input clock : Clock
    input cond : UInt<1>
    input var : UInt<32>
    printf(clock, cond, "test %d\n", var)
    fprintf(clock, cond, "test.txt", "test %d\n", var)
    fprintf(clock, cond, "test.txt", "hello")
    fprintf(clock, cond, "%m.txt", "test %d\n", var)


    ; CHECK:      sv.ifdef  @SYNTHESIS {
    ; CHECK-NEXT: } else {
    ; CHECK-NEXT:   %fd_test.txt = sv.reg : !hw.inout<i32>
    ; CHECK-NEXT:   [[FD_M:%.+]] = sv.reg name "fd_%m.txt" : !hw.inout<i32>
    ; CHECK-NEXT:   sv.initial {
    ; CHECK-NEXT:     [[STR1:%.+]] = sv.constantStr "test.txt"
    ; CHECK-NEXT:     [[FMT1:%.+]] = sv.system "sformatf"([[STR1]]) : (!hw.string) -> !hw.string
    ; CHECK-NEXT:     [[FD1:%.+]] = sv.func.call.procedural @"__circt_lib_logging::FileDescriptor::get"([[FMT1]]) : (!hw.string) -> i32
    ; CHECK-NEXT:     sv.bpassign %fd_test.txt, [[FD1]] : i32
    ; CHECK-NEXT:     [[STR2:%.+]] = sv.constantStr "%m.txt"
    ; CHECK-NEXT:     [[FMT2:%.+]] = sv.system "sformatf"([[STR2]]) : (!hw.string) -> !hw.string
    ; CHECK-NEXT:     [[FD2:%.+]] = sv.func.call.procedural @"__circt_lib_logging::FileDescriptor::get"([[FMT2]]) : (!hw.string) -> i32
    ; CHECK-NEXT:     sv.bpassign [[FD_M]], [[FD2]] : i32
    ; CHECK-NEXT:   }
    ; CHECK-NEXT:   sv.always posedge %clock {
    ; CHECK-NEXT:     [[PRINTF_COND:%.+]] = sv.macro.ref.expr @PRINTF_COND_() : () -> i1
    ; CHECK-NEXT:     [[COND:%.+]] = comb.and bin [[PRINTF_COND]], %cond : i1
    ; CHECK-NEXT:     sv.if [[COND]] {
    ; CHECK-NEXT:       [[PRINTF_FD:%.+]] = sv.macro.ref.expr @PRINTF_FD_() : () -> i32
    ; CHECK-NEXT:       sv.fwrite [[PRINTF_FD]], "test %d\0A"(%var) : i32
    ; CHECK-NEXT:       [[FD3:%.+]] = sv.read_inout %fd_test.txt : !hw.inout<i32>
    ; CHECK-NEXT:       sv.fwrite [[FD3]], "test %d\0A"(%var) : i32
    ; CHECK-NEXT:       [[FD4:%.+]] = sv.read_inout %fd_test.txt : !hw.inout<i32>
    ; CHECK-NEXT:       sv.fwrite [[FD4]], "hello"
    ; CHECK-NEXT:       [[FD5:%.+]] = sv.read_inout [[FD_M]] : !hw.inout<i32>
    ; CHECK-NEXT:       sv.fwrite [[FD5]], "test %d\0A"(%var) : i32
    ; CHECK-NEXT:     }
    ; CHECK-NEXT:   }
    ; CHECK-NEXT: }
