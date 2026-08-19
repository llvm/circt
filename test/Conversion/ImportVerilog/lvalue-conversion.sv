// RUN: circt-translate --import-verilog %s | FileCheck %s
// REQUIRES: slang

// Internal issue in Slang v3 about jump depending on uninitialised value.
// UNSUPPORTED: valgrind

function void funcInout(inout longint wideArg);
endfunction

function void funcOutput(output longint wideArg);
endfunction

task taskInout(inout longint wideArg);
endtask

task taskOutput(output longint wideArg);
endtask

// CHECK-LABEL: moore.module @LvalueConversions()
module LvalueConversions;
  real r;
  initial begin
    // CHECK: [[IN0:%.+]] = moore.read %r : <f64>
    // CHECK: [[CVT0:%.+]] = moore.real_to_int [[IN0]] : f64 -> i64
    // CHECK: [[TMP0:%.+]] = moore.variable [[CVT0]] : <i64>
    // CHECK: func.call @funcInout([[TMP0]])
    // CHECK: [[OUT0:%.+]] = moore.read [[TMP0]] : <i64>
    // CHECK: [[CVT1:%.+]] = moore.sint_to_real [[OUT0]] : i64 -> f64
    // CHECK: moore.blocking_assign %r, [[CVT1]] : f64
    funcInout(r);

    // CHECK: [[TMP1:%.+]] = moore.variable : <i64>
    // CHECK: func.call @funcOutput([[TMP1]])
    // CHECK: [[OUT1:%.+]] = moore.read [[TMP1]] : <i64>
    // CHECK: [[CVT2:%.+]] = moore.sint_to_real [[OUT1]] : i64 -> f64
    // CHECK: moore.blocking_assign %r, [[CVT2]] : f64
    funcOutput(r);

    // CHECK: [[IN1:%.+]] = moore.read %r : <f64>
    // CHECK: [[CVT3:%.+]] = moore.real_to_int [[IN1]] : f64 -> i64
    // CHECK: [[TMP2:%.+]] = moore.variable [[CVT3]] : <i64>
    // CHECK: moore.call_coroutine @taskInout([[TMP2]])
    // CHECK: [[OUT2:%.+]] = moore.read [[TMP2]] : <i64>
    // CHECK: [[CVT4:%.+]] = moore.sint_to_real [[OUT2]] : i64 -> f64
    // CHECK: moore.blocking_assign %r, [[CVT4]] : f64
    taskInout(r);

    // CHECK: [[TMP3:%.+]] = moore.variable : <i64>
    // CHECK: moore.call_coroutine @taskOutput([[TMP3]])
    // CHECK: [[OUT3:%.+]] = moore.read [[TMP3]] : <i64>
    // CHECK: [[CVT5:%.+]] = moore.sint_to_real [[OUT3]] : i64 -> f64
    // CHECK: moore.blocking_assign %r, [[CVT5]] : f64
    taskOutput(r);
  end
endmodule
