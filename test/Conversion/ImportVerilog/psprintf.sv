// RUN: circt-translate --import-verilog %s | FileCheck %s
// RUN: circt-verilog --ir-moore %s
// REQUIRES: slang
// UNSUPPORTED: valgrind

// CHECK-LABEL: func.func private @ReturnPSPrintf(
// CHECK-SAME: [[ARG:%[^,)]+]]: !moore.string
function automatic string ReturnPSPrintf
    (string arg);
  // CHECK: [[FMTARG:%.+]] = moore.fmt.string [[ARG]]
  // CHECK: [[LIT:%.+]] = moore.fmt.literal "!"
  // CHECK: [[FMT:%.+]] = moore.fmt.concat ([[FMTARG]], [[LIT]])
  // CHECK: [[STR:%.+]] = moore.fstring_to_string [[FMT]]
  // CHECK: return [[STR]] : !moore.string
  return $psprintf("%s!",
                   arg);
endfunction

// CHECK-LABEL: func.func private @ReturnSFormatF(
// CHECK-SAME: [[ARG:%[^,)]+]]: !moore.string
function automatic string ReturnSFormatF
    (string arg);
  // CHECK: [[FMTARG:%.+]] = moore.fmt.string [[ARG]]
  // CHECK: [[LIT:%.+]] = moore.fmt.literal "?"
  // CHECK: [[FMT:%.+]] = moore.fmt.concat ([[FMTARG]], [[LIT]])
  // CHECK: [[STR:%.+]] = moore.fstring_to_string [[FMT]]
  // CHECK: return [[STR]] : !moore.string
  return $sformatf("%s?",
                   arg);
endfunction
