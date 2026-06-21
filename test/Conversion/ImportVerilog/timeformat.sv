// RUN: circt-translate --import-verilog %s | FileCheck %s
// REQUIRES: slang

// CHECK: moore.global_variable @__timeformat_state : !moore.ustruct<{unit: i32, precision: i32, suffix: string, min_width: i32}> init {
// CHECK: [[UNIT0:%.+]] = moore.constant -15 : i32
// CHECK: [[PREC0:%.+]] = moore.constant 0 : i32
// CHECK: [[EMPTY:%.+]] = moore.constant_string "" : i0
// CHECK: [[SUF0:%.+]] = moore.int_to_string [[EMPTY]] : i0
// CHECK: [[WIDTH0:%.+]] = moore.constant 20 : i32
// CHECK: [[INIT:%.+]] = moore.struct_create [[UNIT0]], [[PREC0]], [[SUF0]], [[WIDTH0]] : !moore.i32, !moore.i32, !moore.string, !moore.i32 -> ustruct<{unit: i32, precision: i32, suffix: string, min_width: i32}>
// CHECK: moore.yield [[INIT]]
// CHECK: }

// CHECK-LABEL: moore.module @TimeFormatBasic
module TimeFormatBasic;
  // CHECK: moore.procedure initial {
  // CHECK: [[C9:%.+]] = moore.constant 9 : i32
  // CHECK: [[U:%.+]] = moore.neg [[C9]] : i32
  // CHECK: [[BASE0:%.+]] = moore.get_global_variable @__timeformat_state
  // CHECK: [[UREF:%.+]] = moore.struct_extract_ref [[BASE0]], "unit"
  // CHECK: moore.blocking_assign [[UREF]], [[U]] : i32
  // CHECK: [[P:%.+]] = moore.constant 5 : i32
  // CHECK: [[BASE1:%.+]] = moore.get_global_variable @__timeformat_state
  // CHECK: [[PREF:%.+]] = moore.struct_extract_ref [[BASE1]], "precision"
  // CHECK: moore.blocking_assign [[PREF]], [[P]] : i32
  // CHECK: [[SI:%.+]] = moore.constant_string " ns" : i24
  // CHECK: [[S:%.+]] = moore.int_to_string [[SI]] : i24
  // CHECK: [[BASE2:%.+]] = moore.get_global_variable @__timeformat_state
  // CHECK: [[SREF:%.+]] = moore.struct_extract_ref [[BASE2]], "suffix"
  // CHECK: moore.blocking_assign [[SREF]], [[S]] : string
  // CHECK: [[W:%.+]] = moore.constant 10 : i32
  // CHECK: [[BASE3:%.+]] = moore.get_global_variable @__timeformat_state
  // CHECK: [[WREF:%.+]] = moore.struct_extract_ref [[BASE3]], "min_width"
  // CHECK: moore.blocking_assign [[WREF]], [[W]] : i32
  // CHECK: moore.return
  // CHECK: }
  initial begin
    $timeformat(-9, 5, " ns", 10);
  end
endmodule

// CHECK-LABEL: moore.module @TimeFormatNoArgs
module TimeFormatNoArgs;
  // CHECK: moore.procedure initial {
  // CHECK: [[UNIT_DEF:%.+]] = moore.constant -15 : i32
  // CHECK: [[PREC_DEF:%.+]] = moore.constant 0 : i32
  // CHECK: [[EMPTY:%.+]] = moore.constant_string "" : i0
  // CHECK: [[SUF_DEF:%.+]] = moore.int_to_string [[EMPTY]] : i0
  // CHECK: [[WIDTH_DEF:%.+]] = moore.constant 20 : i32
  // CHECK: [[BASE0:%.+]] = moore.get_global_variable @__timeformat_state
  // CHECK: [[UREF:%.+]] = moore.struct_extract_ref [[BASE0]], "unit"
  // CHECK: moore.blocking_assign [[UREF]], [[UNIT_DEF]] : i32
  // CHECK: [[BASE1:%.+]] = moore.get_global_variable @__timeformat_state
  // CHECK: [[PREF:%.+]] = moore.struct_extract_ref [[BASE1]], "precision"
  // CHECK: moore.blocking_assign [[PREF]], [[PREC_DEF]] : i32
  // CHECK: [[BASE2:%.+]] = moore.get_global_variable @__timeformat_state
  // CHECK: [[SREF:%.+]] = moore.struct_extract_ref [[BASE2]], "suffix"
  // CHECK: moore.blocking_assign [[SREF]], [[SUF_DEF]] : string
  // CHECK: [[BASE3:%.+]] = moore.get_global_variable @__timeformat_state
  // CHECK: [[WREF:%.+]] = moore.struct_extract_ref [[BASE3]], "min_width"
  // CHECK: moore.blocking_assign [[WREF]], [[WIDTH_DEF]] : i32
  // CHECK: moore.return
  // CHECK: }
  initial begin
    $timeformat();
  end
endmodule

// CHECK-LABEL: moore.module @TimeFormatPartial
module TimeFormatPartial;
  // CHECK: [[U:%.+]] = moore.variable
  int u;
  // CHECK: moore.procedure initial {
  // CHECK: [[U_VAL:%.+]] = moore.read [[U]]
  // CHECK: [[BASE:%.+]] = moore.get_global_variable @__timeformat_state
  // CHECK: [[UREF:%.+]] = moore.struct_extract_ref [[BASE]], "unit"
  // CHECK: moore.blocking_assign [[UREF]], [[U_VAL]] : i32
  // CHECK: moore.return
  // CHECK: }
  // CHECK-NOT: moore.struct_extract_ref {{%.+}}, "precision"
  initial begin
    $timeformat(u);
  end
endmodule
