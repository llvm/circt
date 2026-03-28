// RUN: circt-translate --import-verilog %s | FileCheck %s
// REQUIRES: slang

// Internal issue in Slang v3 about jump depending on uninitialised value.
// UNSUPPORTED: valgrind

// CHECK: moore.global_variable @__monitor_active_id : !moore.i32 init {
// CHECK:   [[ZERO:%.+]] = moore.constant 0 : i32
// CHECK:   moore.yield [[ZERO]] : i32
// CHECK: }

// CHECK: moore.global_variable @__monitor_enabled : !moore.i1 init {
// CHECK:   [[TRUE:%.+]] = moore.constant 1 : i1
// CHECK:   moore.yield [[TRUE]] : i1
// CHECK: }

// CHECK-LABEL: moore.module @MonitorBasic
module MonitorBasic;
  // CHECK: [[A:%.+]] = moore.variable
  int a;
  // CHECK: moore.procedure initial {
  // CHECK:   [[ID:%.+]] = moore.constant {{[0-9]+}} : i32
  // CHECK:   [[ACTIVE_REF:%.+]] = moore.get_global_variable @__monitor_active_id : <i32>
  // CHECK:   moore.blocking_assign [[ACTIVE_REF]], [[ID]] : i32
  // CHECK:   moore.return
  // CHECK: }
  initial begin
    $monitor("a=%d", a);
  end
  // CHECK: moore.procedure always_comb {
  // CHECK:   [[LIT:%.+]] = moore.fmt.literal "a="
  // CHECK:   [[A_VAL:%.+]] = moore.read [[A]] : <i32>
  // CHECK:   [[FMT:%.+]] = moore.fmt.int decimal [[A_VAL]], align right, pad space signed : i32
  // CHECK:   [[NEWLINE:%.+]] = moore.fmt.literal "\0A"
  // CHECK:   [[MSG:%.+]] = moore.fmt.concat ([[LIT]], [[FMT]], [[NEWLINE]])
  // CHECK:   [[MY_ID:%.+]] = moore.constant {{[0-9]+}} : i32
  // CHECK:   [[ACTIVE_REF:%.+]] = moore.get_global_variable @__monitor_active_id : <i32>
  // CHECK:   [[ACTIVE_ID:%.+]] = moore.read [[ACTIVE_REF]] : <i32>
  // CHECK:   [[IS_ACTIVE:%.+]] = moore.eq [[ACTIVE_ID]], [[MY_ID]] : i32 -> i1
  // CHECK:   [[ENABLED_REF:%.+]] = moore.get_global_variable @__monitor_enabled : <i1>
  // CHECK:   [[ENABLED:%.+]] = moore.read [[ENABLED_REF]] : <i1>
  // CHECK:   [[SHOULD_PRINT_MOORE:%.+]] = moore.and [[IS_ACTIVE]], [[ENABLED]] : i1
  // CHECK:   [[SHOULD_PRINT:%.+]] = moore.to_builtin_int [[SHOULD_PRINT_MOORE]] : i1
  // CHECK:   cf.cond_br [[SHOULD_PRINT]], ^[[BB1:.+]], ^[[BB2:.+]]
  // CHECK: ^[[BB1]]:
  // CHECK:   moore.builtin.display [[MSG]]
  // CHECK:   moore.return
  // CHECK: ^[[BB2]]:
  // CHECK:   moore.return
  // CHECK: }
endmodule

// Test $monitor[boh] variants - TODO: check formatting once implemented
// CHECK-LABEL: moore.module @MonitorFormats
module MonitorFormats;
  // CHECK: [[X:%.+]] = moore.variable
  int x;
  // CHECK: moore.procedure initial {
  // CHECK: }
  initial begin
    $monitor(x);
  end
  // $monitor(x) should format as decimal (same as $display(x))
  // CHECK: moore.procedure always_comb {
  // CHECK:   [[X_VAL_0:%.+]] = moore.read [[X]] : <i32>
  // CHECK:   [[FMT_0:%.+]] = moore.fmt.int decimal [[X_VAL_0]], align right, pad space signed : i32
  // CHECK:   [[NEWLINE_0:%.+]] = moore.fmt.literal "\0A"
  // CHECK:   [[MSG_0:%.+]] = moore.fmt.concat ([[FMT_0]], [[NEWLINE_0]])
  // CHECK:   moore.builtin.display [[MSG_0]]
  // CHECK: }

  // CHECK: moore.procedure initial {
  // CHECK: }
  initial begin
    $monitorb(x);
  end
  // $monitorb(x) should format as binary (same as $displayb(x))
  // CHECK: moore.procedure always_comb {
  // CHECK:   [[X_VAL_1:%.+]] = moore.read [[X]] : <i32>
  // CHECK:   [[FMT_1:%.+]] = moore.fmt.int binary [[X_VAL_1]], align right, pad zero : i32
  // CHECK:   [[NEWLINE_1:%.+]] = moore.fmt.literal "\0A"
  // CHECK:   [[MSG_1:%.+]] = moore.fmt.concat ([[FMT_1]], [[NEWLINE_1]])
  // CHECK:   moore.builtin.display [[MSG_1]]
  // CHECK: }

  // CHECK: moore.procedure initial {
  // CHECK: }
  initial begin
    $monitoro(x);
  end
  // $monitoro(x) should format as octal (same as $displayo(x))
  // CHECK: moore.procedure always_comb {
  // CHECK:   [[X_VAL_2:%.+]] = moore.read [[X]] : <i32>
  // CHECK:   [[FMT_2:%.+]] = moore.fmt.int octal [[X_VAL_2]], align right, pad zero : i32
  // CHECK:   [[NEWLINE_2:%.+]] = moore.fmt.literal "\0A"
  // CHECK:   [[MSG_2:%.+]] = moore.fmt.concat ([[FMT_2]], [[NEWLINE_2]])
  // CHECK:   moore.builtin.display [[MSG_2]]
  // CHECK: }

  // CHECK: moore.procedure initial {
  // CHECK: }
  initial begin
    $monitorh(x);
  end
  // $monitorh(x) should format as hex (same as $displayh(x))
  // CHECK: moore.procedure always_comb {
  // CHECK:   [[X_VAL_3:%.+]] = moore.read [[X]] : <i32>
  // CHECK:   [[FMT_3:%.+]] = moore.fmt.int hex_lower [[X_VAL_3]], align right, pad zero : i32
  // CHECK:   [[NEWLINE_3:%.+]] = moore.fmt.literal "\0A"
  // CHECK:   [[MSG_3:%.+]] = moore.fmt.concat ([[FMT_3]], [[NEWLINE_3]])
  // CHECK:   moore.builtin.display [[MSG_3]]
  // CHECK: }
endmodule

// CHECK-LABEL: moore.module @MonitorControl
module MonitorControl;
  int e;
  // CHECK: moore.procedure initial {
  // CHECK: [[ENABLED_REF_OFF:%.+]] = moore.get_global_variable @__monitor_enabled : <i1>
  // CHECK: [[FALSE:%.+]] = moore.constant 0 : i1
  // CHECK: moore.blocking_assign [[ENABLED_REF_OFF]], [[FALSE]] : i1
  // CHECK: [[ENABLED_REF_ON:%.+]] = moore.get_global_variable @__monitor_enabled : <i1>
  // CHECK: [[TRUE:%.+]] = moore.constant 1 : i1
  // CHECK: moore.blocking_assign [[ENABLED_REF_ON]], [[TRUE]] : i1
  // CHECK: moore.return
  // CHECK: }
  initial begin
    $monitor("e=%d", e);
    #10;
    $monitoroff;
    #10;
    $monitoron;
  end
  // Verify the monitor procedure is created after the initial block
  // CHECK: moore.procedure always_comb {
  // CHECK: moore.return
  // CHECK: }
endmodule

// CHECK-LABEL: moore.module @MultipleMonitors
module MultipleMonitors;
  // CHECK: [[F:%.+]] = moore.variable
  // CHECK: [[G:%.+]] = moore.variable
  int f, g;
  // CHECK: moore.procedure initial {
  // CHECK: moore.return
  // CHECK: }
  initial begin
    $monitor("f=%d", f);
    #10;
    $monitor("g=%d", g);
  end
  // First monitor procedure - monitors variable f
  // CHECK: moore.procedure always_comb {
  // CHECK:   [[LIT_F:%.+]] = moore.fmt.literal "f="
  // CHECK:   [[F_VAL:%.+]] = moore.read [[F]] : <i32>
  // CHECK:   [[FMT_F:%.+]] = moore.fmt.int decimal [[F_VAL]], align right, pad space signed : i32
  // CHECK:   [[NEWLINE_F:%.+]] = moore.fmt.literal "\0A"
  // CHECK:   [[MSG_F:%.+]] = moore.fmt.concat ([[LIT_F]], [[FMT_F]], [[NEWLINE_F]])
  // CHECK:   moore.builtin.display [[MSG_F]]
  // CHECK: }
  // Second monitor procedure - monitors variable g
  // CHECK: moore.procedure always_comb {
  // CHECK:   [[LIT_G:%.+]] = moore.fmt.literal "g="
  // CHECK:   [[G_VAL:%.+]] = moore.read [[G]] : <i32>
  // CHECK:   [[FMT_G:%.+]] = moore.fmt.int decimal [[G_VAL]], align right, pad space signed : i32
  // CHECK:   [[NEWLINE_G:%.+]] = moore.fmt.literal "\0A"
  // CHECK:   [[MSG_G:%.+]] = moore.fmt.concat ([[LIT_G]], [[FMT_G]], [[NEWLINE_G]])
  // CHECK:   moore.builtin.display [[MSG_G]]
  // CHECK: }
endmodule
