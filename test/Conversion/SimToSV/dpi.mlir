// RUN: circt-opt --lower-sim-to-sv %s | FileCheck %s
// RUN: circt-opt --lower-sim-to-sv --lower-seq-to-sv -export-verilog %s | FileCheck %s --check-prefix=VERILOG

sim.func.dpi @dpi(out arg0: i1, in %arg1: i1, out arg2: i1)
// CHECK:       sv.func private @dpi(out arg0 : i1, in %arg1 : i1, out arg2 : i1)
// CHECK-NEXT:  emit.fragment @dpi_dpi_import_fragument {
// CHECK-NEXT:    sv.func.dpi.import @dpi
// CHECK-NEXT:  }

// VERILOG:      import "DPI-C" function void dpi( 
// VERILOG-NEXT:   output bit arg0,
// VERILOG-NEXT:   input  bit arg1,
// VERILOG-NEXT:   output bit arg2
// VERILOG-NEXT: );


// CHECK-LABEL: hw.module @dpi_call
// CHECK-SAME: {emit.fragments = [@dpi_dpi_import_fragument]} 
hw.module @dpi_call(in %clock : !seq.clock, in %enable : i1, in %in: i1,
          out o1: i1, out o2: i1, out o3: i1, out o4: i1, out o5: i1, out o6: i1, out o7: i1, out o8: i1) {
  
  %0, %1 = sim.func.dpi.call @dpi(%in) clock %clock enable %enable: (i1) -> (i1, i1)
  // CHECK: %[[CLK:.+]] = seq.from_clock %clock
  // CHECK-NEXT:  sv.always posedge %[[CLK]] {
  // CHECK-NEXT:    sv.if %enable {
  // CHECK-NEXT:      %[[RESULT:.+]]:2 = sv.func.call.procedural @dpi(%in) : (i1) -> (i1, i1)
  // CHECK-NEXT:      sv.passign %{{.+}}, %[[RESULT]]#0 : i1
  // CHECK-NEXT:      sv.passign %{{.+}}, %[[RESULT]]#1 : i1
  // CHECK-NEXT:    }
  // CHECK-NEXT:  }
  // VERILOG:     always @(posedge clock) begin
  // VERILOG-NEXT:   if (enable) begin
  // VERILOG-NEXT:     dpi([[TMP_RESULT_0:.+]], in, [[TMP_RESULT_1:.+]]);
  // VERILOG-NEXT:     [[RESULT_0:_.+]] <= [[TMP_RESULT_0]];
  // VERILOG-NEXT:     [[RESULT_1:_.+]] <= [[TMP_RESULT_1]];
  // VERILOG-NEXT:   end
  // VERILOG-NEXT: end 

  %2, %3 = sim.func.dpi.call @dpi(%in) clock %clock : (i1) -> (i1, i1)
  // CHECK: %[[CLK:.+]] = seq.from_clock %clock
  // CHECK-NEXT:  sv.always posedge %[[CLK]] {
  // CHECK-NEXT:    %[[RESULT:.+]]:2 = sv.func.call.procedural @dpi(%in) : (i1) -> (i1, i1)
  // CHECK-NEXT:    sv.passign %{{.+}}, %[[RESULT]]#0 : i1
  // CHECK-NEXT:    sv.passign %{{.+}}, %[[RESULT]]#1 : i1
  // CHECK-NEXT:  }
  // VERILOG:     always @(posedge clock) begin
  // VERILOG-NEXT:   dpi([[TMP_RESULT_0:.+]], in, [[TMP_RESULT_1:.+]]);
  // VERILOG-NEXT:   [[RESULT_2:_.+]] <= [[TMP_RESULT_0]];
  // VERILOG-NEXT:   [[RESULT_3:_.+]] <= [[TMP_RESULT_1]];
  // VERILOG-NEXT: end 

  %4, %5 = sim.func.dpi.call @dpi(%in) enable %enable : (i1) -> (i1, i1)
  // CHECK:       sv.alwayscomb {
  // CHECK-NEXT:    sv.if %enable {
  // CHECK-NEXT:      %[[RESULT:.+]]:2 = sv.func.call.procedural @dpi(%in) : (i1) -> (i1, i1)
  // CHECK-NEXT:      sv.bpassign %{{.+}}, %[[RESULT]]#0 : i1
  // CHECK-NEXT:      sv.bpassign %{{.+}}, %[[RESULT]]#1 : i1
  // CHECK-NEXT:    } else {
  // CHECK-NEXT:      %[[X:.+]] = sv.constantX
  // CHECK-NEXT:      sv.bpassign %{{.+}}, %[[X]] : i1
  // CHECK-NEXT:      %[[X:.+]] = sv.constantX
  // CHECK-NEXT:      sv.bpassign %{{.+}}, %[[X]] : i1
  // CHECK-NEXT:    }
  // CHECK-NEXT:  }
  // VERILOG:      always_comb begin
  // VERILOG-NEXT:   if (enable) begin
  // VERILOG-NEXT:     dpi([[RESULT_4:.+]], in, [[RESULT_5:.+]]);
  // VERILOG-NEXT:   end
  // VERILOG-NEXT:   else begin
  // VERILOG-NEXT:     [[RESULT_4]] = 1'bx;
  // VERILOG-NEXT:     [[RESULT_5]] = 1'bx;
  // VERILOG-NEXT:   end
  // VERILOG-NEXT: end

  %6, %7 = sim.func.dpi.call @dpi(%in) : (i1) -> (i1, i1)
  // CHECK:       sv.alwayscomb {
  // CHECK-NEXT:    %[[RESULT:.+]]:2 = sv.func.call.procedural @dpi(%in) : (i1) -> (i1, i1)
  // CHECK-NEXT:    sv.bpassign %{{.+}}, %[[RESULT]]#0 : i1
  // CHECK-NEXT:    sv.bpassign %{{.+}}, %[[RESULT]]#1 : i1
  // CHECK-NEXT:  }
  // VERILOG:      always_comb begin
  // VERILOG-NEXT:   dpi([[RESULT_6:.+]], in, [[RESULT_7:.+]]);
  // VERILOG-NEXT: end 

  // VERILOG: assign o1 = [[RESULT_0]];
  // VERILOG-NEXT: assign o2 = [[RESULT_1]];
  // VERILOG-NEXT: assign o3 = [[RESULT_2]];
  // VERILOG-NEXT: assign o4 = [[RESULT_3]];
  // VERILOG-NEXT: assign o5 = [[RESULT_4]]; 
  // VERILOG-NEXT: assign o6 = [[RESULT_5]]; 
  // VERILOG-NEXT: assign o7 = [[RESULT_6]]; 
  // VERILOG-NEXT: assign o8 = [[RESULT_7]]; 
  hw.output %0, %1, %2, %3, %4, %5, %6, %7: i1, i1, i1, i1, i1, i1, i1, i1
}

sim.func.dpi private @increment_counter(in %in_0 : i64, out out_0 : i32)
sim.func.dpi private @create_counter(out out_0 : i64)
// CHECK-LABEL: hw.module @Issue7191
// Check lowering successes.
hw.module @Issue7191(out result : i32) {
  // CHECK: call.procedural @create_counter
  // CHECK: call.procedural @increment_counter

  %0 = sim.func.dpi.call @create_counter() : () -> i64
  %1 = sim.func.dpi.call @increment_counter(%0) : (i64) -> i32
  hw.output %1 : i32
}
