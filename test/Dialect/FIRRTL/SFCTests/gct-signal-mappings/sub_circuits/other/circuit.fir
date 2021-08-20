; RUN: true

circuit Sub :
  extmodule SubExtern :
    input clockIn : Clock
    output clockOut : Clock
    input someInput : UInt<42>
    output someOutput : UInt<42>

  module Sub :
    wire clock_source : Clock
    wire clock_sink : Clock
    wire data_source : UInt<42>
    wire data_sink : UInt<42>

    inst ext of SubExtern
    ext.clockIn <= clock_source
    ext.someInput <= data_source
    clock_sink <= ext.clockOut
    data_sink <= ext.someOutput

    ; CHECK: inst Sub_signal_mappings of Sub_signal_mappings
    ; CHECK: clock_source <= Sub_signal_mappings.clock_source
    ; CHECK: data_source <= Sub_signal_mappings.data_source
    ; CHECK: Sub_signal_mappings.clock_sink <= clock_sink
    ; CHECK: Sub_signal_mappings.data_sink <= data_sink

  ; CHECK: extmodule Sub_signal_mappings :
  ; CHECK:   output clock_source : Clock
  ; CHECK:   output data_source : UInt<42>
  ; CHECK:   input clock_sink : Clock
  ; CHECK:   input data_sink : UInt<42>
  ; CHECK:   defname = Sub_signal_mappings

; CHECK: module Sub_signal_mappings(
; CHECK:   output        clock_source,
; CHECK:   output [41:0] data_source,
; CHECK:   input         clock_sink,
; CHECK:   input  [41:0] data_sink,
; CHECK: );
; CHECK:   assign clock_source = Top.clock;
; CHECK:   assign data_source = Top.foo.dataOut_x_y_z;
; CHECK:   initial begin
; CHECK:     `ifndef VERILATOR
; CHECK:     force Top.foo.clock = clock_sink;
; CHECK:     force Top.foo.dataIn_a_b_c = data_sink;
; CHECK:     `endif VERILATOR
; CHECK:   end
; CHECK: endmodule
