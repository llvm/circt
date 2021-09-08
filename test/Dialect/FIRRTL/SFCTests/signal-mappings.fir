; RUN: firtool %s --annotation-file %S/signal-mappings-subCircuit.json --firrtl-grand-central --verilog | FileCheck %s

; Subcircuit:

circuit Sub :
  extmodule SubExtern :
    input clockIn : Clock
    output clockOut : Clock
    input someInput : { u: UInt<42>, v: UInt<9001>, w: UInt<1>[2] }
    output someOutput : { u: UInt<42>, v: UInt<9001>, w: UInt<1>[2] }

  module Sub :
    wire clock_source : Clock
    wire clock_sink : Clock
    wire data_source : { u: UInt<42>, v: UInt<9001>, w: UInt<1>[2] }
    wire data_sink : { u: UInt<42>, v: UInt<9001>, w: UInt<1>[2] }
    clock_source is invalid
    data_source is invalid

    inst ext of SubExtern
    ext.clockIn <= clock_source
    ext.someInput <= data_source
    clock_sink <= ext.clockOut
    data_sink <= ext.someOutput

; CHECK: module Sub_signal_mappings(
; CHECK:   input           clock_sink,
; CHECK:   input  [41:0]   data_sink_u,
; CHECK:   input  [9000:0] data_sink_v,
; CHECK:   input           data_sink_w_0, data_sink_w_1,
; CHECK:   output          clock_source,
; CHECK:   output [41:0]   data_source_u,
; CHECK:   output [9000:0] data_source_v,
; CHECK:   output          data_source_w_0, data_source_w_1);

; CHECK:   initial begin
; CHECK:     force Foo.clock = clock_sink;
; CHECK:     force Foo.dataIn_a_b_c = data_sink_u;
; CHECK:     force Foo.dataIn_d = data_sink_v;
; CHECK:     force Foo.dataIn_e = data_sink_w_0;
; CHECK:     force Foo.dataIn_e = data_sink_w_1;
; CHECK:   end
; CHECK:   assign clock_source = Top.clock;
; CHECK:   assign data_source_u = Foo.dataOut_x_y_z;
; CHECK:   assign data_source_v = Foo.dataOut_w;
; CHECK:   assign data_source_w_0 = Foo.dataOut_p;
; CHECK:   assign data_source_w_1 = Foo.dataOut_p;
; CHECK: endmodule


; Main circuit, for reference:

; circuit Top :
;   module Foo :
;     input clock : Clock
;     input reset : Reset
;     input dataIn : {a: {b: {c: UInt<42>}}, d: UInt<9001>}
;     output dataOut : {x: {y: {z: UInt<42>}}, w: UInt<9001>}
;
;     reg stuff : UInt<42>
;     reg stuff2 : UInt<9001>
;     stuff <= dataIn.a.b.c
;     stuff2 <= dataIn.d
;     dataOut.x.y.z <= stuff
;     dataOut.w <= stuff2
;
;   module Top :
;     input clock : Clock
;     input reset : UInt<1>
;
;     inst foo of Foo
;     foo.clock <= asClock(UInt<1>(0))
;     foo.reset <= reset
;     foo.dataIn is invalid
