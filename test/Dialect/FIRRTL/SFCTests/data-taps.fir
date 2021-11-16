; RUN: firtool --firrtl-grand-central --verilog --annotation-file %S/data-taps.anno.json %s | FileCheck %s
; Tests extracted from:
; - github.com/sifive/$internal:
;   - src/test/scala/grandcentral/DataTapsTest.scala

circuit Top :
  extmodule BlackBox :
    input in : UInt<1>
    output out : UInt<1>
    defname = BlackBox

  module Child :
    input clock : Clock
    input reset : Reset
    output io : { flip in : UInt<1>, out : UInt<1>}

    inst localparam of BlackBox
    localparam.out is invalid
    localparam.in is invalid
    localparam.in <= io.in
    io.out <= localparam.out

    mem always :
      data-type => UInt<1>
      depth => 2
      read-latency => 0
      write-latency => 1
      writer => w
      read-under-write => undefined
    always.w.clk <= clock
    always.w.en <= UInt<1>(1)
    always.w.addr <= UInt<1>(0)
    always.w.data <= io.in
    always.w.mask <= UInt<1>(1)

  module ChildWrapper :
    input clock : Clock
    input reset : Reset
    output io : { flip in : UInt<1>, out : UInt<1>}

    inst signed of Child
    signed.clock <= clock
    signed.reset <= reset
    signed.io.in <= io.in
    io.out <= signed.io.out

  extmodule DataTap_2 :
    output _1 : UInt<1>
    output _0 : UInt<1>
    defname = DataTap_2

  extmodule MemTap_2 :
    output _1 : UInt<1>
    output _0 : UInt<1>
    defname = MemTap_2

  module Top :
    input clock : Clock
    input reset : UInt<1>
    output io : { flip in : UInt<1>, out : UInt<1>}

    inst unsigned of ChildWrapper
    unsigned.clock <= clock
    unsigned.reset <= reset
    wire in : UInt<1>
    wire out : UInt<1>
    inst DataTap_2 of DataTap_2
    DataTap_2._0 is invalid
    DataTap_2._1 is invalid
    in <= DataTap_2._0
    out <= DataTap_2._1
    node _child_io_in_T = and(io.in, in)
    unsigned.io.in <= _child_io_in_T
    node _io_out_T = and(unsigned.io.out, out)
    io.out <= _io_out_T
    inst MemTap_2 of MemTap_2

; CHECK: module DataTap_2_impl_0(
; CHECK:   output  _1, _0
; CHECK: );
; CHECK:   assign _1 = Top.unsigned_0.signed_0.localparam_0.foo.bar.out;
; CHECK:   assign _0 = Top.unsigned_0.signed_0.localparam_0.foo.bar.in;
; CHECK: endmodule

; CHECK: module MemTap_2_impl_0(
; CHECK:   output  _1, _0
; CHECK: );
; CHECK:   assign _1 = Top.unsigned_0.signed_0.always_1.Memory[1];
; CHECK:   assign _0 = Top.unsigned_0.signed_0.always_1.Memory[0];
; CHECK: endmodule
