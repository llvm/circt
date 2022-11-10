; RUN: firtool --split-input-file %s | FileCheck %s

; A number of tests use an invalidated wire with a DontTouchAnnotation to block
; constant sinking into the XMR.  This is to ensure that the test can check a
; valid XMR as opposed to "assign foo = 1'h0".

; This test was extracted from:
; - github.com/sifive/$internal:
;   - src/test/scala/grandcentral/DataTapsTest.scala
circuit Top : %[[
  {
    "class":"firrtl.transforms.NoDedupAnnotation",
    "target":"~Top|DataTap_2"
  },
  {
    "class":"sifive.enterprise.grandcentral.DataTapsAnnotation",
    "blackBox":"~Top|DataTap_2",
    "keys":[
      {
        "class":"sifive.enterprise.grandcentral.DataTapModuleSignalKey",
        "module":"~Top|BlackBox",
        "internalPath":"foo.bar.in",
        "portName":"~Top|DataTap_2>_0"
      },
      {
        "class":"sifive.enterprise.grandcentral.DataTapModuleSignalKey",
        "module":"~Top|BlackBox",
        "internalPath":"foo.bar.out",
        "portName":"~Top|DataTap_2>_1"
      }
    ]
  },
  {
    "class": "firrtl.transforms.DontTouchAnnotation",
    "target": "~Top|Child>always"
  },
  {
    "class":"sifive.enterprise.grandcentral.MemTapAnnotation",
    "source": "~Top|Child>always",
    "taps":["~Top|MemTap_2>_0", "~Top|MemTap_2>_1"]
  }
]]
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
    always.w.addr <= io.in
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
    node _child_io_in_T = and(io.in, in)
    unsigned.io.in <= _child_io_in_T
    node _io_out_T = and(unsigned.io.out, out)
    io.out <= _io_out_T
    inst MemTap_2 of MemTap_2

    in <= or(DataTap_2._0, MemTap_2._0)
    out <= or(DataTap_2._1, MemTap_2._1)

; CHECK: module DataTap_2_impl_0(
; CHECK:   output _1,
; CHECK:          _0);
; CHECK:   assign _1 = Top.unsigned_0.signed_0.localparam_0.foo.bar.out;
; CHECK:   assign _0 = Top.unsigned_0.signed_0.localparam_0.foo.bar.in;
; CHECK: endmodule

; CHECK: module MemTap_2_impl_0(
; CHECK:   output  _1,
; CHECK:           _0);
; CHECK:   assign _1 = Top.unsigned_0.signed_0.always_ext.Memory[1];
; CHECK:   assign _0 = Top.unsigned_0.signed_0.always_ext.Memory[0];
; CHECK: endmodule

; // -----

circuit Top : %[[
  {
    "class": "sifive.enterprise.grandcentral.DataTapsAnnotation",
    "blackBox": "~Top|DataTap",
    "keys": [
      {
        "class": "sifive.enterprise.grandcentral.ReferenceDataTapKey",
        "source": "~Top|Top/foo:Foo>f",
        "portName": "~Top|DataTap>b"
      },
      {
        "class": "sifive.enterprise.grandcentral.ReferenceDataTapKey",
        "source": "~Top|Top/foo:Foo>g",
        "portName": "~Top|DataTap>c"
      },
      {
        "class": "sifive.enterprise.grandcentral.ReferenceDataTapKey",
        "source": "~Top|Top>k",
        "portName": "~Top|DataTap>d"
      }
    ]
  },
  {
    "class": "firrtl.transforms.DontTouchAnnotation",
    "target": "~Top|Foo>inv"
  },
  {
    "class": "firrtl.transforms.DontTouchAnnotation",
    "target": "~Top|Top>inv"
  }
]]
  extmodule DataTap :
    output b : UInt<1>
    output c : UInt<1>
    output d : UInt<1>
    defname = DataTap

  module Foo :
    output g : UInt<1>
    wire inv: UInt<1>
    inv is invalid
    wire f : UInt<1>
    g <= inv
    f <= inv

  module Top:
    output io : { b : UInt<1>, c : UInt<1>, d: UInt<1> }

    inst foo of Foo
    inst bar of Foo
    inst dataTap of DataTap
    wire inv: UInt<1>
    inv is invalid
    wire k : UInt<1>
    k <= inv
    io.b <= dataTap.b
    io.c <= dataTap.c
    io.d <= dataTap.d

; CHECK: module DataTap_impl_0(
; CHECK:   output b,
; CHECK:          c,
; CHECK:          d
; CHECK: );
; CHECK:   assign b = Top.foo.f;
; CHECK:   assign c = Top.foo.g;
; CHECK:   assign d = Top.k;
; CHECK: endmodule

; // -----

circuit TestHarness : %[[
  {
    "class":"sifive.enterprise.grandcentral.DataTapsAnnotation",
    "blackBox":"~TestHarness|DataTap",
    "keys":[
      {
        "class":"sifive.enterprise.grandcentral.ReferenceDataTapKey",
        "source":"~TestHarness|TestHarness/system:Top/test:Test>signal",
        "portName":"~TestHarness|DataTap>_0"
      }
    ]
  },
  {
    "class": "firrtl.transforms.DontTouchAnnotation",
    "target": "~TestHarness|Test>inv"
  }
]]
  module TestHarness :
    output io : UInt<1>
    inst system of Top
    io <= system.io
  module Top :
    output io : UInt<1>
    inst test of Test
    inst Companion of Companion
    io <= Companion.io
  extmodule DataTap :
    output _0 : UInt<8>
    defname = DataTap
  module Companion :
    output io : UInt<1>
    inst DataTap of DataTap
    io <= DataTap._0
  module Test :
    wire inv: UInt<1>
    inv is invalid
    wire signal : UInt<8>
    signal <= inv

; CHECK:     module DataTap
; CHECK-NOT: endmodule
; CHECK:       assign _0 = Top.test.signal;

; // -----

circuit Top : %[[
  {
    "class":"sifive.enterprise.grandcentral.DataTapsAnnotation",
    "blackBox":"~Top|DataTap_Submodule",
    "keys":[
      {
        "class":"sifive.enterprise.grandcentral.ReferenceDataTapKey",
        "source":"~Top|Top/dut:DUT/submodule:Submodule>wire_Submodule",
        "portName":"~Top|DataTap_Submodule>_0"
      },
      {
        "class":"sifive.enterprise.grandcentral.ReferenceDataTapKey",
        "source":"~Top|Top/dut:DUT>wire_DUT",
        "portName":"~Top|DataTap_Submodule>_1"
      },
      {
        "class":"sifive.enterprise.grandcentral.ReferenceDataTapKey",
        "source":"~Top|Top>wire_Top",
        "portName":"~Top|DataTap_Submodule>_2"
      },
      {
        "class":"sifive.enterprise.grandcentral.ReferenceDataTapKey",
        "source":"~Top|Top/dut:DUT/submodule:Submodule>port_Submodule",
        "portName":"~Top|DataTap_Submodule>_3"
      },
      {
        "class":"sifive.enterprise.grandcentral.ReferenceDataTapKey",
        "source":"~Top|Top/dut:DUT>port_DUT",
        "portName":"~Top|DataTap_Submodule>_4"
      },
      {
        "class":"sifive.enterprise.grandcentral.ReferenceDataTapKey",
        "source":"~Top|Top>port_Top",
        "portName":"~Top|DataTap_Submodule>_5"
      }
    ]
  },
  {
    "class":"sifive.enterprise.grandcentral.DataTapsAnnotation",
    "blackBox":"~Top|DataTap_DUT",
    "keys":[
      {
        "class":"sifive.enterprise.grandcentral.ReferenceDataTapKey",
        "source":"~Top|Top/dut:DUT/submodule:Submodule>wire_Submodule",
        "portName":"~Top|DataTap_DUT>_0"
      },
      {
        "class":"sifive.enterprise.grandcentral.ReferenceDataTapKey",
        "source":"~Top|Top/dut:DUT>wire_DUT",
        "portName":"~Top|DataTap_DUT>_1"
      },
      {
        "class":"sifive.enterprise.grandcentral.ReferenceDataTapKey",
        "source":"~Top|Top>wire_Top",
        "portName":"~Top|DataTap_DUT>_2"
      },
      {
        "class":"sifive.enterprise.grandcentral.ReferenceDataTapKey",
        "source":"~Top|Top/dut:DUT/submodule:Submodule>port_Submodule",
        "portName":"~Top|DataTap_DUT>_3"
      },
      {
        "class":"sifive.enterprise.grandcentral.ReferenceDataTapKey",
        "source":"~Top|Top/dut:DUT>port_DUT",
        "portName":"~Top|DataTap_DUT>_4"
      },
      {
        "class":"sifive.enterprise.grandcentral.ReferenceDataTapKey",
        "source":"~Top|Top>port_Top",
        "portName":"~Top|DataTap_DUT>_5"
      }
    ]
  },
  {
    "class":"sifive.enterprise.grandcentral.DataTapsAnnotation",
    "blackBox":"~Top|DataTap_Top",
    "keys":[
      {
        "class":"sifive.enterprise.grandcentral.ReferenceDataTapKey",
        "source":"~Top|Top/dut:DUT/submodule:Submodule>wire_Submodule",
        "portName":"~Top|DataTap_Top>_0"
      },
      {
        "class":"sifive.enterprise.grandcentral.ReferenceDataTapKey",
        "source":"~Top|Top/dut:DUT>wire_DUT",
        "portName":"~Top|DataTap_Top>_1"
      },
      {
        "class":"sifive.enterprise.grandcentral.ReferenceDataTapKey",
        "source":"~Top|Top>wire_Top",
        "portName":"~Top|DataTap_Top>_2"
      },
      {
        "class":"sifive.enterprise.grandcentral.ReferenceDataTapKey",
        "source":"~Top|Top/dut:DUT/submodule:Submodule>port_Submodule",
        "portName":"~Top|DataTap_Top>_3"
      },
      {
        "class":"sifive.enterprise.grandcentral.ReferenceDataTapKey",
        "source":"~Top|Top/dut:DUT>port_DUT",
        "portName":"~Top|DataTap_Top>_4"
      },
      {
        "class":"sifive.enterprise.grandcentral.ReferenceDataTapKey",
        "source":"~Top|Top>port_Top",
        "portName":"~Top|DataTap_Top>_5"
      }
    ]
  },
  {
    "class": "firrtl.transforms.DontTouchAnnotation",
    "target": "~Top|Submodule>inv"
  },
  {
    "class": "firrtl.transforms.DontTouchAnnotation",
    "target": "~Top|DUT>inv"
  },
  {
    "class": "firrtl.transforms.DontTouchAnnotation",
    "target": "~Top|Top>inv"
  }
]]
  extmodule DataTap_Submodule :
    output _0 : UInt<1>
    output _1 : UInt<1>
    output _2 : UInt<1>
    output _3 : UInt<1>
    output _4 : UInt<1>
    output _5 : UInt<1>


  extmodule DataTap_DUT :
    output _0 : UInt<1>
    output _1 : UInt<1>
    output _2 : UInt<1>
    output _3 : UInt<1>
    output _4 : UInt<1>
    output _5 : UInt<1>

  extmodule DataTap_Top :
    output _0 : UInt<1>
    output _1 : UInt<1>
    output _2 : UInt<1>
    output _3 : UInt<1>
    output _4 : UInt<1>
    output _5 : UInt<1>

  module Submodule :
    output port_Submodule: UInt<1>
    port_Submodule is invalid

    wire inv: UInt<1>
    inv is invalid

    wire wire_Submodule: UInt<1>
    wire_Submodule <= inv

    inst tap of DataTap_Submodule
    port_Submodule <= or(tap._0, or(tap._1, or(tap._2, or(tap._3, or(tap._4, tap._5)))))

  module DUT :
    output port_DUT: UInt<1>
    port_DUT is invalid

    wire inv: UInt<1>
    inv is invalid

    wire wire_DUT: UInt<1>
    wire_DUT <= inv

    inst submodule of Submodule

    inst tap of DataTap_DUT
    port_DUT <= or(tap._0, or(tap._1, or(tap._2, or(tap._3, or(tap._4, tap._5)))))

  module Top :
    output port_Top : UInt<1>
    port_Top is invalid

    wire inv: UInt<1>
    inv is invalid

    wire wire_Top: UInt<1>
    wire_Top <= inv

    inst dut of DUT
    inst tap of DataTap_Top
    port_Top <= or(tap._0, or(tap._1, or(tap._2, or(tap._3, or(tap._4, tap._5)))))

    ; CHECK:      module DataTap_Submodule
    ; CHECK:      assign _0 = Submodule.wire_Submodule
    ; CHECK-NEXT: assign _1 = DUT.wire_DUT
    ; CHECK-NEXT: assign _2 = Top.wire_Top
    ; CHECK:      assign _3 = Submodule.port_Submodule
    ; CHECK-NEXT: assign _4 = DUT.port_DUT
    ; CHECK-NEXT: assign _5 = Top.port_Top

    ; CHECK: module DataTap_DUT
    ; CHECK:      assign _0 = DUT.submodule.wire_Submodule
    ; CHECK-NEXT: assign _1 = DUT.wire_DUT
    ; CHECK-NEXT: assign _2 = Top.wire_Top
    ; CHECK:      assign _3 = DUT.submodule.port_Submodule
    ; CHECK-NEXT: assign _4 = DUT.port_DUT
    ; CHECK-NEXT: assign _5 = Top.port_Top

    ; CHECK: module DataTap_Top
    ; CHECK:      assign _0 = Top.dut.submodule.wire_Submodule
    ; CHECK-NEXT: assign _1 = Top.dut.wire_DUT
    ; CHECK-NEXT: assign _2 = Top.wire_Top
    ; CHECK:      assign _3 = Top.dut.submodule.port_Submodule
    ; CHECK-NEXT: assign _4 = Top.dut.port_DUT
    ; CHECK-NEXT: assign _5 = Top.port_Top

; // -----

circuit ConstantSinking : %[[
  {
    "class": "sifive.enterprise.grandcentral.DataTapsAnnotation",
    "blackBox": "~ConstantSinking|DataTap",
    "keys": [
      {
        "class": "sifive.enterprise.grandcentral.ReferenceDataTapKey",
        "source": "~ConstantSinking|ConstantSinking>w",
        "portName": "~ConstantSinking|DataTap>a"
      }
    ]
  }
]]
  extmodule DataTap :
    output a : UInt<1>

  module ConstantSinking:
    inst dataTap of DataTap
    node w = UInt<1>(1)

; // -----

circuit Top : %[[
  {
    "class":"firrtl.transforms.DontTouchAnnotation",
    "target":"~Top|Child>sum"
  },
  {
    "class":"sifive.enterprise.grandcentral.DataTapsAnnotation",
    "blackBox":"~Top|DataTap",
    "keys":[
      {
        "class":"sifive.enterprise.grandcentral.DataTapModuleSignalKey",
        "module":"~Top|Child",
        "internalPath":"sum",
        "portName":"~Top|DataTap>_0"
      },
      {
        "class":"sifive.enterprise.grandcentral.DataTapModuleSignalKey",
        "module":"~Top|Child",
        "internalPath":"not.a.real.path",
        "portName":"~Top|DataTap>_1"
      }
    ]
  }
]]
  module Child :
    output io : { flip in : UInt<8>, out : UInt<8>}

    node sum = tail(add(io.in, UInt<1>(1)), 1)

    io.out <= sum

  extmodule DataTap :
    output _0 : UInt<8>
    output _1 : UInt<8>
    defname = DataTap

  module Top :
    output io : { flip in : UInt<8>, out : UInt<8>}
    output taps : UInt<8>[2]

    inst child of Child
    io <= child.io

    inst dt of DataTap
    taps[0] <= dt._0
    taps[1] <= dt._1

; CHECK: module DataTap_impl_0(
; CHECK:   output [7:0] _0,
; CHECK:                _1);
; CHECK:   assign _0 = Top.child.sum;
; CHECK:   assign _1 = Top.child.not.a.real.path;
; CHECK: endmodule
