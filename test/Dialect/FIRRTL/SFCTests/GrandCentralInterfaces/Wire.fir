; RUN: firtool --firrtl-grand-central --verilog --annotation-file %S/Wire.anno.json --annotation-file %S/Extract.anno.json %s | FileCheck %s --check-prefixes CHECK,EXTRACT
; RUN: firtool --firrtl-grand-central --verilog --annotation-file %S/Wire.anno.json %s | FileCheck %s --check-prefixes CHECK,NOEXTRACT
; RUN: firtool --firrtl-grand-central --verilog --annotation-file %S/Wire.anno.json --annotation-file %S/Extract.anno.json %s --annotation-file %S/YAML.anno.json | FileCheck %s --check-prefixes YAML

circuit Top :
  module Submodule :
    input clock : Clock
    input reset : Reset
    input in : { uint : UInt<1>, vec : UInt<1>[2], multivec : UInt<1>[3][2], vecOfBundle : { uint : UInt<4>, sint : SInt<2>}[2], otherOther : { other : { uint : UInt<4>, sint : SInt<2>}}}
    output out : { uint : UInt<1>, vec : UInt<1>[2], multivec : UInt<1>[3][2], vecOfBundle : { uint : UInt<4>, sint : SInt<2>}[2], otherOther : { other : { uint : UInt<4>, sint : SInt<2>}}}

    wire w : { uint : UInt<1>, vec : UInt<1>[2], multivec : UInt<1>[3][2], vecOfBundle : { uint : UInt<4>, sint : SInt<2>}[2], otherOther : { other : { uint : UInt<4>, sint : SInt<2>}}}
    w.otherOther.other.sint <= in.otherOther.other.sint
    w.otherOther.other.uint <= in.otherOther.other.uint
    w.vecOfBundle[0].sint <= in.vecOfBundle[0].sint
    w.vecOfBundle[0].uint <= in.vecOfBundle[0].uint
    w.vecOfBundle[1].sint <= in.vecOfBundle[1].sint
    w.vecOfBundle[1].uint <= in.vecOfBundle[1].uint
    w.vec[0] <= in.vec[0]
    w.vec[1] <= in.vec[1]
    w.multivec[0][0] <= in.multivec[0][0]
    w.multivec[0][1] <= in.multivec[0][1]
    w.multivec[0][2] <= in.multivec[0][2]
    w.multivec[1][0] <= in.multivec[1][0]
    w.multivec[1][1] <= in.multivec[1][1]
    w.multivec[1][2] <= in.multivec[1][2]
    w.uint <= in.uint
    out.otherOther.other.sint <= w.otherOther.other.sint
    out.otherOther.other.uint <= w.otherOther.other.uint
    out.vecOfBundle[0].sint <= w.vecOfBundle[0].sint
    out.vecOfBundle[0].uint <= w.vecOfBundle[0].uint
    out.vecOfBundle[1].sint <= w.vecOfBundle[1].sint
    out.vecOfBundle[1].uint <= w.vecOfBundle[1].uint
    out.vec[0] <= w.vec[0]
    out.vec[1] <= w.vec[1]
    out.multivec[0][0] <= w.multivec[0][0]
    out.multivec[0][1] <= w.multivec[0][1]
    out.multivec[0][2] <= w.multivec[0][2]
    out.multivec[1][0] <= w.multivec[1][0]
    out.multivec[1][1] <= w.multivec[1][1]
    out.multivec[1][2] <= w.multivec[1][2]
    out.uint <= w.uint

  module MyView_companion :
    output io : { }

    wire _WIRE : UInt<1>
    _WIRE <= UInt<1>("h0")

  module DUT :
    input clock : Clock
    input reset : Reset
    input in : { uint : UInt<1>, vec : UInt<1>[2], multivec : UInt<1>[3][2], vecOfBundle : { uint : UInt<4>, sint : SInt<2>}[2], otherOther : { other : { uint : UInt<4>, sint : SInt<2>}}}
    output out : { uint : UInt<1>, vec : UInt<1>[2], multivec : UInt<1>[3][2], vecOfBundle : { uint : UInt<4>, sint : SInt<2>}[2], otherOther : { other : { uint : UInt<4>, sint : SInt<2>}}}

    wire w : { uint : UInt<1>, vec : UInt<1>[2], multivec : UInt<1>[3][2], vecOfBundle : { uint : UInt<4>, sint : SInt<2>}[2], otherOther : { other : { uint : UInt<4>, sint : SInt<2>}}}
    inst submodule of Submodule
    submodule.clock <= clock
    submodule.reset <= reset
    w.otherOther.other.sint <= in.otherOther.other.sint
    w.otherOther.other.uint <= in.otherOther.other.uint
    w.vecOfBundle[0].sint <= in.vecOfBundle[0].sint
    w.vecOfBundle[0].uint <= in.vecOfBundle[0].uint
    w.vecOfBundle[1].sint <= in.vecOfBundle[1].sint
    w.vecOfBundle[1].uint <= in.vecOfBundle[1].uint
    w.vec[0] <= in.vec[0]
    w.vec[1] <= in.vec[1]
    w.multivec[0][0] <= in.multivec[0][0]
    w.multivec[0][1] <= in.multivec[0][1]
    w.multivec[0][2] <= in.multivec[0][2]
    w.multivec[1][0] <= in.multivec[1][0]
    w.multivec[1][1] <= in.multivec[1][1]
    w.multivec[1][2] <= in.multivec[1][2]
    w.uint <= in.uint
    submodule.in.otherOther.other.sint <= w.otherOther.other.sint
    submodule.in.otherOther.other.uint <= w.otherOther.other.uint
    submodule.in.vecOfBundle[0].sint <= w.vecOfBundle[0].sint
    submodule.in.vecOfBundle[0].uint <= w.vecOfBundle[0].uint
    submodule.in.vecOfBundle[1].sint <= w.vecOfBundle[1].sint
    submodule.in.vecOfBundle[1].uint <= w.vecOfBundle[1].uint
    submodule.in.vec[0] <= w.vec[0]
    submodule.in.vec[1] <= w.vec[1]
    submodule.in.multivec[0][0] <= w.multivec[0][0]
    submodule.in.multivec[0][1] <= w.multivec[0][1]
    submodule.in.multivec[0][2] <= w.multivec[0][2]
    submodule.in.multivec[1][0] <= w.multivec[1][0]
    submodule.in.multivec[1][1] <= w.multivec[1][1]
    submodule.in.multivec[1][2] <= w.multivec[1][2]
    submodule.in.uint <= w.uint
    out.otherOther.other.sint <= submodule.out.otherOther.other.sint
    out.otherOther.other.uint <= submodule.out.otherOther.other.uint
    out.vecOfBundle[0].sint <= submodule.out.vecOfBundle[0].sint
    out.vecOfBundle[0].uint <= submodule.out.vecOfBundle[0].uint
    out.vecOfBundle[1].sint <= submodule.out.vecOfBundle[1].sint
    out.vecOfBundle[1].uint <= submodule.out.vecOfBundle[1].uint
    out.vec[0] <= submodule.out.vec[0]
    out.vec[1] <= submodule.out.vec[1]
    out.multivec[0][0] <= submodule.out.multivec[0][0]
    out.multivec[0][1] <= submodule.out.multivec[0][1]
    out.multivec[0][2] <= submodule.out.multivec[0][2]
    out.multivec[1][0] <= submodule.out.multivec[1][0]
    out.multivec[1][1] <= submodule.out.multivec[1][1]
    out.multivec[1][2] <= submodule.out.multivec[1][2]
    out.uint <= submodule.out.uint
    inst MyView_companion of MyView_companion

  module Top :
    input clock : Clock
    input reset : UInt<1>
    input in : { uint : UInt<1>, vec : UInt<1>[2], multivec : UInt<1>[3][2], vecOfBundle : { uint : UInt<4>, sint : SInt<2>}[2], otherOther : { other : { uint : UInt<4>, sint : SInt<2>}}}
    output out : { uint : UInt<1>, vec : UInt<1>[2], multivec : UInt<1>[3][2], vecOfBundle : { uint : UInt<4>, sint : SInt<2>}[2], otherOther : { other : { uint : UInt<4>, sint : SInt<2>}}}

    inst dut of DUT
    dut.clock <= clock
    dut.reset <= reset
    dut.in.otherOther.other.sint <= in.otherOther.other.sint
    dut.in.otherOther.other.uint <= in.otherOther.other.uint
    dut.in.vecOfBundle[0].sint <= in.vecOfBundle[0].sint
    dut.in.vecOfBundle[0].uint <= in.vecOfBundle[0].uint
    dut.in.vecOfBundle[1].sint <= in.vecOfBundle[1].sint
    dut.in.vecOfBundle[1].uint <= in.vecOfBundle[1].uint
    dut.in.vec[0] <= in.vec[0]
    dut.in.vec[1] <= in.vec[1]
    dut.in.multivec[0][0] <= in.multivec[0][0]
    dut.in.multivec[0][1] <= in.multivec[0][1]
    dut.in.multivec[0][2] <= in.multivec[0][2]
    dut.in.multivec[1][0] <= in.multivec[1][0]
    dut.in.multivec[1][1] <= in.multivec[1][1]
    dut.in.multivec[1][2] <= in.multivec[1][2]
    dut.in.uint <= in.uint
    out.otherOther.other.sint <= dut.out.otherOther.other.sint
    out.otherOther.other.uint <= dut.out.otherOther.other.uint
    out.vecOfBundle[0].sint <= dut.out.vecOfBundle[0].sint
    out.vecOfBundle[0].uint <= dut.out.vecOfBundle[0].uint
    out.vecOfBundle[1].sint <= dut.out.vecOfBundle[1].sint
    out.vecOfBundle[1].uint <= dut.out.vecOfBundle[1].uint
    out.vec[0] <= dut.out.vec[0]
    out.vec[1] <= dut.out.vec[1]
    out.multivec[0][0] <= dut.out.multivec[0][0]
    out.multivec[0][1] <= dut.out.multivec[0][1]
    out.multivec[0][2] <= dut.out.multivec[0][2]
    out.multivec[1][0] <= dut.out.multivec[1][0]
    out.multivec[1][1] <= dut.out.multivec[1][1]
    out.multivec[1][2] <= dut.out.multivec[1][2]
    out.uint <= dut.out.uint

    ; NOEXTRACT:      module MyView_companion();
    ; NOEXTRACT-NEXT:   MyView_mapping MyView_mapping{{ *}}();
    ; NOEXTRACT-NEXT: endmodule

    ; CHECK:          module DUT
    ; CHECK-NOT:      endmodule
    ; EXTRACT:          // MyInterface MyView();
    ; NOEXTRACT:        {{^ *}}MyInterface MyView();
    ; EXTRACT:          /* This instance is elsewhere emitted as a bind statement.
    ; EXTRACT-NEXT:        MyView_companion MyView_companion
    ; EXTRACT-NEXT:     */
    ; NOEXTRACT:        {{^ *}}MyView_companion MyView_companion

    ; EXTRACT:        FILE "Wire/firrtl/bindings.sv"
    ; EXTRACT-NOT:    FILE
    ; EXTRACT:        bind DUT MyInterface MyView{{ *}}(.*);
    ; EXTRACT:        bind DUT MyView_companion MyView_companion{{ *}}();
    ; NOEXTRACT-NOT:  FILE "Wire/firrtl/bindings.sv"

    ; EXTRACT:        FILE "Wire/firrtl/gct/MyView_companion.sv"
    ; NOEXTRACT-NOT:  FILE {{.*}}/MyView_companion.sv
    ; EXTRACT:        module MyView_companion();
    ; EXTRACT-NEXT:     MyView_mapping MyView_mapping{{ *}}();
    ; EXTRACT-NEXT:   endmodule

    ; EXTRACT:        FILE "Wire/firrtl/gct/MyView_mapping.sv"
    ; NOEXTRACT-NOT:  FILE {{.*}}/MyView_mapping.sv
    ; CHECK:          module MyView_mapping();
    ; CHECK-NEXT:       assign MyView.uint = Top.dut.w_uint;
    ; CHECK-NEXT:       assign MyView.vec[0] = Top.dut.w_vec_0;
    ; CHECK-NEXT:       assign MyView.vec[1] = Top.dut.w_vec_1;
    ; CHECK-NEXT:       assign MyView.multivec[0][0] = Top.dut.w_multivec_0_0;
    ; CHECK-NEXT:       assign MyView.multivec[0][1] = Top.dut.w_multivec_0_1;
    ; CHECK-NEXT:       assign MyView.multivec[0][2] = Top.dut.w_multivec_0_2;
    ; CHECK-NEXT:       assign MyView.multivec[1][0] = Top.dut.w_multivec_1_0;
    ; CHECK-NEXT:       assign MyView.multivec[1][1] = Top.dut.w_multivec_1_1;
    ; CHECK-NEXT:       assign MyView.multivec[1][2] = Top.dut.w_multivec_1_2;
    ; CHECK-NEXT:       assign MyView.vecOfBundle[0].sint = Top.dut.w_vecOfBundle_0_sint;
    ; CHECK-NEXT:       assign MyView.vecOfBundle[0].uint = Top.dut.w_vecOfBundle_0_uint;
    ; CHECK-NEXT:       assign MyView.vecOfBundle[1].sint = Top.dut.w_vecOfBundle_1_sint;
    ; CHECK-NEXT:       assign MyView.vecOfBundle[1].uint = Top.dut.w_vecOfBundle_1_uint;
    ; CHECK-NEXT:       assign MyView.otherOther.other.sint = Top.dut.w_otherOther_other_sint;
    ; CHECK-NEXT:       assign MyView.otherOther.other.uint = Top.dut.w_otherOther_other_uint;
    ; CHECK-NEXT:       assign MyView.sub_uint = Top.dut.submodule.w_uint;
    ; CHECK-NEXT:       assign MyView.sub_vec[0] = Top.dut.submodule.w_vec_0;
    ; CHECK-NEXT:       assign MyView.sub_vec[1] = Top.dut.submodule.w_vec_1;
    ; CHECK-NEXT:       assign MyView.sub_vecOfBundle[0].sint = Top.dut.submodule.w_vecOfBundle_0_sint;
    ; CHECK-NEXT:       assign MyView.sub_vecOfBundle[0].uint = Top.dut.submodule.w_vecOfBundle_0_uint;
    ; CHECK-NEXT:       assign MyView.sub_vecOfBundle[1].sint = Top.dut.submodule.w_vecOfBundle_1_sint;
    ; CHECK-NEXT:       assign MyView.sub_vecOfBundle[1].uint = Top.dut.submodule.w_vecOfBundle_1_uint;
    ; CHECK:          endmodule

    ; EXTRACT:        FILE "Wire/firrtl/gct/MyInterface.sv"
    ; NOEXTRACT-NOT:  FILE {{.*}}/MyInterface.sv
    ; CHECK:          interface MyInterface;
    ; CHECK-NEXT:       // a wire called 'uint'
    ; CHECK-NEXT:       logic uint;
    ; CHECK-NEXT:       // a vector called 'vec'
    ; CHECK-NEXT:       logic vec[0:1];
    ; CHECK-NEXT:       // a 2D vector called 'multivec'
    ; CHECK-NEXT:       logic multivec[0:1][0:2];
    ; CHECK-NEXT:       // a vector of a bundle
    ; CHECK-NEXT:       VecOfBundle vecOfBundle[2]();
    ; CHECK-NEXT:       // another bundle
    ; CHECK-NEXT:       OtherOther otherOther();
    ; CHECK-NEXT:       // a wire called 'uint' in the submodule
    ; CHECK-NEXT:       logic sub_uint;
    ; CHECK-NEXT:       // a vector called 'vec' in the submodule
    ; CHECK-NEXT:       logic sub_vec[0:1];
    ; CHECK-NEXT:       // a vector of a bundle in the submodule with a
    ; CHECK-NEXT:       // multiline comment
    ; CHECK-NEXT:       Sub_vecOfBundle sub_vecOfBundle[2]();
    ; CHECK-NEXT:     endinterface

    ; EXTRACT:        FILE "Wire/firrtl/gct/VecOfBundle.sv"
    ; NOEXTRACT-NOT:  FILE {{.*}}/VecOfBundle.sv
    ; CHECK:          interface VecOfBundle;
    ; CHECK-NEXT:       logic [1:0] sint;
    ; CHECK-NEXT:       logic [3:0] uint;
    ; CHECK-NEXT:     endinterface

    ; EXTRACT:        FILE "Wire/firrtl/gct/OtherOther.sv"
    ; NOEXTRACT-NOT:  FILE {{.*}}/OtherOther.sv
    ; CHECK:          interface OtherOther;
    ; CHECK-NEXT:       Other other();
    ; CHECK:          endinterface

    ; EXTRACT:        FILE "Wire/firrtl/gct/Other.sv"
    ; NOEXTRACT-NOT:  FILE {{.*}}/Other.sv
    ; CHECK:          interface Other;
    ; CHECK-NEXT:       logic [1:0] sint;
    ; CHECK-NEXT:       logic [3:0] uint;
    ; CHECK:          endinterface

    ; EXTRACT:        FILE "Wire/firrtl/gct/Sub_vecOfBundle.sv"
    ; NOEXTRACT-NOT:  FILE {{.*}}/Sub_vecOfBundle.sv
    ; CHECK:          interface Sub_vecOfBundle;
    ; CHECK-NEXT:       logic [1:0] sint;
    ; CHECK-NEXT:       logic [3:0] uint;
    ; CHECK:          endinterface

    ; YAML:           FILE "Wire/firrtl/gct/gct.yaml"
    ; YAML:             - name: MyInterface
    ; YAML-NEXT:          fields:
    ; YAML-NEXT:            - name: uint
    ; YAML-NEXT:              description: 'a wire called ''uint'''
    ; YAML-NEXT:              dimensions: [  ]
    ; YAML-NEXT:              width: 1
    ; YAML-NEXT:            - name: vec
    ; YAML-NEXT:              description: 'a vector called ''vec'''
    ; YAML-NEXT:              dimensions: [ 2 ]
    ; YAML-NEXT:              width: 1
    ; YAML-NEXT:            - name: multivec
    ; YAML-NEXT:              description: 'a 2D vector called ''multivec'''
    ; YAML-NEXT:              dimensions: [ 3, 2 ]
    ; YAML-NEXT:              width: 1
    ; YAML-NEXT:            - name: sub_uint
    ; YAML-NEXT:              description: 'a wire called ''uint'' in the submodule'
    ; YAML-NEXT:              dimensions: [  ]
    ; YAML-NEXT:              width: 1
    ; YAML-NEXT:            - name: sub_vec
    ; YAML-NEXT:              description: 'a vector called ''vec'' in the submodule'
    ; YAML-NEXT:              dimensions: [ 2 ]
    ; YAML-NEXT:              width: 1
    ; YAML-NEXT:          instances:
    ; YAML-NEXT:            - name: vecOfBundle
    ; YAML-NEXT:              description: a vector of a bundle
    ; YAML-NEXT:              dimensions: [ 2 ]
    ; YAML-NEXT:              interface:
    ; YAML-NEXT:                name: VecOfBundle
    ; YAML-NEXT:                fields:
    ; YAML-NEXT:                  - name: sint
    ; YAML-NEXT:                    dimensions: [  ]
    ; YAML-NEXT:                    width: 2
    ; YAML-NEXT:                  - name: uint
    ; YAML-NEXT:                    dimensions: [  ]
    ; YAML-NEXT:                    width: 4
    ; YAML-NEXT:                instances: []
    ; YAML-NEXT:            - name: otherOther
    ; YAML-NEXT:              description: another bundle
    ; YAML-NEXT:              dimensions: [  ]
    ; YAML-NEXT:              interface:
    ; YAML-NEXT:                name: OtherOther
    ; YAML-NEXT:                fields: []
    ; YAML-NEXT:                instances:
    ; YAML-NEXT:                  - name: other
    ; YAML-NEXT:                    dimensions: [  ]
    ; YAML-NEXT:                    interface:
    ; YAML-NEXT:                      name: Other
    ; YAML-NEXT:                      fields:
    ; YAML-NEXT:                        - name: sint
    ; YAML-NEXT:                          dimensions: [  ]
    ; YAML-NEXT:                          width: 2
    ; YAML-NEXT:                        - name: uint
    ; YAML-NEXT:                          dimensions: [  ]
    ; YAML-NEXT:                          width: 4
    ; YAML-NEXT:                      instances: []
    ; YAML-NEXT:            - name: sub_vecOfBundle
    ; YAML-NEXT:              description: "a vector of a bundle in the submodule with a\n// multiline comment"
    ; YAML-NEXT:              dimensions: [ 2 ]
    ; YAML-NEXT:              interface:
    ; YAML-NEXT:                name: Sub_vecOfBundle
    ; YAML-NEXT:                fields:
    ; YAML-NEXT:                  - name: sint
    ; YAML-NEXT:                    dimensions: [  ]
    ; YAML-NEXT:                    width: 2
    ; YAML-NEXT:                  - name: uint
    ; YAML-NEXT:                    dimensions: [  ]
    ; YAML-NEXT:                    width: 4
    ; YAML-NEXT:                instances: []
