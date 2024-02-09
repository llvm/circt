; RUN: rm -rf %t
; RUN: split-file %s %t
; RUN: firtool %t/parent.fir -o %t/parent --split-verilog
; RUN: firtool %t/child.fir -o %t/child --split-verilog
; RUN: verilator --lint-only -F %t/child/filelist.f -F %t/parent/filelist.f 

; RUN: firtool %t/parent.fir -o %t/parent_aggs --split-verilog --scalarize-ext-modules=false --scalarize-public-modules=false --preserve-aggregate=all
; RUN: firtool %t/child.fir -o %t/child_aggs --split-verilog --scalarize-ext-modules=false --scalarize-public-modules=false --preserve-aggregate=all
; RUN: verilator --lint-only -F %t/child_aggs/filelist.f -F %t/parent_aggs/filelist.f 
; REQUIRES: verilator

;--- parent.fir

FIRRTL version 4.0.0
circuit Parent:
  public module Parent:
  
    inst c of Child
    c.in.x <= UInt(3)
    c.in.y[0] <= shr(read(c.out.p), 3)

    inst st of SecondTop
    st.in.x <= UInt(3)
    st.in.y[0] <= shr(read(st.out.p), 3)

  extmodule Child:
    input in : { x: UInt<5>, y: UInt<2>[1] }
    output out : { z: UInt<1>, p : Probe<UInt<5>> }

  extmodule SecondTop:
    input in : { x: UInt<5>, y: UInt<2>[1] }
    output out : { z: UInt<1>, p : Probe<UInt<5>> }

;--- child.fir

FIRRTL version 4.0.0
circuit Child:
  public module Child:
    input in : { x: UInt<5>, y: UInt<2>[1] }
    output out : { z: UInt<1>, p : Probe<UInt<5>> }

    define out.p = probe(in.x)
    connect out.z, in.y[0]

    ; TODO: Shouldn't need to instantiate this!
    inst st of SecondTop
    connect st.in, in

  public module SecondTop:
    input in : { x: UInt<5>, y: UInt<2>[1] }
    output out : { z: UInt<2>, p : Probe<UInt<5>> }

    define out.p = probe(in.x)
    connect out.z, in.y[0]
