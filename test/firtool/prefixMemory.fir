; RUN: firtool --no-dedup %s --ir-fir | FileCheck %s --check-prefix=SIM-FIR --check-prefix=CHECK-FIR
; RUN: firtool --no-dedup %s --ir-sv | FileCheck %s --check-prefix=SIM-HW --check-prefix=CHECK-HW
; RUN: firtool --no-dedup %s --repl-seq-mem --repl-seq-mem-file=test.txt --ir-fir | FileCheck %s --check-prefix=REPL-FIR --check-prefix=CHECK-FIR
; RUN: firtool --no-dedup %s --repl-seq-mem --repl-seq-mem-file=test.txt --ir-sv | FileCheck %s --check-prefix=REPL-HW --check-prefix=CHECK-HW

FIRRTL version 4.0.0
circuit Foo : %[[
  {
    "class":"chisel3.ModulePrefixAnnotation",
    "prefix":"prefix1_",
    "target":"~Foo|Bar>mem"
  },
  {
    "class":"chisel3.ModulePrefixAnnotation",
    "prefix":"prefix2_",
    "target":"~Foo|Baz>mem"
  },
  {
    "class": "firrtl.transforms.DontTouchAnnotation",
    "target": "~Foo|Bar>mem"
  },
  {
    "class": "firrtl.transforms.DontTouchAnnotation",
    "target": "~Foo|Baz>mem"
  }
]]

  ; REPL-FIR: firrtl.memmodule private @prefix1_mem
  ; REPL-HW: hw.module.extern @prefix1_mem

  ; CHECK-FIR-LABEL: firrtl.module private @Bar
  ; CHECK-HW-LABEL: hw.module private @Bar
  module Bar :
    input clock : Clock
    input reset : Reset
    input readAddr : UInt<3>
    output readData : UInt<32>
    input writeEn : UInt<1>
    input writeAddr : UInt<3>
    input writeData : UInt<32>

    ; REPL-FIR:     firrtl.instance mem sym @{{[^ ]+}} {annotations = [{circt.nonlocal = @memNLA, class = "circt.tracker", id = distinct[0]<>}]} @prefix1_mem
    ; REPL-HW:      hw.instance "mem" sym @{{[^ ]+}} @prefix1_mem
    ; SIM-FIR:      firrtl.mem
    ; SIM-FIR-SAME:   name = "mem"
    ; SIM-FIR-SAME:   prefix = "prefix1_"
    ; SIM-HW:       hw.instance "mem_ext" @prefix1_mem
    mem mem :
      data-type => UInt<1>
      depth => 8
      read-latency => 1
      write-latency => 1
      reader => readData_MPORT
      writer => MPORT
    invalidate mem.readData_MPORT.addr
    invalidate mem.readData_MPORT.clk
    connect mem.readData_MPORT.en, UInt<1>(0h0)
    invalidate mem.MPORT.addr
    invalidate mem.MPORT.clk
    connect mem.MPORT.en, UInt<1>(0h0)
    invalidate mem.MPORT.data
    invalidate mem.MPORT.mask
    connect mem.readData_MPORT.addr, readAddr
    connect mem.readData_MPORT.clk, clock
    connect readData, mem.readData_MPORT.data
    when writeEn :
      connect mem.MPORT.addr, writeAddr
      connect mem.MPORT.clk, clock
      connect mem.MPORT.en, UInt<1>(0h1)
      connect mem.MPORT.mask, UInt<1>(0h0)
      connect mem.MPORT.data, writeData
      connect mem.MPORT.mask, UInt<1>(0h1)

  ; REPL-FIR: firrtl.memmodule private @prefix2_mem
  ; REPL-HW: hw.module.extern @prefix2_mem

  ; CHECK-HW-LABEL: hw.module private @Baz
  module Baz :
    input clock : Clock
    input reset : Reset
    input readAddr : UInt<3>
    output readData : UInt<32>
    input writeEn : UInt<1>
    input writeAddr : UInt<3>
    input writeData : UInt<32>

    ; REPL-FIR:     firrtl.module private @Baz
    ; REPL-FIR-NEXT:     firrtl.instance mem sym @sym {annotations = [{circt.nonlocal = @memNLA_0, class = "circt.tracker", id = distinct[1]<>}]} @prefix2_mem
    ; REPL-HW:      hw.instance "mem" sym @{{[^ ]+}} @prefix2_mem
    ; SIM-FIR:      firrtl.mem
    ; SIM-FIR-SAME:   name = "mem"
    ; SIM-FIR-SAME:   prefix = "prefix2_"
    ; SIM-HW:       hw.instance "mem_ext" @prefix2_mem
    mem mem :
      data-type => UInt<2>
      depth => 8
      read-latency => 1
      write-latency => 1
      reader => readData_MPORT
      writer => MPORT
    invalidate mem.readData_MPORT.addr
    invalidate mem.readData_MPORT.clk
    connect mem.readData_MPORT.en, UInt<1>(0h0)
    invalidate mem.MPORT.addr
    invalidate mem.MPORT.clk
    connect mem.MPORT.en, UInt<1>(0h0)
    invalidate mem.MPORT.data
    invalidate mem.MPORT.mask
    connect mem.readData_MPORT.addr, readAddr
    connect mem.readData_MPORT.clk, clock
    connect readData, mem.readData_MPORT.data
    when writeEn :
      connect mem.MPORT.addr, writeAddr
      connect mem.MPORT.clk, clock
      connect mem.MPORT.en, UInt<1>(0h1)
      connect mem.MPORT.mask, UInt<1>(0h0)
      connect mem.MPORT.data, writeData
      connect mem.MPORT.mask, UInt<1>(0h1)

  public module Foo :
    input clock : Clock
    input reset : UInt<1>
    input readAddr : UInt<3>
    output readData : UInt<32>
    input writeEn : UInt<1>
    input writeAddr : UInt<3>
    input writeData : UInt<32>

    inst bar of Bar
    connect bar.clock, clock
    connect bar.reset, reset
    connect bar.readAddr, readAddr
    connect bar.writeEn, writeEn
    connect bar.writeAddr, writeAddr
    connect bar.writeData, writeData
    inst baz of Baz
    connect baz.clock, clock
    connect baz.reset, reset
    connect baz.readAddr, readAddr
    connect baz.writeEn, writeEn
    connect baz.writeAddr, writeAddr
    connect baz.writeData, writeData
    node _readData_T = xor(bar.readData, baz.readData)
    connect readData, _readData_T
