; RUN: firtool %s --repl-seq-mem --repl-seq-mem-file=test.txt --ir-fir | FileCheck %s 
; RUN: firtool %s --repl-seq-mem --repl-seq-mem-file=test.txt --ir-sv | FileCheck --check-prefix=HW %s

circuit Foo : %[[
  {
    "class":"sifive.enterprise.firrtl.NestedPrefixModulesAnnotation",
    "prefix":"prefix1_",
    "inclusive":true,
    "target":"~Foo|Bar"
  },
  {
    "class":"sifive.enterprise.firrtl.NestedPrefixModulesAnnotation",
    "prefix":"prefix2_",
    "inclusive":true,
    "target":"~Foo|Baz"
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
  module Bar :
    input clock : Clock
    input reset : Reset
    input readAddr : UInt<3>
    output readData : UInt<32>
    input writeEn : UInt<1>
    input writeAddr : UInt<3>
    input writeData : UInt<32>

    mem mem :
      data-type => UInt<1>
      depth => 8
      read-latency => 1
      write-latency => 1
      reader => readData_MPORT
      writer => MPORT
    mem.readData_MPORT.addr is invalid
    mem.readData_MPORT.clk is invalid
    mem.readData_MPORT.en <= UInt<1>("h0")
    mem.MPORT.addr is invalid
    mem.MPORT.clk is invalid
    mem.MPORT.en <= UInt<1>("h0")
    mem.MPORT.data is invalid
    mem.MPORT.mask is invalid
    mem.readData_MPORT.addr <= readAddr
    mem.readData_MPORT.clk <= clock
    readData <= mem.readData_MPORT.data
    when writeEn :
      mem.MPORT.addr <= writeAddr
      mem.MPORT.clk <= clock
      mem.MPORT.en <= UInt<1>("h1")
      mem.MPORT.mask <= UInt<1>("h0")
      mem.MPORT.data <= writeData
      mem.MPORT.mask <= UInt<1>("h1")

  module Baz :
    input clock : Clock
    input reset : Reset
    input readAddr : UInt<3>
    output readData : UInt<32>
    input writeEn : UInt<1>
    input writeAddr : UInt<3>
    input writeData : UInt<32>

    mem mem :
      data-type => UInt<1>
      depth => 8
      read-latency => 1
      write-latency => 1
      reader => readData_MPORT
      writer => MPORT
    mem.readData_MPORT.addr is invalid
    mem.readData_MPORT.clk is invalid
    mem.readData_MPORT.en <= UInt<1>("h0")
    mem.MPORT.addr is invalid
    mem.MPORT.clk is invalid
    mem.MPORT.en <= UInt<1>("h0")
    mem.MPORT.data is invalid
    mem.MPORT.mask is invalid
    mem.readData_MPORT.addr <= readAddr
    mem.readData_MPORT.clk <= clock
    readData <= mem.readData_MPORT.data
    when writeEn :
      mem.MPORT.addr <= writeAddr
      mem.MPORT.clk <= clock
      mem.MPORT.en <= UInt<1>("h1")
      mem.MPORT.mask <= UInt<1>("h0")
      mem.MPORT.data <= writeData
      mem.MPORT.mask <= UInt<1>("h1")

  module Foo :
    input clock : Clock
    input reset : UInt<1>
    input readAddr : UInt<3>
    output readData : UInt<32>
    input writeEn : UInt<1>
    input writeAddr : UInt<3>
    input writeData : UInt<32>

    inst bar of Bar
    bar.clock <= clock
    bar.reset <= reset
    bar.readAddr <= readAddr
    bar.writeEn <= writeEn
    bar.writeAddr <= writeAddr
    bar.writeData <= writeData
    inst baz of Baz
    baz.clock <= clock
    baz.reset <= reset
    baz.readAddr <= readAddr
    baz.writeEn <= writeEn
    baz.writeAddr <= writeAddr
    baz.writeData <= writeData
    node _readData_T = xor(bar.readData, baz.readData)
    readData <= _readData_T

; CHECK-LABEL:  firrtl.module private @prefix1_Bar
; CHECK:          firrtl.instance mem @prefix1_mem
; CHECK-LABEL:  firrtl.module private @prefix2_Baz
; CHECK:          firrtl.instance mem @prefix2_mem
; CHECK:        firrtl.memmodule private @prefix1_mem
; CHECK:        firrtl.memmodule private @prefix2_mem

; HW-LABEL: hw.module.extern @prefix1_mem
; HW-LABEL: hw.module.extern @prefix2_mem
