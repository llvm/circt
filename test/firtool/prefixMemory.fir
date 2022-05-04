; RUN: firtool %s --format=fir --repl-seq-mem --repl-seq-mem-file="dummy" --ir-fir | FileCheck %s 
; RUN: firtool %s --format=fir  --repl-seq-mem --repl-seq-mem-file="dummy" --ir-sv | FileCheck --check-prefix=HW %s

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
; CHECK:        = firrtl.mem Undefined  {depth = 8 : i64, groupID = 4294967294 : ui32, modName = "prefix1_mem_ext", name = "prefix1_mem", portNames = ["MPORT", "readData_MPORT"], readLatency = 1 : i32, writeLatency = 1 : i32}
; CHECK-LABEL:  firrtl.module private @prefix2_Baz
; CHECK:        = firrtl.mem Undefined  {depth = 8 : i64, groupID = 4294967293 : ui32, modName = "prefix2_mem_ext", name = "prefix2_mem", portNames = ["MPORT", "readData_MPORT"], readLatency = 1 : i32, writeLatency = 1 : i32}

; HW-LABEL:  hw.module.extern @prefix2_mem_ext(%R0_addr: i3, %R0_en: i1, %R0_clk: i1, %W0_addr: i3, %W0_en: i1, %W0_clk: i1, %W0_data: i1) -> (R0_data: i1)
; HW-LABEL:  hw.module.extern @prefix1_mem_ext(%R0_addr: i3, %R0_en: i1, %R0_clk: i1, %W0_addr: i3, %W0_en: i1, %W0_clk: i1, %W0_data: i1) -> (R0_data: i1)
