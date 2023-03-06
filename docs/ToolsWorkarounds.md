# EDA Tool Workarounds

This documents various bugs found in EDA tools and their workarounds in circt.
Each but will have a brief description, example code, and the mitigation added
(with links to the commit when possible).  


# Inline Array calculations can cause synthesis failures

Some tools have bugs (version dependent) in const prop in this case.

## Example
```
module Foo (input clock, input in, output [2:0] out);
  reg [2:0] state;
  wire [7:0][2:0] array = 24'h4 << 6;
  wire [2:0] a = array[state];
  wire [2:0] b = array[state + 3'h1 + 3'h1];
  // works:      array[state + (3'h1 + 3'h1)]
  // works:      array[state + 3'h2]
  always @(posedge clock) state <= in ? a : b;
  assign out = b;
endmodule
```

## Workaround

Flag added to export verilog to force array index calculations to not be inline.

https://github.com/llvm/circt/commit/15a1f95f2d59767f20b459a12ac42338de22bc97

# Memory semantics changed by synthesis

Read/Write forwarding behavior is dependent on memory size, since the synthesis 
tool changes it's mapping based on that.  The "optimized" mapping does not 
preserve the behavior of the verilog.  This is a known issue reported on various
forums by multiple people.  There are some version dependencies on when this
manifests.

## Example
```
Qux:
  module Qux:
    input clock: Clock
    input addr: UInt<1>
    input r: {en: UInt<1>, flip data: {a: UInt<32>, b: UInt<32>}, addr: UInt<1>}
    input w: {en: UInt<1>, data: {a: UInt<32>, b: UInt<32>}, addr: UInt<1>, mask: {a: UInt<1>, b: UInt<1>}}

    mem m :
      data-type => {a: UInt<32>, b: UInt<32>}
      depth => 1
      reader => r
      writer => w
      read-latency => 0
      write-latency => 1
      read-under-write => undefined

    m.r.clk <= clock
    m.r.en <= r.en
    m.r.addr <= r.addr
    r.data <= m.r.data

    m.w.clk <= clock
    m.w.en <= w.en
    m.w.addr <= w.addr
    m.w.data <= w.data
    m.w.mask <= w.mask
```
Compile with either firtool -repl-seq-mem -repl-seq-mem-file=mem.conf Foo.fir and firrtl -i Foo.fir.

## Workaround

FIRRTL memory lowering has a flag to generate attributes on memory 
implementations that preserve the behavior described in the verilog.  This is 
not a general solution, this bug could impact anyone making memory-looking 
things.  It was decided not to try to reverse engineer the conditions which
cause the bug to manifest (since they are version dependent), thus there isn't
a universal fix that can be applied in the generated verilog.

https://github.com/llvm/circt/commit/e9f443be475e0ef796c0c6af1ce09d6e783fcd5a