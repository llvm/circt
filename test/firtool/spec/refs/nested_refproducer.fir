; RUN: firtool %s
; RUN: firtool %s -preserve-aggregate=all -scalarize-top-module=false
FIRRTL version 3.0.0
circuit RefProducer :
  ; SPEC EXAMPLE BEGIN
  module RefProducer :
    input a : UInt<4>
    input en : UInt<1>
    input clk : Clock
    output thereg : Probe<UInt>

    when en :
      reg myreg : UInt, clk
      connect myreg, a
      define thereg = probe(myreg)
  ; SPEC EXAMPLE END
