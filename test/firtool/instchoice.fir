; RUN: firtool %s --parse-only | FileCheck %s

FIRRTL version 4.0.0
circuit Foo:
  ; CHECK: firrtl.option @Platform
  option Platform:
    ; CHECK-NEXT: @FPGA
    FPGA
    ; CHECK-NEXT: @ASIC
    ASIC

  module DefaultTarget:
    input clock: Clock

  module FPGATarget:
    input clock: Clock

  module ASICTarget:
    input clock: Clock

  module Foo:
    input clock: Clock

    ; CHECK: %inst_clock = firrtl.instance_choice inst interesting_name @DefaultTarget
    ; CHECK-SAME: alternatives @Platform {
    ; CHECK-SAME:   @FPGA -> @FPGATarget,
    ; CHECK-SAME:   @ASIC -> @ASICTarget
    ; CHECK-SAME: } (in clock: !firrtl.clock)
    instchoice inst of DefaultTarget, Platform :
      FPGA => FPGATarget
      ASIC => ASICTarget

    inst.clock <= clock


