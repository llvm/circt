; RUN: firtool %s --parse-only --select-instance-choice=Platform=ASIC | FileCheck %s
; RUN: firtool %s --ir-hw --disable-opt --select-instance-choice=Platform=ASIC | FileCheck %s --check-prefix=ASIC
; RUN: firtool %s --ir-hw --disable-opt --select-default-for-unspecified-instance-choice | FileCheck %s --check-prefixes=DEFAULT

FIRRTL version 4.1.0
; CHECK: firrtl.circuit "Foo" attributes {select_inst_choice = ["Platform=ASIC"]} {
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

  public module Foo:
    input clock: Clock

    ; CHECK:      %inst_clock = firrtl.instance_choice inst interesting_name @DefaultTarget
    ; CHECK-SAME: alternatives @Platform {
    ; CHECK-SAME:   @FPGA -> @FPGATarget,
    ; CHECK-SAME:   @ASIC -> @ASICTarget
    ; CHECK-SAME: } (in clock: !firrtl.clock)
    ; ASIC: hw.instance "inst" @ASICTarget
    ; DEFAULT: hw.instance "inst" @DefaultTarget
    instchoice inst of DefaultTarget, Platform :
      FPGA => FPGATarget
      ASIC => ASICTarget

    connect inst.clock, clock
