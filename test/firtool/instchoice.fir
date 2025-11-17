; RUN: firtool %s --parse-only --select-instance-choice=Platform=ASIC | FileCheck %s
; RUN: firtool %s --ir-fir --select-default-for-unspecified-instance-choice | FileCheck %s --check-prefixes=DEFAULT
; RUN: firtool %s --ir-fir --select-instance-choice=Platform=FPGA | FileCheck %s --check-prefix=FPGA
; RUN: firtool %s --ir-fir --select-instance-choice=Platform=ASIC | FileCheck %s --check-prefix=ASIC

FIRRTL version 5.1.0
; CHECK: firrtl.circuit "Foo"
; CHECK-SAME: select_inst_choice = ["Platform=ASIC"]
circuit Foo:
  ; CHECK: firrtl.option @Platform
  option Platform:
    ; CHECK-NEXT: @FPGA
    FPGA
    ; CHECK-NEXT: @ASIC
    ASIC

  module DefaultTarget:
    input clock: Clock
    inst ext of DefaultExt

  module FPGATarget:
    input clock: Clock
    inst ext of FPGAExt

  module ASICTarget:
    input clock: Clock
    inst ext of ASICExt

  ; CHECK-LABEL: @Foo
  public module Foo:
    input clock: Clock

    ; CHECK:      %inst_clock = firrtl.instance_choice inst interesting_name @DefaultTarget
    ; CHECK-SAME: alternatives @Platform {
    ; CHECK-SAME:   @FPGA -> @FPGATarget,
    ; CHECK-SAME:   @ASIC -> @ASICTarget
    ; CHECK-SAME: } (in clock: !firrtl.clock)
    ; DEFAULT: firrtl.instance inst @DefaultTarget
    ; FPGA:    firrtl.instance inst @FPGATarget
    ; ASIC:    firrtl.instance inst @ASICTarget
    instchoice inst of DefaultTarget, Platform :
      FPGA => FPGATarget
      ASIC => ASICTarget

    connect inst.clock, clock

  extmodule DefaultExt:
  extmodule FPGAExt:
  extmodule ASICExt:

  domain ClockDomain:

  module DefaultTargetWithDomain:
    input A: Domain of ClockDomain
    input clock: Clock domains [A]
    inst ext of DefaultExt

  module FPGATargetWithDomain:
    input A: Domain of ClockDomain
    input clock: Clock domains [A]
    inst ext of FPGAExt

  module ASICTargetWithDomain:
    input A: Domain of ClockDomain
    input clock: Clock domains [A]
    inst ext of ASICExt

  ; CHECK-LABEL: @Bar
  public module Bar:
    input A: Domain of ClockDomain
    input clock: Clock domains [A]

    ; CHECK:      %inst_A, %inst_clock = firrtl.instance_choice inst interesting_name
    ; CHECK-SAME: @DefaultTargetWithDomain
    ; CHECK-SAME: alternatives @Platform {
    ; CHECK-SAME:   @FPGA -> @FPGATargetWithDomain,
    ; CHECK-SAME:   @ASIC -> @ASICTargetWithDomain
    ; CHECK-SAME: } (in A: !firrtl.domain of @ClockDomain, in clock: !firrtl.clock domains [A])
    ; DEFAULT: firrtl.instance inst @DefaultTargetWithDomain
    ; FPGA:    firrtl.instance inst @FPGATargetWithDomain
    ; ASIC:    firrtl.instance inst @ASICTargetWithDomain
    instchoice inst of DefaultTargetWithDomain, Platform :
      FPGA => FPGATargetWithDomain
      ASIC => ASICTargetWithDomain

    connect inst.clock, clock
    domain_define inst.A = A
