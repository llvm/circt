// RUN: firtool %s --format=mlir -verilog | FileCheck %s --check-prefix=VERILOG
// RUN: firtool %s --format=mlir -split-verilog -o=%t | FileCheck %s --check-prefix=FIRTOOL
// RUN: FileCheck %s --check-prefix=VERILOG-FOO < %t/foo.v
// RUN: FileCheck %s --check-prefix=VERILOG-BAR < %t/bar.v
// RUN: FileCheck %s --check-prefix=VERILOG-USB < %t/usb.v
// RUN: FileCheck %s --check-prefix=VERILOG-PLL < %t/pll.v
// RUN: FileCheck %s --check-prefix=VERILOG-INOUT-3 < %t/inout_3.v
// RUN: FileCheck %s --check-prefix=VERILOG-INOUT-0 < %t/inout_0.v
// RUN: FileCheck %s --check-prefix=VERILOG-INOUT-1 < %t/inout_1.v
// RUN: FileCheck %s --check-prefix=VERILOG-INOUT-2 < %t/inout_2.v

sv.verbatim "// I'm everywhere"
sv.ifdef.procedural "VERILATOR" {
  sv.verbatim "// Hello"
} else {
  sv.verbatim "// World"
}
sv.verbatim ""

rtl.module @foo(%a: i1) -> (%b: i1) {
  rtl.output %a : i1
}
rtl.module @bar(%x: i1) -> (%y: i1) {
  rtl.output %x : i1
}
sv.interface @usb {
  sv.interface.signal @valid : i1
  sv.interface.signal @ready : i1
}
rtl.module.extern @pll ()

rtl.module @inout(%inout: i1) -> (%output: i1) {
  rtl.output %inout : i1
}

// This is made collide with the first renaming attempt of the `@inout` module
// above.
rtl.module.extern @inout_0 () -> ()
rtl.module.extern @inout_1 () -> ()
rtl.module.extern @inout_2 () -> ()

// FIRTOOL:      foo.sv
// FIRTOOL-NEXT: bar.sv
// FIRTOOL-NEXT: usb.sv
// FIRTOOL-NEXT: pll.sv
// FIRTOOL-NEXT: inout_3.sv
// FIRTOOL-NEXT: inout_0.sv
// FIRTOOL-NEXT: inout_1.sv
// FIRTOOL-NEXT: inout_2.sv

// VERILOG-FOO:       // I'm everywhere
// VERILOG-FOO-NEXT:  `ifdef VERILATOR
// VERILOG-FOO-NEXT:    // Hello
// VERILOG-FOO-NEXT:  `else
// VERILOG-FOO-NEXT:    // World
// VERILOG-FOO-NEXT:  `endif
// VERILOG-FOO-LABEL: module foo(
// VERILOG-FOO:       endmodule

// VERILOG-BAR:       // I'm everywhere
// VERILOG-BAR-NEXT:  `ifdef VERILATOR
// VERILOG-BAR-NEXT:    // Hello
// VERILOG-BAR-NEXT:  `else
// VERILOG-BAR-NEXT:    // World
// VERILOG-BAR-NEXT:  `endif
// VERILOG-BAR-LABEL: module bar
// VERILOG-BAR:       endmodule

// VERILOG-USB:       // I'm everywhere
// VERILOG-USB-NEXT:  `ifdef VERILATOR
// VERILOG-USB-NEXT:    // Hello
// VERILOG-USB-NEXT:  `else
// VERILOG-USB-NEXT:    // World
// VERILOG-USB-NEXT:  `endif
// VERILOG-USB-LABEL: interface usb;
// VERILOG-USB:       endinterface

// VERILOG-PLL:        // I'm everywhere
// VERILOG-PLL-NEXT:   `ifdef VERILATOR
// VERILOG-PLL-NEXT:     // Hello
// VERILOG-PLL-NEXT:   `else
// VERILOG-PLL-NEXT:     // World
// VERILOG-PLL-NEXT:   `endif
// VERILOG-PLL:        // external module pll

// VERILOG-INOUT-3:       // I'm everywhere
// VERILOG-INOUT-3-NEXT:  `ifdef VERILATOR
// VERILOG-INOUT-3-NEXT:    // Hello
// VERILOG-INOUT-3-NEXT:  `else
// VERILOG-INOUT-3-NEXT:    // World
// VERILOG-INOUT-3-NEXT:  `endif
// VERILOG-INOUT-3-LABEL: module inout_3(
// VERILOG-INOUT-3:       endmodule

// VERILOG-INOUT-0:    // external module inout_0
// VERILOG-INOUT-1:    // external module inout_1
// VERILOG-INOUT-2:    // external module inout_2


// VERILOG:       // I'm everywhere
// VERILOG-NEXT:  `ifdef VERILATOR
// VERILOG-NEXT:    // Hello
// VERILOG-NEXT:  `else
// VERILOG-NEXT:    // World
// VERILOG-NEXT:  `endif
// VERILOG-LABEL: module foo(
// VERILOG:       endmodule
// VERILOG-LABEL: module bar
// VERILOG:       endmodule
// VERILOG-LABEL: interface usb;
// VERILOG:       endinterface
// VERILOG:       // external module pll
// VERILOG:       // external module inout_0
// VERILOG:       // external module inout_1
// VERILOG:       // external module inout_2
