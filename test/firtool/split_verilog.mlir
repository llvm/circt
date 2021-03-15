// RUN: firtool %s --format=mlir -split-verilog -o=%t
// RUN: FileCheck %s --check-prefix=VERILOG-FOO < %t/foo.v
// RUN: FileCheck %s --check-prefix=VERILOG-BAR < %t/bar.v
// RUN: FileCheck %s --check-prefix=VERILOG-USB < %t/usb.v
// RUN: FileCheck %s --check-prefix=VERILOG-PLL < %t/pll.v

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
