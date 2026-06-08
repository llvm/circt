// RUN: circt-translate --import-verilog -mlir-print-debuginfo %s | FileCheck %s --check-prefix=DEFAULT
// RUN: circt-translate --import-verilog --import-verilog-absolute-source-locations -mlir-print-debuginfo %s | FileCheck %s --check-prefix=ABSOLUTE
// REQUIRES: slang

// Internal issue in Slang v3 about jump depending on uninitialised value.
// UNSUPPORTED: valgrind

// Windows uses different paths formats; only check the Unix format.
// UNSUPPORTED: system-windows

// DEFAULT:      moore.module @Empty()
// DEFAULT-NEXT:   moore.output
// DEFAULT-NEXT: } loc(#[[LOC:.+]])
// DEFAULT:      #[[LOC]] = loc("{{[^/].*}}":{{[0-9]+:[0-9]+}})

// ABSOLUTE:      moore.module @Empty()
// ABSOLUTE-NEXT:   moore.output
// ABSOLUTE-NEXT: } loc(#[[LOC:.+]])
// ABSOLUTE:      #[[LOC]] = loc("/{{.*}}":{{[0-9]+:[0-9]+}})

module Empty;
endmodule
