// RUN: circt-verilog --ir-moore -mlir-print-debuginfo %s \
// RUN:   | FileCheck %s --check-prefix=DEFAULT
// RUN: circt-verilog --ir-moore -mlir-print-debuginfo --make-source-locations-proximate=false %s \
// RUN:   | FileCheck %s --check-prefix=NOPROX
// REQUIRES: slang

// Internal issue in Slang v3 about jump depending on uninitialised value.
// UNSUPPORTED: valgrind

// Windows uses different paths formats; only check the Unix format.
// UNSUPPORTED: system-windows

// The "proximate" locations are relative paths, so they don't start with a `/`.
// DEFAULT:      moore.module @Empty()
// DEFAULT-NEXT:   moore.output
// DEFAULT-NEXT: } loc(#[[LOC:.+]])
// DEFAULT:      #[[LOC]] = loc("{{[^/].*}}":{{[0-9]+:[0-9]+}})

// The "non-proximate" locations use the path given on the command line,
// i.e., `%s`, which is absolute.
// NOPROX:      moore.module @Empty()
// NOPROX-NEXT:   moore.output
// NOPROX-NEXT: } loc(#[[LOC:.+]])
// NOPROX:      #[[LOC]] = loc("/{{[^.].*}}":{{[0-9]+:[0-9]+}})

module Empty;
endmodule
