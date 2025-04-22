; RUN: firtool %s | FileCheck %s
; RUN: firtool %s --extract-test-code | FileCheck --check-prefix=ETC %s

FIRRTL version 5.1.0
; CHECK: `ifndef __CIRCT_LIB_LOGGING
; CHECK-NEXT:   // CIRCT Logging Library
; CHECK-NEXT:   package __circt_lib_logging;
; CHECK-NEXT:     class FileDescriptor;
; CHECK-NEXT:       static int global_id [string];
; CHECK-NEXT:       static function int get(string name);
; CHECK-NEXT:         if (global_id.exists(name) == 32'h0)
; CHECK-NEXT:           global_id[name] = $fopen(name);
; CHECK-NEXT:         return global_id[name];
; CHECK-NEXT:       endfunction
; CHECK-NEXT:     endclass
; CHECK-NEXT:   endpackage
; CHECK-NEXT:   `define __CIRCT_LIB_LOGGING
; CHECK-NEXT: `endif // not def __CIRCT_LIB_LOGGING
circuit PrintTest:
  ; CHECK-LABEL: module PrintTest
  ; ETC-LABEL: module PrintTest_assert
  ; ETC-LABEL: PrintTest
  ; ETC-NOT: $fwrite
  ; ETC-NOT: $fflush
  public module PrintTest :
    input clock : Clock
    input cond : UInt<1>
    input a : UInt<8>

    ; CHECK: if ((`PRINTF_COND_) & cond) begin
    ; CHECK: $fwrite(`PRINTF_FD_, "binary: %b %0b %8b\n", a, a, a);
    printf(clock, cond, "binary: %b %0b %8b\n", a, a, a)

    ; CHECK-NEXT: $fwrite(`PRINTF_FD_, "decimal: %d %0d %3d\n", a, a, a);
    printf(clock, cond, "decimal: %d %0d %3d\n", a, a, a)

    ; CHECK-NEXT: $fwrite(`PRINTF_FD_, "hexadecimal: %x %0x %2x\n", a, a, a);
    printf(clock, cond, "hexadecimal: %x %0x %2x\n", a, a, a)

    ; CHECK-NEXT: $fwrite(`PRINTF_FD_, "ASCII character: %c\n", a);
    printf(clock, cond, "ASCII character: %c\n", a)

    ; CHECK-NEXT: $fwrite(`PRINTF_FD_, "literals: %%\n");
    printf(clock, cond, "literals: %%\n")

    ; CHECK-NEXT: $fwrite(`PRINTF_FD_, "[%0t]: %m\n", $time);
    ; CHECK-NEXT: end
    printf(clock, cond, "[{{SimulationTime}}]: {{HierarchicalModuleName}}\n")

    ; CHECK:      if (cond) begin
    ; CHECK-NEXT: ___circt_lib_logging3A3AFileDescriptor3A3Aget_0_1 = __circt_lib_logging::FileDescriptor::get("test.txt");
    ; CHECK-NEXT: $fwrite(___circt_lib_logging3A3AFileDescriptor3A3Aget_0_1, "hello");
    fprintf(clock, cond, "test.txt", "hello")

    ; CHECK-NEXT: ___circt_lib_logging3A3AFileDescriptor3A3Aget_0_0 = __circt_lib_logging::FileDescriptor::get($sformatf("%m%c.txt",
    ; CHECK-NEXT:                                                                                                        8'h61));
    ; CHECK-NEXT: $fwrite(___circt_lib_logging3A3AFileDescriptor3A3Aget_0_0,
    ; CHECK-NEXT:         "[%0t]: static file name (w/ substitution)\n", $time);
    node c = UInt<8>(97)
    fprintf(clock, cond, "{{HierarchicalModuleName}}%c.txt", c, "[{{SimulationTime}}]: static file name (w/ substitution)\n")

    ; CHECK-NEXT: $fflush(`PRINTF_FD_);
    ; CHECK-NEXT: ___circt_lib_logging3A3AFileDescriptor3A3Aget_0 = __circt_lib_logging::FileDescriptor::get($sformatf("%0t%d.txt",
    ; CHECK-NEXT:                                                                                                      $time,
    ; CHECK-NEXT:                                                                                                      a));
    fflush(clock, cond)

    ; CHECK-NEXT: $fflush(___circt_lib_logging3A3AFileDescriptor3A3Aget_0);
    ; CHECK-NEXT: end
    fflush(clock, cond, "{{SimulationTime}}%d.txt", a);
