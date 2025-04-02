; RUN: firtool %s | FileCheck %s

FIRRTL version 5.0.0
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
  public module PrintTest :
    input clock : Clock
    input cond : UInt<1>
    input a : UInt<8>

    ; CHECK:       reg [31:0] fd_test_txt;
    ; CHECK-NEXT:  initial begin
    ; CHECK-NEXT:    fd_test_txt = __circt_lib_logging::FileDescriptor::get("test.txt");
    ; CHECK-NEXT:  end

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
    printf(clock, cond, "[{{SimulationTime}}]: {{HierarchicalModuleName}}\n")
    
    ; CHECK-NEXT: $fwrite(fd_test_txt, "hello");
    fprintf(clock, cond, "test.txt", "hello")

    ; CHECK-NEXT:         $fwrite(fd_test_txt, "[%0t]: %m\n", $time);
    fprintf(clock, cond, "test.txt", "[{{SimulationTime}}]: {{HierarchicalModuleName}}\n")
