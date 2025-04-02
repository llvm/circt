; RUN: firtool %s | FileCheck %s

FIRRTL version 5.0.0
circuit PrintTest:
  ; CHECK-LABEL: module PrintTest
  public module PrintTest :
    input clock : Clock
    input cond : UInt<1>
    input a : UInt<8>


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
    
    fprintf(clock, cond, "test.txt", "test %m %d\n", var)
    fprintf(clock, cond, "test.txt", "hello")
    fprintf(clock, cond, "%m.txt", "test %d\n", var)


