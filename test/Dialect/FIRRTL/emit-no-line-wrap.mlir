// RUN: circt-translate --export-firrtl %s | FileCheck %s --check-prefix WRAP
// RUN: circt-translate --export-firrtl %s --target-line-length=0 | FileCheck %s --check-prefix NOWRAP

// Ensure we wrap according to default line length
// WRAP:      connect o,
// WRAP-NEXT:   tail(add(add(add(add(add(add(add(add(a, b), c), d), a), b), c), d), a), 8)

// Ensure we don't wrap with target line length 0, but we still see newlines
// NOWRAP:      input a : UInt<8>
// NOWRAP:      output o : UInt<8>
// NOWRAP:      connect o, tail(add(add(add(add(add(add(add(add(a, b), c), d), a), b), c), d), a), 8)

firrtl.circuit "Foo" {
  firrtl.module @Foo(in %a: !firrtl.uint<8>, in %b: !firrtl.uint<8>,
                     in %c: !firrtl.uint<8>, in %d: !firrtl.uint<8>,
                     out %o: !firrtl.uint<8>) {
    %0 = firrtl.add %a, %b : (!firrtl.uint<8>, !firrtl.uint<8>) -> !firrtl.uint<9>
    %1 = firrtl.add %0, %c : (!firrtl.uint<9>, !firrtl.uint<8>) -> !firrtl.uint<10>
    %2 = firrtl.add %1, %d : (!firrtl.uint<10>, !firrtl.uint<8>) -> !firrtl.uint<11>
    %3 = firrtl.add %2, %a : (!firrtl.uint<11>, !firrtl.uint<8>) -> !firrtl.uint<12>
    %4 = firrtl.add %3, %b : (!firrtl.uint<12>, !firrtl.uint<8>) -> !firrtl.uint<13>
    %5 = firrtl.add %4, %c : (!firrtl.uint<13>, !firrtl.uint<8>) -> !firrtl.uint<14>
    %6 = firrtl.add %5, %d : (!firrtl.uint<14>, !firrtl.uint<8>) -> !firrtl.uint<15>
    %7 = firrtl.add %6, %a : (!firrtl.uint<15>, !firrtl.uint<8>) -> !firrtl.uint<16>
    %8 = firrtl.tail %7, 8 : (!firrtl.uint<16>) -> !firrtl.uint<8>
    firrtl.connect %o, %8 : !firrtl.uint<8>, !firrtl.uint<8>
  }
}
