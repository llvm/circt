// RUN: circt-translate --export-firrtl %s --target-line-length=40 | sed -e 's/ @\[.*\]//' | FileCheck %s --check-prefix WRAP
// RUN: circt-translate --export-firrtl %s --target-line-length=40 --disable-line-wrap | sed -e 's/ @\[.*\]//' | FileCheck %s --check-prefix NOWRAP

// With a narrow target line length the connect's expression wraps onto its own
// indented line.
// WRAP:      connect o,
// WRAP-NEXT:   tail(add(add(add(a, b), c), d), 3)

// With --disable-line-wrap the expression stays on one line, while structural
// newlines between ports and statements are preserved.
// NOWRAP:      input a : UInt<8>
// NOWRAP:      output o : UInt<8>
// NOWRAP:      connect o, tail(add(add(add(a, b), c), d), 3)

firrtl.circuit "Foo" {
  firrtl.module @Foo(in %a: !firrtl.uint<8>, in %b: !firrtl.uint<8>,
                     in %c: !firrtl.uint<8>, in %d: !firrtl.uint<8>,
                     out %o: !firrtl.uint<8>) {
    %0 = firrtl.add %a, %b : (!firrtl.uint<8>, !firrtl.uint<8>) -> !firrtl.uint<9>
    %1 = firrtl.add %0, %c : (!firrtl.uint<9>, !firrtl.uint<8>) -> !firrtl.uint<10>
    %2 = firrtl.add %1, %d : (!firrtl.uint<10>, !firrtl.uint<8>) -> !firrtl.uint<11>
    %3 = firrtl.tail %2, 3 : (!firrtl.uint<11>) -> !firrtl.uint<8>
    firrtl.connect %o, %3 : !firrtl.uint<8>, !firrtl.uint<8>
  }
}
