// RUN: circt-translate --export-firrtl %s --target-line-length=8 --disable-line-wrap | FileCheck %s

// Ensure statement isn't wrapped when it overflows the target line length
// CHECK:      connect o, add(a, b)

firrtl.circuit "Foo" {
  firrtl.module @Foo(in %a: !firrtl.uint<8>, in %b: !firrtl.uint<8>,
                     out %o: !firrtl.uint<9>) {
    %0 = firrtl.add %a, %b : (!firrtl.uint<8>, !firrtl.uint<8>) -> !firrtl.uint<9>
    firrtl.connect %o, %0 : !firrtl.uint<9>, !firrtl.uint<9>
  }
}
