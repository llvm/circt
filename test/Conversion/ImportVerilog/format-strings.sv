// RUN: circt-verilog --ir-moore %s | FileCheck %s
// REQUIRES: slang
// UNSUPPORTED: valgrind

// The %v specifier prints the signal strength of a net (IEEE 1800-2017
// § 21.2.1.5 "Strength format"). The Moore dialect does not model drive
// strength, so driven bits report strong strength: St0, St1, StX, or HiZ for
// high-impedance bits. Vector operands print one strength per bit, most
// significant bit first, separated by spaces.

// CHECK-LABEL: moore.module @FormatStrength(
// CHECK-DAG: [[ONE:%.+]] = moore.constant 1 : l1
// CHECK-DAG: [[ZERO:%.+]] = moore.constant 0 : l1
// CHECK-DAG: [[Z:%.+]] = moore.constant bZ : l1
module FormatStrength(input wire w, input wire [1:0] v);
  bit b;
  initial begin
    // Four-valued scalar: compare against Z, 0, and 1; anything else is X.
    // CHECK: [[W:%.+]] = moore.read %w
    // CHECK: [[ISZ:%.+]] = moore.case_eq [[W]], [[Z]] : l1
    // CHECK: [[ISZERO:%.+]] = moore.case_eq [[W]], [[ZERO]] : l1
    // CHECK: [[ISONE:%.+]] = moore.case_eq [[W]], [[ONE]] : l1
    // CHECK: [[LIT1:%.+]] = moore.fmt.literal "St1"
    // CHECK: [[STR1:%.+]] = moore.fstring_to_string [[LIT1]]
    // CHECK: [[LITX:%.+]] = moore.fmt.literal "StX"
    // CHECK: [[STRX:%.+]] = moore.fstring_to_string [[LITX]]
    // CHECK: moore.conditional [[ISONE]] : i1 -> string
    // CHECK: moore.yield [[STR1]]
    // CHECK: moore.yield [[STRX]]
    // CHECK: [[LIT0:%.+]] = moore.fmt.literal "St0"
    // CHECK: [[STR0:%.+]] = moore.fstring_to_string [[LIT0]]
    // CHECK: moore.conditional [[ISZERO]] : i1 -> string
    // CHECK: [[LITZ:%.+]] = moore.fmt.literal "HiZ"
    // CHECK: [[STRZ:%.+]] = moore.fstring_to_string [[LITZ]]
    // CHECK: [[SELZ:%.+]] = moore.conditional [[ISZ]] : i1 -> string
    // CHECK: moore.yield [[STRZ]]
    // CHECK: moore.fmt.string [[SELZ]]
    $display("%v", w);

    // Vector: one strength per bit, MSB first, separated by spaces.
    // CHECK: [[V:%.+]] = moore.read %v
    // CHECK: moore.extract [[V]] from 1 : l2 -> l1
    // CHECK: moore.fmt.string
    // CHECK: moore.fmt.literal " "
    // CHECK: moore.extract [[V]] from 0 : l2 -> l1
    // CHECK: moore.fmt.string
    $display("%v", v);

    // Two-valued: the bit itself selects between St1 and St0.
    // CHECK: [[B:%.+]] = moore.read %b
    // CHECK: [[SELB:%.+]] = moore.conditional [[B]] : i1 -> string
    // CHECK: moore.yield [[STR1]]
    // CHECK: moore.yield [[STR0]]
    // CHECK: moore.fmt.string [[SELB]]
    $display("%v", b);
  end
endmodule
