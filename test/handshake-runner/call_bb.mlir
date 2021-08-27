// RUN: handshake-runner %s | FileCheck %s
// RUN: circt-opt -create-dataflow %s | handshake-runner | FileCheck %s
// CHECK: 763 2996
module {
  func @muladd(%1:index, %2:index, %3:index) -> (index) {
    %i2 = muli %1, %2 : index
    %i3 = addi %3, %i2 : index
	 return %i3 : index
  }

  func @main() -> (index, index) {
    %c0 = constant 0 : index
    %c101 = constant 101 : index
    %c102 = constant 102 : index
    %0 = addi %c0, %c0 : index
    %c1 = constant 1 : index
    %1 = addi %0, %c102 : index
    %c103 = constant 103 : index
    %c104 = constant 104 : index
    %c105 = constant 105 : index
    %c106 = constant 106 : index
    %c107 = constant 107 : index
    %c108 = constant 108 : index
    %c109 = constant 109 : index
    %c2 = constant 2 : index
  	 %3 = call @muladd(%c104, %c2, %c103) : (index, index, index) -> index
    %c3 = constant 3 : index
    %4 = muli %c105, %c3 : index
    %5 = addi %3, %4 : index
    %c4 = constant 4 : index
    %6 = muli %c106, %c4 : index
    %7 = addi %5, %6 : index
    %c5 = constant 5 : index
    %8 = muli %c107, %c5 : index
    %9 = addi %7, %8 : index
    %c6 = constant 6 : index
    %10 = muli %c108, %c6 : index
    %11 = addi %9, %10 : index
    %c7 = constant 7 : index
    %12 = muli %c109, %c7 : index
    %13 = addi %11, %12 : index
    return %12, %13 : index, index
  }
}
