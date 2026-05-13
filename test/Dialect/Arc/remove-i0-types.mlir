// RUN: circt-opt --arc-remove-i0-types %s | FileCheck %s

// The i0 argument is removed entirely from the function signature.
// CHECK-LABEL: func.func @TakesI0()
// CHECK-NEXT:    return
// CHECK-NEXT:  }
func.func @TakesI0(%a: i0) {
  return
}

// All i0 operations are dead code and are erased.
// CHECK-LABEL: func.func @DeadCode()
// CHECK-NEXT:    return
// CHECK-NEXT:  }
func.func @DeadCode(%a: i0) {
  %0 = comb.and %a, %a : i0
  %1 = comb.or %0, %a : i0
  return
}

// Both the i0 argument and the i0 result are removed.
// CHECK-LABEL: func.func @ReturnsI0()
// CHECK-NEXT:    return
// CHECK-NEXT:  }
func.func @ReturnsI0(%a: i0) -> i0 {
  return %a : i0
}

// The i0 constant and return value are removed.
// CHECK-LABEL: func.func @ConstantI0()
// CHECK-NEXT:    return
// CHECK-NEXT:  }
func.func @ConstantI0() -> i0 {
  %cst = hw.constant 0 : i0
  return %cst : i0
}

// The i0 index argument is removed, and the single-element array is unwrapped
// to its element type. The array_get is replaced by the element directly.
// CHECK-LABEL: func.func @ArrayGet(%arg0: i32) -> i32
// CHECK-NEXT:    return %arg0 : i32
// CHECK-NEXT:  }
func.func @ArrayGet(%a: !hw.array<1xi32>, %b: i0) -> i32 {
  %0 = hw.array_get %a[%b] : !hw.array<1xi32>, i0
  return %0 : i32
}

// The single-element array create is replaced by its element directly.
// CHECK-LABEL: func.func @ArrayCreate(%arg0: i32) -> i32
// CHECK-NEXT:    return %arg0 : i32
// CHECK-NEXT:  }
func.func @ArrayCreate(%a: i32) -> !hw.array<1xi32> {
  %0 = hw.array_create %a : i32
  return %0 : !hw.array<1xi32>
}

// The single-element array inject is replaced by the injected element; the i0
// index argument is removed.
// CHECK-LABEL: func.func @ArrayInject(%arg0: i32, %arg1: i32) -> i32
// CHECK-NEXT:    return %arg1 : i32
// CHECK-NEXT:  }
func.func @ArrayInject(%a: !hw.array<1xi32>, %b: i0, %c: i32) -> !hw.array<1xi32> {
  %0 = hw.array_inject %a[%b], %c : !hw.array<1xi32>, i0
  return %0 : !hw.array<1xi32>
}

// The i0 argument is removed from both the caller and the callee signature.
// CHECK-LABEL: func.func @Call()
// CHECK-NEXT:    call @TakesI0() : () -> ()
// CHECK-NEXT:    return
// CHECK-NEXT:  }
func.func @Call(%a: i0) {
  func.call @TakesI0(%a) : (i0) -> ()
  return
}

// The i0 return value is removed from both the caller and the callee signature.
// CHECK-LABEL: func.func @Call2()
// CHECK-NEXT:    call @ReturnsI0() : () -> ()
// CHECK-NEXT:    return
// CHECK-NEXT:  }
func.func @Call2(%a: i0) -> i0 {
  %0 = func.call @ReturnsI0(%a) : (i0) -> (i0)
  return %0 : i0
}

// The single-element array inside the struct is unwrapped to its element type.
// CHECK-LABEL: func.func @StructCreate(%arg0: i32) -> !hw.struct<x: i32>
// CHECK-NEXT:    %0 = hw.struct_create (%arg0) : !hw.struct<x: i32>
// CHECK-NEXT:    return %0 : !hw.struct<x: i32>
// CHECK-NEXT:  }
func.func @StructCreate(%a: !hw.array<1xi32>) -> !hw.struct<x: !hw.array<1xi32>> {
  %0 = hw.struct_create (%a) : !hw.struct<x: !hw.array<1xi32>>
  return %0 : !hw.struct<x: !hw.array<1xi32>>
}

// CHECK-LABEL: func.func @StructExtract(%arg0: !hw.struct<x: i32>) -> i32
// CHECK-NEXT:    %x = hw.struct_extract %arg0["x"] : !hw.struct<x: i32>
// CHECK-NEXT:    return %x : i32
// CHECK-NEXT:  }
func.func @StructExtract(%a: !hw.struct<x: i32>) -> i32 {
  %0 = hw.struct_extract %a["x"] : !hw.struct<x: i32>
  return %0 : i32
}

// Mixed i0 and non-i0 arguments: the i0 arguments are removed while the
// non-i0 arguments are preserved.
// CHECK-LABEL: func.func @MixedArgs(%arg0: i32, %arg1: i8) -> i32
// CHECK-NEXT:    return %arg0 : i32
// CHECK-NEXT:  }
func.func @MixedArgs(%a: i0, %b: i32, %c: i0, %d: i8) -> i32 {
  return %b : i32
}

// Mixed i0 and non-i0 return values: the i0 results are removed.
// CHECK-LABEL: func.func @MixedResults(%arg0: i32) -> i32
// CHECK-NEXT:    return %arg0 : i32
// CHECK-NEXT:  }
func.func @MixedResults(%a: i32) -> (i32, i0) {
  %cst = hw.constant 0 : i0
  return %a, %cst : i32, i0
}

// Calling a function with mixed i0 and non-i0 arguments and results.
// CHECK-LABEL: func.func @CallMixed(%arg0: i32) -> i32
// CHECK-NEXT:    %0 = call @MixedResults(%arg0) : (i32) -> i32
// CHECK-NEXT:    return %0 : i32
// CHECK-NEXT:  }
func.func @CallMixed(%a: i32) -> i32 {
  %cst = hw.constant 0 : i0
  %0:2 = func.call @MixedResults(%a) : (i32) -> (i32, i0)
  return %0#0 : i32
}

// A struct with an i0 field: the i0 field type is removed.
// CHECK-LABEL: func.func @StructWithI0Field(%arg0: i32) -> !hw.struct<x: i32>
// CHECK-NEXT:    %0 = hw.struct_create (%arg0) : !hw.struct<x: i32>
// CHECK-NEXT:    return %0 : !hw.struct<x: i32>
// CHECK-NEXT:  }
func.func @StructWithI0Field(%a: i32, %b: i0) -> !hw.struct<x: i32, y: i0> {
  %0 = hw.struct_create (%a, %b) : !hw.struct<x: i32, y: i0>
  return %0 : !hw.struct<x: i32, y: i0>
}

// Extracting a non-i0 field from a struct that also contains an i0 field.
// CHECK-LABEL: func.func @StructExtractNonI0(%arg0: !hw.struct<x: i32>) -> i32
// CHECK-NEXT:    %x = hw.struct_extract %arg0["x"] : !hw.struct<x: i32>
// CHECK-NEXT:    return %x : i32
// CHECK-NEXT:  }
func.func @StructExtractNonI0(%a: !hw.struct<x: i32, y: i0>) -> i32 {
  %0 = hw.struct_extract %a["x"] : !hw.struct<x: i32, y: i0>
  return %0 : i32
}

// Extracting an i0 field from a struct that also contains a non-i0 field.
// CHECK-LABEL: func.func @StructExtractI0(%arg0: !hw.struct<x: i32>)
// CHECK-NEXT:    return
// CHECK-NEXT:  }
func.func @StructExtractI0(%a: !hw.struct<x: i32, y: i0>) -> i0 {
  %0 = hw.struct_extract %a["y"] : !hw.struct<x: i32, y: i0>
  return %0 : i0
}

// A function that has all i0 arguments: all are removed, leaving an empty
// argument list.
// CHECK-LABEL: func.func @AllI0Args()
// CHECK-NEXT:    return
// CHECK-NEXT:  }
func.func @AllI0Args(%a: i0, %b: i0, %c: i0) {
  return
}

// Multi-element array should remain an array type (only single-element arrays
// are unwrapped).
// CHECK-LABEL: func.func @MultiElementArray(%arg0: !hw.array<4xi32>) -> !hw.array<4xi32>
// CHECK-NEXT:    return %arg0 : !hw.array<4xi32>
// CHECK-NEXT:  }
func.func @MultiElementArray(%a: !hw.array<4xi32>) -> !hw.array<4xi32> {
  return %a : !hw.array<4xi32>
}

// Non-i0 types should pass through unchanged.
// CHECK-LABEL: func.func @NoI0(%arg0: i32, %arg1: i64) -> i32
// CHECK-NEXT:    return %arg0 : i32
// CHECK-NEXT:  }
func.func @NoI0(%a: i32, %b: i64) -> i32 {
  return %a : i32
}

// CHECK-LABEL: func.func @ArrayOfArrayOf1
// CHECK-SAME: !hw.array<2xi32>
func.func @ArrayOfArrayOf1(%arg0: !hw.array<2x!hw.array<1xi32>>) ->
  !hw.array<2x!hw.array<1xi32>> {
  return %arg0 : !hw.array<2x!hw.array<1xi32>>
}

// CHECK-LABEL: func.func @StateType
// CHECK-SAME: !arc.state<i32>
func.func @StateType(%arg0: !arc.state<!hw.array<1xi32>>) {
  return
}

// CHECK-LABEL: func.func @AggregateConstantBecomesScalar
// CHECK-SAME: -> i32
// CHECK-NEXT: hw.constant 0 : i32
func.func @AggregateConstantBecomesScalar() -> !hw.array<1xi32> {
  %0 = hw.aggregate_constant [0 : i32] : !hw.array<1xi32>
  return %0 : !hw.array<1xi32>
}

// CHECK-LABEL: func.func @AggregateConstantStaysAggregate
// CHECK-NEXT: hw.aggregate_constant [0 : i32, 1 : i32] : !hw.array<2xi32>
func.func @AggregateConstantStaysAggregate() -> !hw.array<2x!hw.array<1xi32>> {
  %0 = hw.aggregate_constant [[0 : i32], [1 : i32]] : !hw.array<2x!hw.array<1xi32>>
  return %0 : !hw.array<2x!hw.array<1xi32>>
}
