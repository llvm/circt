// RUN: circt-opt --verify-roundtrip %s | circt-opt | FileCheck %s

// CHECK-LABEL: module
module {

// CHECK-LABEL: func @netType(%arg0: !sv.net<i42>)
func.func @netType(%arg0: !sv.net<i42>) {
 return
}


// CHECK-LABEL: func @varType(%arg0: !sv.var<i42>)
func.func @varType(%arg0: !sv.var<i42>) {
 return
}

// CHECK-LABEL: func @netEnum(%arg0: !sv.net<!hw.enum<Pass, Fail>>)
func.func @netEnum(%arg0: !sv.net<!hw.enum<Pass, Fail>>) {
  return
}

// CHECK-LABEL: func @netPackedArray(%arg0: !sv.net<!hw.array<4xi8>>)
func.func @netPackedArray(%arg0: !sv.net<!hw.array<4xi8>>) {
  return
}

// CHECK-LABEL: func @netPackedStruct(%arg0: !sv.net<!hw.struct<a: i1, b: !hw.array<4xi2>>>)
func.func @netPackedStruct(%arg0: !sv.net<!hw.struct<a: i1, b: !hw.array<4xi2>>>) {
  return
}

// CHECK-LABEL: func @netEqualUnionWidth(%arg0: !sv.net<!hw.union<a: i8, b: i8>>)
func.func @netEqualUnionWidth(%arg0: !sv.net<!hw.union<a: i8, b: i8>>) {
  return
}

// CHECK-LABEL: func @netUnequalUnionWidth(%arg0: !sv.net<!hw.union<a: i16, b: i8>>)
func.func @netUnequalUnionWidth(%arg0: !sv.net<!hw.union<a: i16, b: i8>>) {
  return
}

// CHECK-LABEL: func @netUnpackedDimensions(%arg0: !sv.net<!hw.uarray<4xuarray<8xarray<2xi4>>>>)
func.func @netUnpackedDimensions(%arg0: !sv.net<!hw.uarray<4x!hw.uarray<8x!hw.array<2xi4>>>>) {
  return
}

// CHECK-LABEL: func @netTypeAlias(%arg0: !sv.net<!hw.typealias<@types::@word, !hw.uarray<8xi8>>>)
func.func @netTypeAlias(%arg0: !sv.net<!hw.typealias<@types::@word, !hw.uarray<8xi8>>>) {
  return
}

}

// CHECK-LABEL: func @varString(%arg0: !sv.var<!hw.string>)
func.func @varString(%arg0: !sv.var<!hw.string>) {
  return 
}

// CHECK-LABEL: func @varUnpackedArray(%arg0: !sv.var<!hw.uarray<4xi32>>)
func.func @varUnpackedArray(%arg0: !sv.var<!hw.uarray<4xi32>>) {
  return 
}

// CHECK-LABEL: func @varNestedUnpackedArrays(%arg0: !sv.var<!hw.uarray<4xuarray<8xi32>>>)
func.func @varNestedUnpackedArrays(%arg0: !sv.var<!hw.uarray<4xuarray<8xi32>>>) {
  return
}

// CHECK-LABEL: func @varPackedStruct(%arg0: !sv.var<!hw.struct<a: i1, b: !hw.array<4xi2>>>)
func.func @varPackedStruct(%arg0: !sv.var<!hw.struct<a: i1, b: !hw.array<4xi2>>>) {
  return
}

