// RUN: circt-opt %s -simple-canonicalizer | FileCheck %s

// CHECK-LABEL: @dyn_extract_slice_to_static_extract
// CHECK-SAME: %[[INT:.*]]: i32
// CHECK-SAME: %[[SIG:.*]]: !llhd.sig<i32>
// CHECK-SAME: %[[ARRAY:.*]]: !llhd.array<10xi1>
// CHECK-SAME: %[[ARRAYSIG:.*]]: !llhd.sig<!llhd.array<10xi1>>
func @dyn_extract_slice_to_static_extract(%int : i32, %sig : !llhd.sig<i32>, %array : !llhd.array<10xi1>, %arraysig : !llhd.sig<!llhd.array<10xi1>>)
    -> (i16, i16, !llhd.sig<i16>, !llhd.sig<i16>, !llhd.array<5xi1>, !llhd.array<5xi1>, !llhd.sig<!llhd.array<5xi1>>, !llhd.sig<!llhd.array<5xi1>>) {
  %ind1 = llhd.const 1 : i8
  %ind2 = constant 2 : i8
  // CHECK-NEXT: %[[EXTINT1:.*]] = llhd.extract_slice %[[INT]], 1 : i32 -> i16
  %0 = llhd.dyn_extract_slice %int, %ind1 : (i32, i8) -> i16
  // CHECK-NEXT: %[[EXTINT2:.*]] = llhd.extract_slice %[[INT]], 2 : i32 -> i16
  %1 = llhd.dyn_extract_slice %int, %ind2 : (i32, i8) -> i16
  // CHECK-NEXT: %[[EXTSIG1:.*]] = llhd.extract_slice %[[SIG]], 1 : !llhd.sig<i32> -> !llhd.sig<i16>
  %2 = llhd.dyn_extract_slice %sig, %ind1 : (!llhd.sig<i32>, i8) -> !llhd.sig<i16>
  // CHECK-NEXT: %[[EXTSIG2:.*]] = llhd.extract_slice %[[SIG]], 2 : !llhd.sig<i32> -> !llhd.sig<i16>
  %3 = llhd.dyn_extract_slice %sig, %ind2 : (!llhd.sig<i32>, i8) -> !llhd.sig<i16>
  // CHECK-NEXT: %[[EXTARRAY1:.*]] = llhd.extract_slice %[[ARRAY]], 1 : !llhd.array<10xi1> -> !llhd.array<5xi1>
  %4 = llhd.dyn_extract_slice %array, %ind1 : (!llhd.array<10xi1>, i8) -> !llhd.array<5xi1>
  // CHECK-NEXT: %[[EXTARRAY2:.*]] = llhd.extract_slice %[[ARRAY]], 2 : !llhd.array<10xi1> -> !llhd.array<5xi1>
  %5 = llhd.dyn_extract_slice %array, %ind2 : (!llhd.array<10xi1>, i8) -> !llhd.array<5xi1>
  // CHECK-NEXT: %[[EXTARRAYSIG1:.*]] = llhd.extract_slice %[[ARRAYSIG]], 1 : !llhd.sig<!llhd.array<10xi1>> -> !llhd.sig<!llhd.array<5xi1>>
  %6 = llhd.dyn_extract_slice %arraysig, %ind1 : (!llhd.sig<!llhd.array<10xi1>>, i8) -> !llhd.sig<!llhd.array<5xi1>>
  // CHECK-NEXT: %[[EXTARRAYSIG2:.*]] = llhd.extract_slice %[[ARRAYSIG]], 2 : !llhd.sig<!llhd.array<10xi1>> -> !llhd.sig<!llhd.array<5xi1>>
  %7 = llhd.dyn_extract_slice %arraysig, %ind2 : (!llhd.sig<!llhd.array<10xi1>>, i8) -> !llhd.sig<!llhd.array<5xi1>>

  // CHECK-NEXT: return %[[EXTINT1]], %[[EXTINT2]], %[[EXTSIG1]], %[[EXTSIG2]], %[[EXTARRAY1]], %[[EXTARRAY2]], %[[EXTARRAYSIG1]], %[[EXTARRAYSIG2]] : i16, i16, !llhd.sig<i16>, !llhd.sig<i16>, !llhd.array<5xi1>, !llhd.array<5xi1>, !llhd.sig<!llhd.array<5xi1>>, !llhd.sig<!llhd.array<5xi1>>
  return %0, %1, %2, %3, %4, %5, %6, %7 : i16, i16, !llhd.sig<i16>, !llhd.sig<i16>, !llhd.array<5xi1>, !llhd.array<5xi1>, !llhd.sig<!llhd.array<5xi1>>, !llhd.sig<!llhd.array<5xi1>>
}

// CHECK-LABEL: @dyn_extract_element_to_static_extract
// CHECK-SAME: %[[ARRAY:.*]]: !llhd.array<10xi1>
// CHECK-SAME: %[[ARRAYSIG:.*]]: !llhd.sig<!llhd.array<10xi1>>
func @dyn_extract_element_to_static_extract(%array : !llhd.array<10xi1>, %arraysig : !llhd.sig<!llhd.array<10xi1>>)
    -> (i1, i1, !llhd.sig<i1>, !llhd.sig<i1>) {
  %ind1 = llhd.const 1 : i8
  %ind2 = constant 2 : i8
  // CHECK-NEXT: %[[EXTSIG1:.*]] = llhd.extract_element %[[ARRAY]], 1 : !llhd.array<10xi1> -> i1
  %0 = llhd.dyn_extract_element %array, %ind1 : (!llhd.array<10xi1>, i8) -> i1
  // CHECK-NEXT: %[[EXTSIG2:.*]] = llhd.extract_element %[[ARRAY]], 2 : !llhd.array<10xi1> -> i1
  %1 = llhd.dyn_extract_element %array, %ind2 : (!llhd.array<10xi1>, i8) -> i1
  // CHECK-NEXT: %[[EXTARRAYSIG1:.*]] = llhd.extract_element %[[ARRAYSIG]], 1 : !llhd.sig<!llhd.array<10xi1>> -> !llhd.sig<i1>
  %2 = llhd.dyn_extract_element %arraysig, %ind1 : (!llhd.sig<!llhd.array<10xi1>>, i8) -> !llhd.sig<i1>
  // CHECK-NEXT: %[[EXTARRAYSIG2:.*]] = llhd.extract_element %[[ARRAYSIG]], 2 : !llhd.sig<!llhd.array<10xi1>> -> !llhd.sig<i1>
  %3 = llhd.dyn_extract_element %arraysig, %ind2 : (!llhd.sig<!llhd.array<10xi1>>, i8) -> !llhd.sig<i1>

  // CHECK-NEXT: return %[[EXTSIG1]], %[[EXTSIG2]], %[[EXTARRAYSIG1]], %[[EXTARRAYSIG2]] : i1, i1, !llhd.sig<i1>, !llhd.sig<i1>
  return %0, %1, %2, %3 : i1, i1, !llhd.sig<i1>, !llhd.sig<i1>
}

// CHECK-LABEL: @extract_slice_folding
// CHECK-SAME: %[[INT:.*]]: i32
// CHECK-SAME: %[[SIG:.*]]: !llhd.sig<i32>
// CHECK-SAME: %[[ARRAY:.*]]: !llhd.array<10xi1>
// CHECK-SAME: %[[ARRAYSIG:.*]]: !llhd.sig<!llhd.array<10xi1>>
func @extract_slice_folding(%int : i32, %sig : !llhd.sig<i32>, %array : !llhd.array<10xi1>, %arraysig : !llhd.sig<!llhd.array<10xi1>>)
    -> (i32, !llhd.sig<i32>, !llhd.array<10xi1>, !llhd.sig<!llhd.array<10xi1>>, i8, !llhd.sig<i8>, !llhd.array<2xi1>, !llhd.sig<!llhd.array<2xi1>>,
      i4, !llhd.array<2xi1>, i4, !llhd.array<2xi1>, i4, !llhd.array<2xi1>, i4, !llhd.array<2xi1>, i24, i24, !llhd.array<2xi1>, !llhd.array<1xi1>) {
  // CHECK-NEXT: %[[VAL_4:.*]] = llhd.const 3 : i32
  %c = llhd.const 3 : i32

  %0 = llhd.extract_slice %int, 0 : i32 -> i32
  %1 = llhd.extract_slice %sig, 0 : !llhd.sig<i32> -> !llhd.sig<i32>
  %2 = llhd.extract_slice %array, 0 : !llhd.array<10xi1> -> !llhd.array<10xi1>
  %3 = llhd.extract_slice %arraysig, 0 : !llhd.sig<!llhd.array<10xi1>> -> !llhd.sig<!llhd.array<10xi1>>

  %exint = llhd.extract_slice %int, 2 : i32 -> i16
  %exsig = llhd.extract_slice %sig, 2 : !llhd.sig<i32> -> !llhd.sig<i16>
  %exarray = llhd.extract_slice %array, 2 : !llhd.array<10xi1> -> !llhd.array<5xi1>
  %exarraysig = llhd.extract_slice %arraysig, 2 : !llhd.sig<!llhd.array<10xi1>> -> !llhd.sig<!llhd.array<5xi1>>
  // CHECK-NEXT: %[[VAL_5:.*]] = llhd.extract_slice %[[INT]], 3 : i32 -> i8
  %4 = llhd.extract_slice %exint, 1 : i16 -> i8
  // CHECK-NEXT: %[[VAL_6:.*]] = llhd.extract_slice %[[SIG]], 3 : !llhd.sig<i32> -> !llhd.sig<i8>
  %5 = llhd.extract_slice %exsig, 1 : !llhd.sig<i16> -> !llhd.sig<i8>
  // CHECK-NEXT: %[[VAL_7:.*]] = llhd.extract_slice %[[ARRAY]], 3 : !llhd.array<10xi1> -> !llhd.array<2xi1>
  %6 = llhd.extract_slice %exarray, 1 : !llhd.array<5xi1> -> !llhd.array<2xi1>
  // CHECK-NEXT: %[[VAL_8:.*]] = llhd.extract_slice %[[ARRAYSIG]], 3 : !llhd.sig<!llhd.array<10xi1>> -> !llhd.sig<!llhd.array<2xi1>>
  %7 = llhd.extract_slice %exarraysig, 1 : !llhd.sig<!llhd.array<5xi1>> -> !llhd.sig<!llhd.array<2xi1>>

  // CHECK-NEXT: %[[VAL_9:.*]] = llhd.insert_slice %[[INT]], %[[VAL_5]], 7 : i32, i8
  %insint = llhd.insert_slice %int, %4, 7 : i32, i8
  // CHECK-NEXT: %[[VAL_10:.*]] = llhd.insert_slice %[[ARRAY]], %[[VAL_7]], 6 : !llhd.array<10xi1>, !llhd.array<2xi1>
  %insarray = llhd.insert_slice %array, %6, 6 : !llhd.array<10xi1>, !llhd.array<2xi1>
  // extract only from the part before the inserted slice
  // CHECK-NEXT: %[[VAL_11:.*]] = llhd.extract_slice %[[INT]], 3 : i32 -> i4
  %8 = llhd.extract_slice %insint, 3 : i32 -> i4
  // CHECK-NEXT: %[[VAL_12:.*]] = llhd.extract_slice %[[ARRAY]], 4 : !llhd.array<10xi1> -> !llhd.array<2xi1>
  %9 = llhd.extract_slice %insarray, 4 : !llhd.array<10xi1> -> !llhd.array<2xi1>
  // extract only from the inserted slice
  // CHECK-NEXT: %[[VAL_13:.*]] = llhd.extract_slice %[[INT]], 4 : i32 -> i4
  %10 = llhd.extract_slice %insint, 8 : i32 -> i4
  %11 = llhd.extract_slice %insarray, 6 : !llhd.array<10xi1> -> !llhd.array<2xi1>
  // extract only from the part after the inserted slice
  // CHECK-NEXT: %[[VAL_14:.*]] = llhd.extract_slice %[[INT]], 15 : i32 -> i4
  %12 = llhd.extract_slice %insint, 15 : i32 -> i4
  // CHECK-NEXT: %[[VAL_15:.*]] = llhd.extract_slice %[[ARRAY]], 8 : !llhd.array<10xi1> -> !llhd.array<2xi1>
  %13 = llhd.extract_slice %insarray, 8 : !llhd.array<10xi1> -> !llhd.array<2xi1>
  // extract from both
  // CHECK-NEXT: %[[VAL_16:.*]] = llhd.extract_slice %[[VAL_9]], 6 : i32 -> i4
  %14 = llhd.extract_slice %insint, 6 : i32 -> i4
  // CHECK-NEXT: %[[VAL_17:.*]] = llhd.extract_slice %[[VAL_10]], 5 : !llhd.array<10xi1> -> !llhd.array<2xi1>
  %15 = llhd.extract_slice %insarray, 5 : !llhd.array<10xi1> -> !llhd.array<2xi1>

  // CHECK-NEXT: %[[VAL_18:.*]] = llhd.shr %[[INT]], %[[INT]], %[[VAL_4]] : (i32, i32, i32) -> i32
  %shrint = llhd.shr %int, %int, %c : (i32, i32, i32) -> i32
  // only extract from base value
  // CHECK-NEXT: %[[VAL_19:.*]] = llhd.extract_slice %[[INT]], 8 : i32 -> i24
  %16 = llhd.extract_slice %shrint, 5 : i32 -> i24
  // offset too big
  // CHECK-NEXT: %[[VAL_20:.*]] = llhd.extract_slice %[[VAL_18]], 6 : i32 -> i24
  %17 = llhd.extract_slice %shrint, 6 : i32 -> i24
  // CHECK-NEXT: %[[VAL_21:.*]] = llhd.shr %[[ARRAY]], %[[ARRAY]], %[[VAL_4]] : (!llhd.array<10xi1>, !llhd.array<10xi1>, i32) -> !llhd.array<10xi1>
  %shrarray = llhd.shr %array, %array, %c : (!llhd.array<10xi1>, !llhd.array<10xi1>, i32) -> !llhd.array<10xi1>
  // only extract from base value
  // CHECK-NEXT: %[[VAL_22:.*]] = llhd.extract_slice %[[ARRAY]], 8 : !llhd.array<10xi1> -> !llhd.array<2xi1>
  %18 = llhd.extract_slice %shrarray, 5 : !llhd.array<10xi1> -> !llhd.array<2xi1>
  // offset too big
  // CHECK-NEXT: %[[VAL_23:.*]] = llhd.extract_slice %[[VAL_21]], 7 : !llhd.array<10xi1> -> !llhd.array<1xi1>
  %19 = llhd.extract_slice %shrarray, 7 : !llhd.array<10xi1> -> !llhd.array<1xi1>

  // CHECK-NEXT: return %[[INT]], %[[SIG]], %[[ARRAY]], %[[ARRAYSIG]], %[[VAL_5]], %[[VAL_6]], %[[VAL_7]], %[[VAL_8]], %[[VAL_11]], %[[VAL_12]], %[[VAL_13]], %[[VAL_7]], %[[VAL_14]], %[[VAL_15]], %[[VAL_16]], %[[VAL_17]], %[[VAL_19]], %[[VAL_20]], %[[VAL_22]], %[[VAL_23]] : i32, !llhd.sig<i32>, !llhd.array<10xi1>, !llhd.sig<!llhd.array<10xi1>>, i8, !llhd.sig<i8>, !llhd.array<2xi1>, !llhd.sig<!llhd.array<2xi1>>, i4, !llhd.array<2xi1>, i4, !llhd.array<2xi1>, i4, !llhd.array<2xi1>, i4, !llhd.array<2xi1>, i24, i24, !llhd.array<2xi1>, !llhd.array<1xi1>
  return %0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19
    : i32, !llhd.sig<i32>, !llhd.array<10xi1>, !llhd.sig<!llhd.array<10xi1>>, i8, !llhd.sig<i8>, !llhd.array<2xi1>, !llhd.sig<!llhd.array<2xi1>>,
      i4, !llhd.array<2xi1>, i4, !llhd.array<2xi1>, i4, !llhd.array<2xi1>, i4, !llhd.array<2xi1>, i24, i24, !llhd.array<2xi1>, !llhd.array<1xi1>
}

// CHECK-LABEL: @extract_element_folding
// CHECK-SAME: %[[INT1:.*]]: i32, %[[INT2:.*]]: i32, %[[INT3:.*]]: i32, %[[INT4:.*]]: i32, %[[INT5:.*]]: i32, %[[INT6:.*]]: i32
// CHECK-SAME: %[[SHORT:.*]]: i16
// CHECK-SAME: %[[BYTE:.*]]: i8
func @extract_element_folding(%int1 : i32, %int2 : i32, %int3 : i32, %int4 : i32, %int5 : i32, %int6 : i32, %short : i16, %byte : i8) -> (i8, i32, i32, i32, i32) {
  %amt = llhd.const 1 : i32
  %tup = llhd.tuple %int1, %short, %byte : tuple<i32, i16, i8>
  %arr = llhd.array %int1, %int2, %int3 : !llhd.array<3xi32>
  %hidden = llhd.array %int4, %int5, %int6 : !llhd.array<3xi32>
  %arruni = llhd.array_uniform %int1 : !llhd.array<3xi32>
  %shr = llhd.shr %arr, %hidden, %amt : (!llhd.array<3xi32>, !llhd.array<3xi32>, i32) -> !llhd.array<3xi32>

  %0 = llhd.extract_element %tup, 2 : tuple<i32, i16, i8> -> i8
  %1 = llhd.extract_element %arr, 2 : !llhd.array<3xi32> -> i32
  %2 = llhd.extract_element %arruni, 2 : !llhd.array<3xi32> -> i32
  %3 = llhd.extract_element %shr, 1 : !llhd.array<3xi32> -> i32
  %4 = llhd.extract_element %shr, 2 : !llhd.array<3xi32> -> i32

  // CHECK-NEXT: %[[BYTE]], %[[INT3]], %[[INT1]], %[[INT3]], %[[INT4]] :
  return %0, %1, %2, %3, %4 : i8, i32, i32, i32, i32
}
