// RUN: circt-opt -canonicalize='top-down=true region-simplify=true' %s | FileCheck %s

// CHECK-LABEL: @collapse_or
// CHECK-DAG: [[START_A:%.*]] = hw.constant 99 : i32
// CHECK-DAG: [[END_A:%.*]] = hw.constant 106 : i32
// CHECK-DAG: [[CHECK_START_A:%.*]] = comb.icmp bin ugt %arg, [[START_A]] : i32
// CHECK-DAG: [[CHECK_END_A:%.*]] = comb.icmp bin ult %arg, [[END_A]] : i32
// CHECK-DAG: [[START_B:%.*]] = hw.constant 1009 : i32
// CHECK-DAG: [[END_B:%.*]] = hw.constant 1015 : i32
// CHECK-DAG: [[CHECK_START_B:%.*]] = comb.icmp bin ugt %arg, [[START_B]] : i32
// CHECK-DAG: [[CHECK_END_B:%.*]] = comb.icmp bin ult %arg, [[END_B]] : i32
// CHECK-DAG: [[RANGE_B:%.*]] = comb.and bin [[CHECK_START_B]], [[CHECK_END_B]] : i1
// CHECK-DAG: [[RANGE_A:%.*]] = comb.and bin [[CHECK_START_A]], [[CHECK_END_A]] : i1
// CHECK-DAG: [[RESULT:%.*]] = comb.or bin [[RANGE_A]], [[RANGE_B]] : i1
// CHECK-DAG: hw.output [[RESULT]] : i1
hw.module @collapse_or(in %arg: i32, out cond: i1) {
  %cst0 = hw.constant 100 : i32
  %is0 = comb.icmp bin eq %cst0, %arg : i32
  %cst1 = hw.constant 101 : i32
  %is1 = comb.icmp bin eq %cst1, %arg : i32
  %cst2 = hw.constant 102 : i32
  %is2 = comb.icmp bin eq %arg, %cst2 : i32
  %cst3 = hw.constant 103 : i32
  %is3 = comb.icmp bin eq %arg, %cst3 : i32
  %cst4 = hw.constant 104 : i32
  %is4 = comb.icmp bin eq %arg, %cst4 : i32
  %cst5 = hw.constant 105 : i32
  %is5 = comb.icmp bin eq %arg, %cst5 : i32

  %cst10 = hw.constant 1010 : i32
  %is10 = comb.icmp bin eq %cst10, %arg : i32
  %cst11 = hw.constant 1011 : i32
  %is11 = comb.icmp bin eq %arg, %cst11 : i32
  %cst12 = hw.constant 1012 : i32
  %is12 = comb.icmp bin eq %arg, %cst12 : i32
  %cst13 = hw.constant 1013 : i32
  %is13 = comb.icmp bin eq %arg, %cst13 : i32
  %cst14 = hw.constant 1014 : i32
  %is14 = comb.icmp bin eq %cst14, %arg : i32

  %in_range = comb.or bin %is0, %is1, %is3, %is4, %is2, %is5, %is11, %is10, %is12, %is13, %is14 : i1

  hw.output %in_range : i1
}

// CHECK-LABEL: collapse_or_chain
// CHECK-DAG: [[START:%.*]] = hw.constant 106 : i32
// CHECK-DAG: [[END:%.*]] = hw.constant 99 : i32
// CHECK-DAG: [[CHECK_START:%.*]] = comb.icmp bin ugt %arg, [[END]] : i32
// CHECK-DAG: [[CHECK_END:%.*]] = comb.icmp bin ult %arg, [[START]] : i32
// CHECK-DAG: [[RESULT:%.*]] = comb.and bin [[CHECK_START]], [[CHECK_END]] : i1
// CHECK-DAG: hw.output [[RESULT]] : i1
hw.module @collapse_or_chain(in %arg: i32, out cond: i1) {
  %cst0 = hw.constant 100 : i32
  %is0 = comb.icmp bin eq %cst0, %arg : i32
  %cst1 = hw.constant 101 : i32
  %is1 = comb.icmp bin eq %cst1, %arg : i32
  %cst2 = hw.constant 102 : i32
  %is2 = comb.icmp bin eq %arg, %cst2 : i32
  %cst3 = hw.constant 103 : i32
  %is3 = comb.icmp bin eq %arg, %cst3 : i32
  %cst4 = hw.constant 104 : i32
  %is4 = comb.icmp bin eq %arg, %cst4 : i32
  %cst5 = hw.constant 105 : i32
  %is5 = comb.icmp bin eq %arg, %cst5 : i32

  %is0_1 = comb.or bin %is0, %is1 : i1
  %is2_3 = comb.or bin %is2, %is3 : i1
  %is4_5 = comb.or bin %is4, %is5 : i1
  %is0_3 = comb.or bin %is0_1, %is2_3 : i1
  %is0_5 = comb.or bin %is0_3, %is4_5 : i1

  hw.output %is0_5 : i1
}

// CHECK-LABEL: @merge_ranges
// CHECK-DAG: [[START:%.*]] = hw.constant 300 : i32
// CHECK-DAG: [[END:%.*]] = hw.constant 100 : i32
// CHECK-DAG: [[CHECK_START:%.*]] = comb.icmp bin ugt %arg, [[END]] : i32
// CHECK-DAG: [[CHECK_END:%.*]] = comb.icmp bin ult %arg, [[START]] : i32
// CHECK-DAG: [[RESULT:%.*]] = comb.and bin [[CHECK_START]], [[CHECK_END]] : i1
// CHECK-DAG: hw.output [[RESULT]] : i1
hw.module @merge_ranges(in %arg: i32, out cond: i1) {
  %start_a = hw.constant 100 : i32
  %end_a = hw.constant 200 : i32
  %check_start_a = comb.icmp bin ugt %arg, %start_a : i32
  %check_end_a = comb.icmp bin ult %arg, %end_a : i32
  %in_a = comb.and bin %check_start_a, %check_end_a : i1

  %start_b = hw.constant 199 : i32
  %end_b = hw.constant 300 : i32
  %check_start_b = comb.icmp bin ugt %arg, %start_b : i32
  %check_end_b = comb.icmp bin ult %arg, %end_b : i32
  %in_b = comb.and bin %check_end_b, %check_start_b : i1

  %in_range = comb.or bin %in_a, %in_b : i1

  hw.output %in_range : i1
}

// CHECK-LABEL: @extend_ranges
// CHECK-DAG: [[START:%.*]] = hw.constant 201 : i32
// CHECK-DAG: [[END:%.*]] = hw.constant 100 : i32
// CHECK-DAG: [[CHECK_START:%.*]] = comb.icmp bin ugt %arg, [[END]] : i32
// CHECK-DAG: [[CHECK_END:%.*]] = comb.icmp bin ult %arg, [[START]] : i32
// CHECK-DAG: [[RESULT:%.*]] = comb.and bin [[CHECK_START]], [[CHECK_END]] : i1
// CHECK-DAG: hw.output [[RESULT]] : i1
hw.module @extend_ranges(in %arg: i32, out cond: i1) {
  %start_a = hw.constant 100 : i32
  %end_a = hw.constant 200 : i32
  %check_start_a = comb.icmp bin ugt %arg, %start_a : i32
  %check_end_a = comb.icmp bin ult %arg, %end_a : i32
  %in_a = comb.and bin %check_start_a, %check_end_a : i1

  %elem = hw.constant 200 : i32
  %eq_elem = comb.icmp bin eq %arg, %elem : i32

  %in_range = comb.or bin %in_a, %eq_elem : i1

  hw.output %in_range : i1
}

// CHECK-LABEL: @make_lower_bound
// CHECK-DAG: [[BOUND:%.*]] = hw.constant 6 : i32
// CHECK-DAG: [[RESULT:%.*]] = comb.icmp bin ult %arg, [[BOUND]] : i32
// CHECK-DAG: hw.output [[RESULT]] : i1
hw.module @make_lower_bound(in %arg: i32, out cond: i1) {
  %cst0 = hw.constant 0 : i32
  %is0 = comb.icmp bin eq %cst0, %arg : i32
  %cst1 = hw.constant 1 : i32
  %is1 = comb.icmp bin eq %cst1, %arg : i32
  %cst2 = hw.constant 2 : i32
  %is2 = comb.icmp bin eq %arg, %cst2 : i32
  %cst3 = hw.constant 3 : i32
  %is3 = comb.icmp bin eq %arg, %cst3 : i32
  %cst4 = hw.constant 4 : i32
  %is4 = comb.icmp bin eq %arg, %cst4 : i32
  %cst5 = hw.constant 5 : i32
  %is5 = comb.icmp bin eq %arg, %cst5 : i32

  %is0_1 = comb.or bin %is0, %is1 : i1
  %is2_3 = comb.or bin %is2, %is3 : i1
  %is4_5 = comb.or bin %is4, %is5 : i1
  %is0_3 = comb.or bin %is0_1, %is2_3 : i1
  %is0_5 = comb.or bin %is0_3, %is4_5 : i1

  hw.output %is0_5 : i1
}

// CHECK-LABEL: @merge_with_lower_bound
// CHECK-DAG: [[END:%.*]] = hw.constant 30 : i32
// CHECK-DAG: [[CHECK_END:%.*]] = comb.icmp bin ult %arg, [[END]] : i32
// CHECK-DAG: hw.output [[CHECK_END]] : i1
hw.module @merge_with_lower_bound(in %arg: i32, out cond: i1) {
  %0 = hw.constant 6 : i32
  %1 = comb.icmp bin ult %arg, %0 : i32

  %2 = hw.constant 20 : i32
  %3 = comb.icmp bin ult %arg, %2 : i32

  %4 = hw.constant 10 : i32
  %5 = comb.icmp bin eq %arg, %4 : i32

  %6 = hw.constant 19 : i32
  %7 = hw.constant 30 : i32
  %8 = comb.icmp bin ugt %arg, %6 : i32
  %9 = comb.icmp bin ult %arg, %7 : i32
  %10 = comb.and bin %8, %9 : i1

  %11 = comb.or bin %1, %3, %5, %10 : i1

  hw.output %11 : i1
}

// CHECK-LABEL: @merge_with_upper_bound
// CHECK-DAG: [[CST:%.+]] = hw.constant -5 : i4
// CHECK-DAG: [[RESULT:%.+]] = comb.icmp bin ugt %arg, [[CST]] : i4
// CHECK-DAG: hw.output [[RESULT]] : i1
hw.module @merge_with_upper_bound(in %arg: i4, out cond: i1) {
  %cst15 = hw.constant 15 : i4
  %is15 = comb.icmp bin eq %cst15, %arg : i4
  %cst14 = hw.constant 14 : i4
  %is14 = comb.icmp bin eq %cst14, %arg : i4
  %cst13 = hw.constant 13 : i4
  %is13 = comb.icmp bin eq %arg, %cst13 : i4
  %cst12 = hw.constant 12 : i4
  %is12 = comb.icmp bin eq %arg, %cst12 : i4

  %in_range = comb.or bin %is12, %is13, %is14, %is15 : i1

  hw.output %in_range : i1
}

// CHECK-LABEL: @merge_all
// CHECK-DAG: [[CST:%.+]] = hw.constant true
// CHECK-DAG: hw.output [[CST]] : i1
hw.module @merge_all(in %cond : i2, out out : i1) {
  %c-1_i2 = hw.constant -1 : i2
  %c1_i2 = hw.constant 1 : i2
  %c0_i2 = hw.constant 0 : i2
  %c-2_i2 = hw.constant -2 : i2
  %0 = comb.icmp bin eq %cond, %c0_i2 : i2
  %1 = comb.icmp bin eq %cond, %c1_i2 : i2
  %2 = comb.icmp bin eq %cond, %c-2_i2 : i2
  %3 = comb.icmp bin eq %cond, %c-1_i2 : i2
  %4 = comb.or bin %0, %1, %2, %3 : i1
  hw.output %4 : i1
}

