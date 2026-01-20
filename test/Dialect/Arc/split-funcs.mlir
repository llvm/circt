// RUN: circt-opt %s --arc-split-funcs=split-bound=2 | FileCheck %s

func.func @Simple(%arg1: i4, %arg2: i4) -> (i4) {
    // CHECK-LABEL: func.func @Simple_split_func0(%arg0: i4, %arg1: i4) -> (i4, i4) {
    // CHECK-NEXT: [[OP0:%.+]] = comb.add %arg0, %arg1 : i4
    // CHECK-NEXT: [[OP1:%.+]] = comb.xor %arg0, %arg1 : i4
    // CHECK-NEXT: return [[OP0]], [[OP1]] : i4, i4
    // CHECK-NEXT: }
    %0 = comb.add %arg1, %arg2 : i4
    %1 = comb.xor %arg1, %arg2 : i4
    // CHECK-LABEL: func.func @Simple_split_func1(%arg0: i4, %arg1: i4) -> (i4, i4) {
    // CHECK-NEXT: [[CALL0:%.+]]:2 = call @Simple_split_func0(%arg0, %arg1) : (i4, i4) -> (i4, i4)
    // CHECK-NEXT: [[OP2:%.+]] = comb.and %arg0, %arg1 : i4
    // CHECK-NEXT: [[OP3:%.+]] = comb.add [[CALL0]]#0, [[CALL0]]#1 : i4
    // CHECK-NEXT: return [[OP2]], [[OP3]] : i4, i4
    // CHECK-NEXT: }
    %2 = comb.and %arg1, %arg2 : i4
    %3 = comb.add %0, %1 : i4
    // CHECK-LABEL: func.func @Simple(%arg0: i4, %arg1: i4) -> i4 {
    // CHECK-NEXT: [[CALL1:%.+]]:2 = call @Simple_split_func1(%arg0, %arg1) : (i4, i4) -> (i4, i4)
    // CHECK-NEXT: [[OP5:%.+]] = comb.add [[CALL1]]#0, [[CALL1]]#1 : i4
    // CHECK-NEXT: return [[OP5]] : i4
    // CHECK-NEXT: }
    %4 = comb.add %2, %3 : i4
    return %4 : i4
}

func.func @LongReuse(%arg1: i4, %arg2: i4) -> (i4) {
    // CHECK-LABEL: func.func @LongReuse_split_func0(%arg0: i4, %arg1: i4) -> (i4, i4) {
    // CHECK-NEXT: [[OP0:%.+]] = comb.add %arg0, %arg1 : i4
    // CHECK-NEXT: [[OP1:%.+]] = comb.xor %arg0, %arg1 : i4
    // CHECK-NEXT: return [[OP0]], [[OP1]] : i4, i4
    // CHECK-NEXT: }
    %0 = comb.add %arg1, %arg2 : i4
    %1 = comb.xor %arg1, %arg2 : i4
    // CHECK-LABEL: func.func @LongReuse_split_func1(%arg0: i4, %arg1: i4) -> (i4, i4) {
    // CHECK-NEXT: [[CALL0:%.+]]:2 = call @LongReuse_split_func0(%arg0, %arg1) : (i4, i4) -> (i4, i4)
    // CHECK-NEXT: [[OP2:%.+]] = comb.and %arg0, %arg1 : i4
    // CHECK-NEXT: [[OP3:%.+]] = comb.or %arg0, %arg1 : i4
    // CHECK-NEXT: return [[CALL0]]#0, [[CALL0]]#1 : i4, i4
    // CHECK-NEXT: }
    %2 = comb.and %arg1, %arg2 : i4
    %3 = comb.or %arg1, %arg2 : i4
    // CHECK-LABEL: func.func @LongReuse(%arg0: i4, %arg1: i4) -> i4 {
    // CHECK-NEXT: [[CALL1:%.+]]:2 = call @LongReuse_split_func1(%arg0, %arg1) : (i4, i4) -> (i4, i4)
    // CHECK-NEXT: [[OP5:%.+]] = comb.add [[CALL1]]#0, [[CALL1]]#1 : i4
    // CHECK-NEXT: return [[OP5]] : i4
    // CHECK-NEXT: }
    %4 = comb.add %0, %1 : i4
    return %4 : i4
}

// Ignore extern functions
// CHECK-LABEL: func.func private @extern()
func.func private @extern()
