  dc.func @max(%arg0: !dc.value<i64>, %arg1: !dc.value<i64>, %arg2: !dc.value<i64>, %arg3: !dc.value<i64>, %arg4: !dc.value<i64>, %arg5: !dc.value<i64>, %arg6: !dc.value<i64>, %arg7: !dc.value<i64>) -> !dc.value<i64> {
    %token, %outputs = dc.unpack %arg0 : (!dc.value<i64>) -> i64
    %token_0, %outputs_1 = dc.unpack %arg1 : (!dc.value<i64>) -> i64
    %0 = arith.cmpi slt, %outputs, %outputs_1 : i64
    %1 = arith.select %0, %outputs, %outputs_1 : i64
    %token_2, %outputs_3 = dc.unpack %arg2 : (!dc.value<i64>) -> i64
    %token_4, %outputs_5 = dc.unpack %arg3 : (!dc.value<i64>) -> i64
    %2 = arith.cmpi slt, %outputs_3, %outputs_5 : i64
    %3 = arith.select %2, %outputs_3, %outputs_5 : i64
    %token_6, %outputs_7 = dc.unpack %arg4 : (!dc.value<i64>) -> i64
    %token_8, %outputs_9 = dc.unpack %arg5 : (!dc.value<i64>) -> i64
    %4 = arith.cmpi slt, %outputs_7, %outputs_9 : i64
    %5 = arith.select %4, %outputs_7, %outputs_9 : i64
    %token_10, %outputs_11 = dc.unpack %arg6 : (!dc.value<i64>) -> i64
    %token_12, %outputs_13 = dc.unpack %arg7 : (!dc.value<i64>) -> i64
    %6 = arith.cmpi slt, %outputs_11, %outputs_13 : i64
    %7 = arith.select %6, %outputs_11, %outputs_13 : i64
    %8 = arith.cmpi slt, %1, %3 : i64
    %9 = arith.select %8, %1, %3 : i64
    %10 = arith.cmpi slt, %5, %7 : i64
    %11 = arith.select %10, %5, %7 : i64
    %12 = arith.cmpi slt, %9, %11 : i64
    %13 = dc.join %token, %token_0, %token_2, %token_4, %token_6, %token_8, %token_10, %token_12
    %14 = arith.select %12, %9, %11 : i64
    %15 = dc.pack %13[%14] : (i64) -> !dc.value<i64>
    dc.return %15 : !dc.value<i64>
  }