func.func @Simple(%arg0: i4, %arg1: i4) -> (i4) {
    %0 = comb.add %arg0, %arg1 : i4
    %1 = comb.xor %arg0, %arg1 : i4
    %2 = comb.and %arg0, %arg1 : i4
    %3 = comb.add %0, %1 : i4
    %4 = comb.add %2, %3 : i4
    return %4 : i4
}