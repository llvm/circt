#map = affine_map<(d0)[s0, s1] -> (d0 * s1 + s0)>
module attributes {torch.debug_module_name = "DotModule"} {
  handshake.func @forward(%arg0: memref<5xi32, #map>, %arg1: memref<5xi32, #map>, %arg2: none, ...) -> (i32, none) attributes {argNames = ["in0", "in1", "inCtrl"], resNames = ["out0", "outCtrl"]} {
    %0:2 = extmemory[ld = 1, st = 0] (%arg1 : memref<5xi32, #map>) (%addressResults_9) {id = 1 : i32} : (index) -> (i32, none)
    %1:2 = extmemory[ld = 1, st = 0] (%arg0 : memref<5xi32, #map>) (%addressResults) {id = 0 : i32} : (index) -> (i32, none)
    %2:5 = fork [5] %arg2 : none
    %3 = constant %2#3 {value = 0 : i32} : i32
    %4 = constant %2#2 {value = 0 : index} : index
    %5 = constant %2#1 {value = 5 : index} : index
    %6 = constant %2#0 {value = 1 : index} : index
    %7 = buffer [1] seq %17#0 {initValues = [0]} : i1
    %8:5 = fork [5] %7 : i1
    %9 = mux %8#4 [%2#4, %21] : i1, none
    %10 = mux %8#3 [%5, %trueResult] : i1, index
    %11:2 = fork [2] %10 : index
    %12 = mux %8#2 [%6, %19#0] : i1, index
    %13 = mux %8#1 [%4, %24] : i1, index
    %14:2 = fork [2] %13 : index
    %15 = mux %8#0 [%3, %23] : i1, i32
    %16 = arith.cmpi slt, %14#0, %11#0 : index
    %17:6 = fork [6] %16 : i1
    %trueResult, %falseResult = cond_br %17#5, %11#1 : index
    sink %falseResult : index
    %trueResult_0, %falseResult_1 = cond_br %17#4, %12 : index
    sink %falseResult_1 : index
    %trueResult_2, %falseResult_3 = cond_br %17#3, %9 : none
    %trueResult_4, %falseResult_5 = cond_br %17#2, %14#1 : index
    sink %falseResult_5 : index
    %trueResult_6, %falseResult_7 = cond_br %17#1, %15 : i32
    %18:3 = fork [3] %trueResult_4 : index
    %19:2 = fork [2] %trueResult_0 : index
    %20:3 = fork [3] %trueResult_2 : none
    %21 = join %20#1, %1#1, %0#1 : none
    %dataResult, %addressResults = load [%18#2] %1#0, %20#2 : index, i32
    %dataResult_8, %addressResults_9 = load [%18#1] %0#0, %20#0 : index, i32
    %22 = arith.muli %dataResult, %dataResult_8 : i32
    %23 = arith.addi %trueResult_6, %22 : i32
    %24 = arith.addi %18#0, %19#1 : index
    return %falseResult_7, %falseResult_3 : i32, none
  }
}

