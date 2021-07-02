module  {
  func @convolution(%arg0: memref<16x16xi32>, %arg1: memref<5x5xi32> {hir.bank = [0, 1]}, %arg2: memref<14x14xi32>) {
    %c0_i32 = constant 0 : i32
    %c0 = constant 0 : index
    %c10 = constant 10 : index
    %c1 = constant 1 : index
    scf.for %arg3 = %c0 to %c10 step %c1 {
      %c0_0 = constant 0 : index
      %c10_1 = constant 10 : index
      %c1_2 = constant 1 : index
      scf.for %arg4 = %c0_0 to %c10_1 step %c1_2 {
        memref.store %c0_i32, %arg2[%arg3, %arg4] : memref<14x14xi32>
        %c0_3 = constant 0 : index
        %c4 = constant 4 : index
        %c1_4 = constant 1 : index
        scf.for %arg5 = %c0_3 to %c4 step %c1_4 {
          %c0_5 = constant 0 : index
          %c4_6 = constant 4 : index
          %c1_7 = constant 1 : index
          scf.for %arg6 = %c0_5 to %c4_6 step %c1_7 {
            %0 = addi %arg3, %arg5 : index
            %1 = addi %arg4, %arg6 : index
            %2 = memref.load %arg0[%0, %1] : memref<16x16xi32>
            %3 = memref.load %arg1[%arg5, %arg6] : memref<5x5xi32>
            %4 = muli %2, %3 : i32
            %5 = memref.load %arg2[%arg3, %arg4] : memref<14x14xi32>
            %6 = addi %4, %5 : i32
            memref.store %6, %arg2[%arg3, %arg4] : memref<14x14xi32>
          }{unroll=true}
        }
      }
    }
    return
  }
  func @test(%arg0: memref<16x16xi32>) -> (i32){
  //func @test(%arg0: memref<16x16xi32>){
    %c0 = constant 0 : index
    %c4 = constant 4 : index
    %c1 = constant 1 : index
    scf.for %arg3 = %c0 to %c4 step %c1 {
      %0 = memref.load %arg0[%c0, %c0] : memref<16x16xi32>
    }
    %1 = memref.load %arg0[%c0, %c1] : memref<16x16xi32>
    return %1:i32
    
  }
}

