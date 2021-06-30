func @convolution(
%img: memref<16x16xi32>, 
%kernel : memref<5x5xi32> {hir.bank=[0,1]}, 
%res : memref<14x14xi32>){

  %0 = constant 0:i32
  affine.for %i = 0 to 10 step 1{
    affine.for %j = 0 to 10 step 1{

      affine.store %0, %res[%i,%j] {sample_attr="blah"}: memref<14x14xi32>//0

      affine.for %k1 = 0 to 4 step 1 {
        affine.for %k2 = 0 to 4 step 1{
          %v_img = affine.load %img[%i+%k1, %j+%k2] : memref<16x16xi32>//1
          %v_kernel = affine.load %kernel[%k1, %k2] : memref<5x5xi32>//2
          %mult = muli %v_img,  %v_kernel : i32
          %v_prev = affine.load %res[%i,%j] : memref<14x14xi32>//3
          %v = addi %mult, %v_prev : i32
          affine.store %v, %res[%i,%j] : memref<14x14xi32>//4
        }
      }
    }
  }

  return 
}

// -----

func @test(
%img: memref<16x16xi32>, 
%kernel : memref<5x5xi32> {hir.bank=[0,1]}, 
%res : memref<14x14xi32>){

  %0 = constant 0:index
  %1 = constant 1:i32
  
  //affine.store %1, %res[%0,%0] : memref<14x14xi32>//0
  affine.for %k1 = 0 to 4 step 1 {
    %v_prev = affine.load %res[%0,%0] : memref<14x14xi32>//3
    %v = addi %1, %v_prev : i32
    affine.store %v, %res[%0,%0] : memref<14x14xi32>//4
  }
  return 
}
