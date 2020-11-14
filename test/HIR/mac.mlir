hir.func @mac at %t (%a :i32, %b :i32, %c :i32) -> (i32 delay 3){
  %1 = hir.constant 1
  %2 = hir.constant 2

  //old multiplier
  //%m = hir.call @mult_2stage (%a,%b) at %t 
  //      : (i32 , i32) -> (i32 delay 2)

  //new multiplier
  %m = hir.call @mult_3stage (%a,%b) at %t 
        : (i32, i32) -> (i32 delay 3)

  //Delay added to match old multiplier.
  %c2= hir.delay %c by %2 at %t : i32 -> i32
  
  %res = hir.add (%m,%c2) : (i32, i32) -> (i32)
  %res1 = hir.delay %res by %1 at %t offset %2
        : i32 -> i32

  hir.return (%res1) : (i32)
}
