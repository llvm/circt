hir.def @mac at %t (%a:i32 , %b:i32 , %c:i32 delay 2)->(i32 delay 3){
  %1 = hir.constant 1
  %2 = hir.constant 2
  %m = hir.call @mult (%a,%b) at %t : (i32 , i32) -> (i32 delay 2)
  %res = hir.add (%m,%c) : (i32,i32) -> (i32)
  %res1 = hir.delay %res by %1 at %t offset %2: i32 -> i32
  hir.return (%res) : (i32)
}
