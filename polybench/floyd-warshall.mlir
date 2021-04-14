#bram_r = {"rd"= 1}
#bram_w = {"wr"= 1}
#reg_r  = {"rd" = 0}
#reg_w  = {"wr"= 1}

hir.func @floyd_warshall at %t(
%n:i32, 
%path_r:!hir.memref<8x8xi32,#bram_r>,
%path_w:!hir.memref<8x8xi32,#bram_w>,
){

  %0 = hir.constant (0):!hir.const
  %1 = hir.constant (1):!hir.const
  %4 = hir.constant (4):!hir.const
  %5 = hir.constant (5):!hir.const
  %6 = hir.constant (6):!hir.const
  %8 = hir.constant (8):!hir.const
  %9 = hir.constant (9):!hir.const


  hir.for %i :i32 = %0 :!hir.const to %8 :!hir.const step %1 :!hir.const 
    iter_time(%ti = %t  +  %1 ){
    
    %tfj=hir.for %j :i32 = %0 :!hir.const to %8 :!hir.const step %1 :!hir.const 
      iter_time(%tj = %ti  +  %1 ){

      //load path[i][j]
      %p_i_j = hir.load %path[%i,%j] at %tj

      %tfk=hir.for %k :i32 = %0 :!hir.const to %8 :!hir.const step %1 :!hir.const 
        iter_time(%tk = %tj  +  %1 ){
        : !hir.memref<8x8xi32, #bram_r>[i32,i32] -> i32

        //load p_i_j_reg

        //load path[i][k]

        //load path[k][j]

        //sum = add path[i][k]+path[k][j]

        //cond = hir.lt (path[k][j], sum)
        //out =hir.select (cond,path[i][j],sum)
        //store out to p_i_j_reg

        hir.yield at %tk + %2
        }
       hir.yield at %tfk + %1
    }

    hir.yield at %tfj+%1
  }

  hir.return
}
