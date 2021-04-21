#bram_r = {"rd"= 1}
#bram_w = {"wr"= 1}
#reg_r  = {"rd" = 0}
#reg_w  = {"wr"= 1}

hir.func @floyd_warshall at %t(
%n:i32, 
%path_r:!hir.memref<8x8xi32,#bram_r>,
%path_w:!hir.memref<8x8xi32,#bram_w>
){

  %0 = hir.constant (0):!hir.const
  %1 = hir.constant (1):!hir.const
  %2 = hir.constant (2):!hir.const
  %3 = hir.constant (3):!hir.const
  %4 = hir.constant (4):!hir.const
  %5 = hir.constant (5):!hir.const
  %6 = hir.constant (6):!hir.const
  %8 = hir.constant (8):!hir.const
  %9 = hir.constant (9):!hir.const


  hir.for %k :i32 = %0 :!hir.const to %8 :!hir.const step %1 :!hir.const 
  iter_time(%tk = %t  +  %1 ){

    %tfi = hir.for %i :i32 = %0 :!hir.const to %8 :!hir.const step %1 :!hir.const 
    iter_time(%ti = %tk  +  %1 ){

      %tfj=hir.for %j :i32 = %0 :!hir.const to %8 :!hir.const step %1 :!hir.const 
      iter_time(%tj = %ti  +  %1 ){

        //load path[i][j]
        %p_i_j = hir.load %path_r[%i,%j] at %tj
        : !hir.memref<8x8xi32, #bram_r>[i32,i32] -> i32

        %p_i_j2 = hir.delay %p_i_j by %2 at %tj : i32 -> i32

        //load path[i][k]
        %p_i_k = hir.load %path_r[%i,%k] at %tj+%1
        :!hir.memref<8x8xi32, #bram_r>[i32,i32] -> i32

        %p_i_k1 = hir.delay %p_i_k by %1 at %tk+%1 : i32 -> i32

        //load path[k][j]
        %p_k_j = hir.load %path_r[%k,%j] at %tj + %2
        :!hir.memref<8x8xi32, #bram_r>[i32,i32] -> i32

        //sum = add path[i][k]+path[k][j]
        %sum = hir.add(%p_i_k1,%p_k_j) : (i32,i32)->(i32)

        //cond = hir.lt (path[i][j], sum)
        %cond = hir.lt (%p_i_j2, %sum) : (i32,i32) -> (i1)

        //out =hir.select (cond,path[i][j],sum)
        %out = hir.call @mux(%cond, %p_i_j2, %sum) at %tj + %3 
        : !hir.func<(i1, i32,i32) -> (i32)>

        //store out to p_reg_r
        hir.store %out to %path_w[%i,%j] at %tj+%3
        : (i32, !hir.memref<8x8xi32, #bram_w>[i32,i32])

        hir.yield at %tj + %4
      }
      hir.yield at %tfj + %1
    }
    hir.yield at %tfi+%1
  }
  hir.return
}
