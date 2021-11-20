#bram_r = {"rd_latency"= 1}
#bram_w = {"wr_latency"= 1}
#reg_r  = {"rd_latency" = 0}
#reg_w  = {"wr_latency"= 1}
hir.func @floyd_warshall at %t(
%n:i32, 
%path_r:!hir.memref<8x8xi32> ports [#bram_r],
%path_w:!hir.memref<8x8xi32> ports[#bram_w]
){

  %c0_i4 = hw.constant 0:i4
  %c1_i4 = hw.constant 1:i4
  %c2_i4 = hw.constant 2:i4
  %c3_i4 = hw.constant 3:i4
  %c4_i4 = hw.constant 4:i4
  %c5_i4 = hw.constant 5:i4
  %c6_i4 = hw.constant 6:i4
  %c8_i4 = hw.constant 8:i4
  %c9_i4 = hw.constant 9:i4


  hir.for %k :i4 = %c0_i4 to %c8_i4 step %c1_i4  
    iter_time(%tk = %t  +  1 ){
    %k_i3 = comb.extract %k from 0 :(i4)->(i3)

    %tfi = hir.for %i :i4 = %c0_i4  to %c8_i4  step %c1_i4
      iter_time(%ti = %tk  +  1 ){
      %i_i3 = comb.extract %i from 0 :(i4)->(i3)

      %tfj=hir.for %j :i4 = %c0_i4  to %c8_i4  step %c1_i4  
        iter_time(%tj = %ti  +  1 ){
        %j_i3 = comb.extract %j from 0 :(i4)->(i3)

        //load path[i][j]
        %p_i_j = hir.load %path_r[port 0][%i_i3,%j_i3] at %tj
        : !hir.memref<8x8xi32> 

        %p_i_j2 = hir.delay %p_i_j by 2 at %tj : i32 

        //load path[i][k]
        %p_i_k = hir.load %path_r[port 0][%i_i3,%k_i3] at %tj+1
        : !hir.memref<8x8xi32> 

        %p_i_k1 = hir.delay %p_i_k by 1 at %tj+2 : i32 

        //load path[k][j]
        %p_k_j = hir.load %path_r[port 0][%k_i3,%j_i3] at %tj + 2
        : !hir.memref<8x8xi32>

        //sum = add path[i][k]+path[k][j]
        %sum = comb.add %p_i_k1,%p_k_j  : i32

        //cond = hir.lt (path[i][j], sum)
        %cond = comb.icmp ult %p_i_j2, %sum : i32

        //out =hir.select (cond,path[i][j],sum)
        %out = comb.mux %cond, %p_i_j2, %sum  :i32

        //store out to p_reg_r
        hir.store %out to %path_w[port 0][%i_i3,%j_i3] at %tj + 3
        : !hir.memref<8x8xi32>

        hir.next_iter at %tj + 4
      }
      hir.next_iter at %tfj + 1
    }
    hir.next_iter at %tfi + 1
  }
  hir.return
}
