#bram_r = {"rd"= 1}
#bram_w = {"wr"= 1}
#reg_r  = {"rd" = 0}
#reg_w  = {"wr"= 1}

hir.func @line_buffer at %t(
%inp_t : !hir.group<!hir.time>,
%inp : !hir.group<f32>,
%outp_t :!hir.group<send !hir.time>,
%outp : !hir.array<send 2x2xf32>){

  %buff_r,%buff_w = hir.alloca("bram") :!hir.memref<2x16xf32,[1], #bram_r>,
                    !hir.memref<2x16xf32,[1], #bram_w>

  %wndw_r,%wndw_w = hir.alloca("bram") :!hir.memref<2x2xf32,[0,1], #reg_r>,
  !hir.memref<2x2xf32,[0,1], #reg_w>

  %0 = hir.constant (0):!hir.const
  %1 = hir.constant (1):!hir.const
  %K1Minus1 = hir.constant (1):!hir.const
  %Ni = hir.constant (16):!hir.const
  %Nj = hir.constant (16):!hir.const

  hir.for %i :i32 = %0 :!hir.const to %Ni :!hir.const step %1 :!hir.const 
    iter_time(%ti = %t  +  %1 ){

    %tf=hir.for %j :i32 = %0 :!hir.const to %Nj :!hir.const step %1 :!hir.const 
      iter_time(%tj = %ti  +  %1 ){

      //read the new input from stream.
      %tv = hir.recv %inp_t[%0] at %t : !hir.group<!hir.time>[!hir.const] -> !hir.time
      %v = hir.recv %inp[%0] at %tv : !hir.group<f32>[!hir.const] -> f32

      %v1 = hir.delay %v by %1 at %tv: f32 -> f32

      //update line buffer and window.
      hir.unroll_for %k1 = 0 to 1 step 1 iter_time(%tk1 = %tv){
        hir.yield at %tk1 
        %k1Plus1 = hir.add (%k1,%1) :(!hir.const, !hir.const) -> (!hir.const)
        %val = hir.load %buff_r[%k1Plus1,%j] at %tk1
          :!hir.memref<2x16xf32,[1], #bram_r>[!hir.const,i32] -> f32
        hir.store %val to %buff_w[%k1,%j] at %tk1  +  %1 
          :(f32, !hir.memref<2x16xf32, [1], #bram_w>[!hir.const,i32])
        hir.store %val to %wndw_w[%k1,%0] at %tk1  +  %1 
          :(f32, !hir.memref<2x2xf32, [0,1], #reg_w>[!hir.const,!hir.const])
        hir.send %val to %outp[%k1,%0] at %tk1 + %1
          :f32 to
          !hir.array<send 2x2xf32>[!hir.const,!hir.const]
      }

      //insert the new input from stream.
      hir.store %v1 to %buff_w[%K1Minus1,%j] at %tv + %1
        :(f32, !hir.memref<2x16xf32, [1], #bram_w>[!hir.const,i32])
      hir.store %v1 to %wndw_w[%K1Minus1,%0] at %tv + %1
        :(f32, !hir.memref<2x2xf32, [0,1], #reg_w>[!hir.const,!hir.const])

      hir.send %v1 to %outp[%K1Minus1,%0] at %tv + %1
        :f32 to
        !hir.array<send 2x2xf32>[!hir.const,!hir.const]

      %t_send = hir.delay %tv by %1 at %tv :!hir.time -> !hir.time
      hir.send %t_send to %outp_t[%0] at %t_send
        :!hir.time to
        !hir.group<send !hir.time>[!hir.const]

      // shift the window.
      hir.unroll_for %k1 = 0 to 2 step 1 iter_time(%tk1 = %tv){
        hir.yield at %tk1 
        hir.unroll_for %k2 = 0 to 1 step 1 iter_time(%tk2 = %tk1){
          hir.yield at %tk2
          %val = hir.load %wndw_r[%k1,%k2] at %tk2 
            :!hir.memref<2x2xf32, [0,1], #reg_r>[!hir.const,!hir.const] -> f32
          %k2Plus1 = hir.add(%k2,%1) :(!hir.const, !hir.const) -> (!hir.const)
          hir.store %val to %wndw_w[%k1,%k2Plus1] at %tk2 + %1
            :(f32, !hir.memref<2x2xf32, [0,1], #reg_w> [!hir.const,!hir.const])
          hir.send %val to %outp[%k1,%k2Plus1] at %tk2 + %1
            :f32 to
            !hir.array<send 2x2xf32>[!hir.const,!hir.const]
        }
      }
      hir.yield at %tv + %1
    }
    hir.yield at %tf + %1
  }
  hir.return
}
