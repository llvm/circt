wire[$msb_lb:0] lb$n_loop = v$n_lb;
wire[$msb_ub:0] ub$n_loop = v$n_ub;
wire[$msb_step:0] step$n_loop = v$n_step;
wire tstart$n_loop = t$n_tstart;

reg[$msb_idx:0] prev_idx$n_loop;
reg[$msb_ub:0] saved_ub$n_loop;
reg[$msb_step:0] saved_step$n_loop;
wire[$msb_idx:0] idx$n_loop = tstart$n_loop? lb$n_loop 
                              : tloop_in$n_loop ? (prev_idx$n_loop+saved_step) 
                              : prev_idx$n_loop;
wire tloop_out$n_loop = tstart$n_loop 
                        || (idx$n_loop < (tstart$n_loop?ub$n_loop
                          :saved_ub$n_loop)?tloop_in$n_loop
                          :1'b0);
always@(posedge $clk) saved_idx$n_loop <= idx$n_loop;
always@(posedge $clk) if (tstart$n_loop) begin
  saved_ub$n_loop   <= ub$n_loop;
  saved_step <= step$n_loop;
end

wire t$n_tloop = tloop_out$n_loop;
wire[$msb_idx:0] v$n_loop = idx$n_loop;

