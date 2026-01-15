hw.module @ComplexMux(in %Clk : !seq.clock, in %In : !hw.array<5xarray<4xi3>>) {
  %c3_i3 = hw.constant 3 : i3
  %0 = hw.array_get %In[%c3_i3] {sv.namehint = "In__3"} : !hw.array<5xarray<4xi3>>, i3
  %In__3__reg1 = seq.compreg sym @In__3__reg1 %0, %Clk : !hw.array<4xi3>
  %In__3__reg2 = seq.compreg sym @In__3__reg2 %In__3__reg1, %Clk : !hw.array<4xi3>
  %In__3__reg3 = seq.compreg sym @In__3__reg3 %In__3__reg2, %Clk : !hw.array<4xi3>
}
