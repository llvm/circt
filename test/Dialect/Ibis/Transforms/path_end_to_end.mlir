// The ultimate tunneling test - by having 3 intermediate levels for both up-
// and downwards tunneling, we are sure test all the possible combinations of
// tunneling. This is more of a sanity check that everything works as expected.

// RUN: circt-opt --split-input-file --ibis-tunneling \
// RUN:    --ibis-lower-portrefs --ibis-clean-selfdrivers \
// RUN:    --ibis-convert-containers-to-hw

ibis.design @foo {
ibis.container @D_up {
  %this = ibis.this <@D_up>
  %d = ibis.path [
    #ibis.step<parent : !ibis.scoperef>,
    #ibis.step<parent : !ibis.scoperef>,
    #ibis.step<parent : !ibis.scoperef>,
    #ibis.step<parent : !ibis.scoperef>,
    #ibis.step<child, @a_down : !ibis.scoperef> : !ibis.scoperef,
    #ibis.step<child, @b : !ibis.scoperef> : !ibis.scoperef,
    #ibis.step<child, @c : !ibis.scoperef> : !ibis.scoperef,
    #ibis.step<child, @d : !ibis.scoperef<@D_down>> : !ibis.scoperef<@D_down>]
  // Write an input port
  %clk_ref = ibis.get_port %d, @clk_in : !ibis.scoperef<@D_down> -> !ibis.portref<in i1>
  %clk = hw.constant 1 : i1
  ibis.port.write %clk_ref, %clk : !ibis.portref<in i1>

  // Read an input port
  %clk_ref_2 = ibis.get_port %d, @clk_in : !ibis.scoperef<@D_down> -> !ibis.portref<out i1>
  %clk_in_val = ibis.port.read %clk_ref_2 : !ibis.portref<out i1>

  // Read an output port
  %clk_out_ref = ibis.get_port %d, @clk_out : !ibis.scoperef<@D_down> -> !ibis.portref<out i1>
  %clk_out_val = ibis.port.read %clk_out_ref : !ibis.portref<out i1>
}
ibis.container @C_up {
  %this = ibis.this <@C_up>
  %d = ibis.container.instance @d, @D_up
}
ibis.container @B_up {
  %this = ibis.this <@B_up>
  %c = ibis.container.instance @c, @C_up
  
}

ibis.container @A_up {
  %this = ibis.this <@A_up>
  %b = ibis.container.instance @b, @B_up
}

ibis.container @Top {
  %this = ibis.this <@Top>
  %a_down = ibis.container.instance @a_down, @A_down
  %a_up = ibis.container.instance @a_up, @A_up
}
ibis.container @A_down {
  %this = ibis.this <@A_down>
  %b = ibis.container.instance @b, @B_down
}
ibis.container @B_down {
  %this = ibis.this <@B_down>
  %c = ibis.container.instance @c, @C_down
}
ibis.container @C_down {
  %this = ibis.this <@C_down>
  %d = ibis.container.instance @d, @D_down
}
ibis.container @D_down {
  %this = ibis.this <@D_down>
  %clk = ibis.port.input "clk_in" sym @clk_in : i1
  %clk_out = ibis.port.output "clk_out" sym @clk_out : i1
  %clk.val = ibis.port.read %clk : !ibis.portref<in i1>
  ibis.port.write %clk_out, %clk.val : !ibis.portref<out i1>
}
}
