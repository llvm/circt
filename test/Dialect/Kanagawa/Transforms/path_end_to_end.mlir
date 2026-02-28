// The ultimate tunneling test - by having 3 intermediate levels for both up-
// and downwards tunneling, we are sure test all the possible combinations of
// tunneling. This is more of a sanity check that everything works as expected.

// RUN: circt-opt --split-input-file --kanagawa-tunneling \
// RUN:    --kanagawa-lower-portrefs --kanagawa-clean-selfdrivers \
// RUN:    --kanagawa-convert-containers-to-hw

kanagawa.design @foo {
kanagawa.container sym @D_up {
  %d = kanagawa.path [
    #kanagawa.step<parent : !kanagawa.scoperef>,
    #kanagawa.step<parent : !kanagawa.scoperef>,
    #kanagawa.step<parent : !kanagawa.scoperef>,
    #kanagawa.step<parent : !kanagawa.scoperef>,
    #kanagawa.step<child, @a_down : !kanagawa.scoperef> : !kanagawa.scoperef,
    #kanagawa.step<child, @b : !kanagawa.scoperef> : !kanagawa.scoperef,
    #kanagawa.step<child, @c : !kanagawa.scoperef> : !kanagawa.scoperef,
    #kanagawa.step<child, @d : !kanagawa.scoperef<@D_down>> : !kanagawa.scoperef<@D_down>]
  // Write an input port
  %clk_ref = kanagawa.get_port %d, @clk_in : !kanagawa.scoperef<@D_down> -> !kanagawa.portref<in i1>
  %clk = hw.constant 1 : i1
  kanagawa.port.write %clk_ref, %clk : !kanagawa.portref<in i1>

  // Read an input port
  %clk_ref_2 = kanagawa.get_port %d, @clk_in : !kanagawa.scoperef<@D_down> -> !kanagawa.portref<out i1>
  %clk_in_val = kanagawa.port.read %clk_ref_2 : !kanagawa.portref<out i1>

  // Read an output port
  %clk_out_ref = kanagawa.get_port %d, @clk_out : !kanagawa.scoperef<@D_down> -> !kanagawa.portref<out i1>
  %clk_out_val = kanagawa.port.read %clk_out_ref : !kanagawa.portref<out i1>
}
kanagawa.container sym @C_up {
  %d = kanagawa.container.instance @d, @D_up
}
kanagawa.container sym @B_up {
  %c = kanagawa.container.instance @c, @C_up
  
}

kanagawa.container sym @A_up {
  %b = kanagawa.container.instance @b, @B_up
}

kanagawa.container sym @Top {
  %a_down = kanagawa.container.instance @a_down, @A_down
  %a_up = kanagawa.container.instance @a_up, @A_up
}
kanagawa.container sym @A_down {
  %b = kanagawa.container.instance @b, @B_down
}
kanagawa.container sym @B_down {
  %c = kanagawa.container.instance @c, @C_down
}
kanagawa.container sym @C_down {
  %d = kanagawa.container.instance @d, @D_down
}
kanagawa.container sym @D_down {
  %clk = kanagawa.port.input "clk_in" sym @clk_in : i1
  %clk_out = kanagawa.port.output "clk_out" sym @clk_out : i1
  %clk.val = kanagawa.port.read %clk : !kanagawa.portref<in i1>
  kanagawa.port.write %clk_out, %clk.val : !kanagawa.portref<out i1>
}
}
