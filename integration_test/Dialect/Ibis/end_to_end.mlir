// XFAIL: *
// See https://github.com/llvm/circt/issues/6658
// RUN: ibistool -lo %s

ibis.design @foo {

// A class hierarchy with a shared parent, and accessing between the children

ibis.class @C1 {
  %this = ibis.this <@C1>
  %out = ibis.port.output "out" sym @out : i32
  %c0 = hw.constant 42 : i32
  ibis.port.write %out, %c0 : !ibis.portref<out i32>
}

ibis.class @C2 {
  %this = ibis.this <@C2>

  %go_port = ibis.port.input "go" sym @go : i1
  %clk_port = ibis.port.input "clk" sym @clk : !seq.clock
  %rst_port = ibis.port.input "rst" sym @rst : i1
  %done_port = ibis.port.output "done" sym @done : i1
  %out_port = ibis.port.output "out" sym @out : i32

  ibis.container @MyMethod {
    %t = ibis.this <@MyMethod>

    // Grab parent go, clk, reset inputs - note that the requested direction of
    // these are flipped wrt. the defined direction of the ports. The semantics
    // are now that get_port defines the intended usage of the port (in => i'll write to the port, out => i'll read from the port).
    %parent = ibis.path [
      #ibis.step<parent : !ibis.scoperef<@C2>>
    ]
    %go_ref = ibis.get_port %parent, @go : !ibis.scoperef<@C2> -> !ibis.portref<out i1>
    %go = ibis.port.read %go_ref : !ibis.portref<out i1>
    %clk_ref = ibis.get_port %parent, @clk : !ibis.scoperef<@C2> -> !ibis.portref<out !seq.clock>
    %clk = ibis.port.read %clk_ref : !ibis.portref<out !seq.clock>
    %rst_ref = ibis.get_port %parent, @rst : !ibis.scoperef<@C2> -> !ibis.portref<out i1>
    %rst = ibis.port.read %rst_ref : !ibis.portref<out i1>

    // Grab sibling c1's output
    %sibling = ibis.path [
      #ibis.step<parent : !ibis.scoperef>,
      #ibis.step<parent : !ibis.scoperef>,
      #ibis.step<child , @c1 : !ibis.scoperef<@C1>>
    ]
    %sibling_out_ref = ibis.get_port %sibling, @out : !ibis.scoperef<@C1> -> !ibis.portref<out i32>
    %sibling_out = ibis.port.read %sibling_out_ref : !ibis.portref<out i32>

    %res, %done = pipeline.scheduled(%a0 : i32 = %sibling_out) clock(%clk) reset(%rst) go(%go) entryEn(%s0_enable) -> (out : i32) {
        %0 = comb.mul %a0, %a0 : i32
        pipeline.stage ^bb1
      ^bb1(%s1_enable : i1):
        %1 = comb.mul %0, %a0 : i32
        pipeline.stage ^bb2
      ^bb2(%s2_enable : i1):
        %2 = comb.sub %1, %0 : i32
        pipeline.stage ^bb3
      ^bb3(%s3_enable : i1):
        pipeline.return %2 : i32
    }

    // Assign parent done port and output
    %parent_done_ref = ibis.get_port %parent, @done : !ibis.scoperef<@C2> -> !ibis.portref<in i1>
    ibis.port.write %parent_done_ref, %done : !ibis.portref<in i1>
    %parent_out_ref = ibis.get_port %parent, @out : !ibis.scoperef<@C2> -> !ibis.portref<in i32>
    ibis.port.write %parent_out_ref, %res : !ibis.portref<in i32>
  }
}

ibis.class @Parent {
  %this = ibis.this <@Parent>
  %c1 = ibis.instance @c1, <@foo::@C1>
  %c2 = ibis.instance @c2, <@foo::@C2>

  %go = ibis.port.input "go" sym @go : i1
  %clk = ibis.port.input "clk" sym @clk : !seq.clock
  %rst = ibis.port.input "rst" sym @rst : i1

  %done = ibis.port.output "done" sym @done : i1
  %out = ibis.port.output "out" sym @out : i32

  // Wire up to c2
  %go_ref = ibis.get_port %c2, @go : !ibis.scoperef<@C2> -> !ibis.portref<in i1>
  %go_val = ibis.port.read %go : !ibis.portref<in i1>
  ibis.port.write %go_ref, %go_val : !ibis.portref<in i1>

  %clk_ref = ibis.get_port %c2, @clk : !ibis.scoperef<@C2> -> !ibis.portref<in !seq.clock>
  %clk_val = ibis.port.read %clk : !ibis.portref<in !seq.clock>
  ibis.port.write %clk_ref, %clk_val : !ibis.portref<in !seq.clock>

  %rst_ref = ibis.get_port %c2, @rst : !ibis.scoperef<@C2> -> !ibis.portref<in i1>
  %rst_val = ibis.port.read %rst : !ibis.portref<in i1>
  ibis.port.write %rst_ref, %rst_val : !ibis.portref<in i1>

  %done_ref = ibis.get_port %c2, @done : !ibis.scoperef<@C2> -> !ibis.portref<out i1>
  %done_val = ibis.port.read %done_ref : !ibis.portref<out i1>
  ibis.port.write %done, %done_val : !ibis.portref<out i1>

  %out_ref = ibis.get_port %c2, @out : !ibis.scoperef<@C2> -> !ibis.portref<out i32>
  %out_val = ibis.port.read %out_ref : !ibis.portref<out i32>
  ibis.port.write %out, %out_val : !ibis.portref<out i32>
}

}
