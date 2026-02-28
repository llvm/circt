// XFAIL: *
// See https://github.com/llvm/circt/issues/6658
// RUN: kanagawatool -lo %s

kanagawa.design @foo {

// A class hierarchy with a shared parent, and accessing between the children

kanagawa.class sym @C1 {
  %out = kanagawa.port.output "out" sym @out : i32
  %c0 = hw.constant 42 : i32
  kanagawa.port.write %out, %c0 : !kanagawa.portref<out i32>
}

kanagawa.class sym @C2 {
  %go_port = kanagawa.port.input "go" sym @go : i1
  %clk_port = kanagawa.port.input "clk" sym @clk : !seq.clock
  %rst_port = kanagawa.port.input "rst" sym @rst : i1
  %done_port = kanagawa.port.output "done" sym @done : i1
  %out_port = kanagawa.port.output "out" sym @out : i32

  kanagawa.container sym @MyMethod {
    // Grab parent go, clk, reset inputs - note that the requested direction of
    // these are flipped wrt. the defined direction of the ports. The semantics
    // are now that get_port defines the intended usage of the port (in => i'll write to the port, out => i'll read from the port).
    %parent = kanagawa.path [
      #kanagawa.step<parent : !kanagawa.scoperef<@foo::@C2>>
    ]
    %go_ref = kanagawa.get_port %parent, @go : !kanagawa.scoperef<@foo::@C2> -> !kanagawa.portref<out i1>
    %go = kanagawa.port.read %go_ref : !kanagawa.portref<out i1>
    %clk_ref = kanagawa.get_port %parent, @clk : !kanagawa.scoperef<@foo::@C2> -> !kanagawa.portref<out !seq.clock>
    %clk = kanagawa.port.read %clk_ref : !kanagawa.portref<out !seq.clock>
    %rst_ref = kanagawa.get_port %parent, @rst : !kanagawa.scoperef<@foo::@C2> -> !kanagawa.portref<out i1>
    %rst = kanagawa.port.read %rst_ref : !kanagawa.portref<out i1>

    // Grab sibling c1's output
    %sibling = kanagawa.path [
      #kanagawa.step<parent : !kanagawa.scoperef>,
      #kanagawa.step<parent : !kanagawa.scoperef>,
      #kanagawa.step<child , @c1 : !kanagawa.scoperef<@foo::@C1>>
    ]
    %sibling_out_ref = kanagawa.get_port %sibling, @out : !kanagawa.scoperef<@foo::@C1> -> !kanagawa.portref<out i32>
    %sibling_out = kanagawa.port.read %sibling_out_ref : !kanagawa.portref<out i32>

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
    %parent_done_ref = kanagawa.get_port %parent, @done : !kanagawa.scoperef<@foo::@C2> -> !kanagawa.portref<in i1>
    kanagawa.port.write %parent_done_ref, %done : !kanagawa.portref<in i1>
    %parent_out_ref = kanagawa.get_port %parent, @out : !kanagawa.scoperef<@foo::@C2> -> !kanagawa.portref<in i32>
    kanagawa.port.write %parent_out_ref, %res : !kanagawa.portref<in i32>
  }
}

kanagawa.class sym @Parent {
  %c1 = kanagawa.instance @c1, <@foo::@C1>
  %c2 = kanagawa.instance @c2, <@foo::@C2>

  %go = kanagawa.port.input "go" sym @go : i1
  %clk = kanagawa.port.input "clk" sym @clk : !seq.clock
  %rst = kanagawa.port.input "rst" sym @rst : i1

  %done = kanagawa.port.output "done" sym @done : i1
  %out = kanagawa.port.output "out" sym @out : i32

  // Wire up to c2
  %go_ref = kanagawa.get_port %c2, @go : !kanagawa.scoperef<@foo::@C2> -> !kanagawa.portref<in i1>
  %go_val = kanagawa.port.read %go : !kanagawa.portref<in i1>
  kanagawa.port.write %go_ref, %go_val : !kanagawa.portref<in i1>

  %clk_ref = kanagawa.get_port %c2, @clk : !kanagawa.scoperef<@foo::@C2> -> !kanagawa.portref<in !seq.clock>
  %clk_val = kanagawa.port.read %clk : !kanagawa.portref<in !seq.clock>
  kanagawa.port.write %clk_ref, %clk_val : !kanagawa.portref<in !seq.clock>

  %rst_ref = kanagawa.get_port %c2, @rst : !kanagawa.scoperef<@foo::@C2> -> !kanagawa.portref<in i1>
  %rst_val = kanagawa.port.read %rst : !kanagawa.portref<in i1>
  kanagawa.port.write %rst_ref, %rst_val : !kanagawa.portref<in i1>

  %done_ref = kanagawa.get_port %c2, @done : !kanagawa.scoperef<@foo::@C2> -> !kanagawa.portref<out i1>
  %done_val = kanagawa.port.read %done_ref : !kanagawa.portref<out i1>
  kanagawa.port.write %done, %done_val : !kanagawa.portref<out i1>

  %out_ref = kanagawa.get_port %c2, @out : !kanagawa.scoperef<@foo::@C2> -> !kanagawa.portref<out i32>
  %out_val = kanagawa.port.read %out_ref : !kanagawa.portref<out i32>
  kanagawa.port.write %out, %out_val : !kanagawa.portref<out i32>
}

}
