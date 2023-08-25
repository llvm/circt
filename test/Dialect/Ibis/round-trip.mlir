// RUN: circt-opt %s | circt-opt | FileCheck %s

// CHECK-LABEL:  ibis.class @A {
// CHECK-NEXT:    %this = ibis.this @A 
// CHECK-NEXT:    %in = ibis.port.input @in : i1
// CHECK-NEXT:    %out = ibis.port.output @out : i1
// CHECK-NEXT:  }

// CHECK-LABEL:  ibis.class @C {
// CHECK-NEXT:    %this = ibis.this @C 
// CHECK-NEXT:    %C_in = ibis.port.input @C_in : i1
// CHECK-NEXT:    %C_out = ibis.port.output @C_out : i1
// CHECK-NEXT:    %in_wire, %in_wire.out = ibis.wire.input @in_wire : i1
// CHECK-NEXT:    %true = hw.constant true
// CHECK-NEXT:    %out_wire = ibis.wire.output @out_wire, %true : i1
// CHECK-NEXT:    %a = ibis.instance @a, @A 
// CHECK-NEXT:    ibis.container @D {
// CHECK-NEXT:      %this_0 = ibis.this @D 
// CHECK-NEXT:      %parent = ibis.path [#ibis.step<parent : !ibis.scoperef<@C>> : !ibis.scoperef<@C>]
// CHECK-NEXT:      %parent.C_in.ref = ibis.get_port %parent, @C_in : !ibis.scoperef<@C> -> !ibis.portref<in i1>
// CHECK-NEXT:      %parent.C_out.ref = ibis.get_port %parent, @C_out : !ibis.scoperef<@C> -> !ibis.portref<out i1>
// CHECK-NEXT:      %true_1 = hw.constant true
// CHECK-NEXT:      ibis.port.write %parent.C_in.ref, %true_1 : !ibis.portref<in i1>
// CHECK-NEXT:      %parent.C_out.ref.val = ibis.port.read %parent.C_out.ref : !ibis.portref<out i1>
// CHECK-NEXT:      %parent.a = ibis.path [#ibis.step<parent : !ibis.scoperef> : !ibis.scoperef, #ibis.step<child, @a : !ibis.scoperef<@A>> : !ibis.scoperef<@A>]
// CHECK-NEXT:      %parent.a.in.ref = ibis.get_port %parent.a, @in : !ibis.scoperef<@A> -> !ibis.portref<in i1>
// CHECK-NEXT:      %parent.a.out.ref = ibis.get_port %parent.a, @out : !ibis.scoperef<@A> -> !ibis.portref<out i1>
// CHECK-NEXT:      ibis.port.write %parent.a.in.ref, %parent.C_out.ref.val : !ibis.portref<in i1>
// CHECK-NEXT:      %parent.a.out.ref.val = ibis.port.read %parent.a.out.ref : !ibis.portref<out i1>
// CHECK-NEXT:    }
// CHECK-NEXT:  }

ibis.class @A {
  %this = ibis.this @A
  ibis.port.input @in : i1
  ibis.port.output @out : i1
}

ibis.class @C {
  %this = ibis.this @C
  ibis.port.input @C_in : i1
  ibis.port.output @C_out : i1

  %in_wire, %in_wire.val = ibis.wire.input @in_wire : i1
  %true = hw.constant 1 : i1
  %out_wire = ibis.wire.output @out_wire, %true : i1

  // Instantiation
  %a = ibis.instance @a, @A

  ibis.container @D {
    %this_d = ibis.this @D
    %parent_C = ibis.path [
      #ibis.step<parent : !ibis.scoperef<@C>>
    ]
    // Test local read/writes
    %c_in_p = ibis.get_port %parent_C, @C_in : !ibis.scoperef<@C> -> !ibis.portref<in i1>
    %c_out_p = ibis.get_port %parent_C, @C_out : !ibis.scoperef<@C> -> !ibis.portref<out i1>
    %t = hw.constant true
    ibis.port.write %c_in_p, %t : !ibis.portref<in i1>
    %c_out = ibis.port.read %c_out_p : !ibis.portref<out i1>

    // Test cross-container read/writes
    %A.in_parent = ibis.path [
      #ibis.step<parent : !ibis.scoperef>,
      #ibis.step<child , @a : !ibis.scoperef<@A>>
    ]
    %A.in_p = ibis.get_port %A.in_parent, @in : !ibis.scoperef<@A> -> !ibis.portref<in i1>
    %A.out_p = ibis.get_port %A.in_parent, @out : !ibis.scoperef<@A> -> !ibis.portref<out i1>
    ibis.port.write %A.in_p, %c_out : !ibis.portref<in i1>
    %A.out = ibis.port.read %A.out_p : !ibis.portref<out i1>
  }
}
