// RUN: circt-opt %s | circt-opt | FileCheck %s

// CHECK-LABEL:  ibis.class @A {
// CHECK-NEXT:    %0 = ibis.this @A 
// CHECK-NEXT:    ibis.port.input @A_in : i1
// CHECK-NEXT:    ibis.port.output @A_out : i1
// CHECK-NEXT:  }

// CHECK-LABEL:   ibis.class @C {
// CHECK-NEXT:    %0 = ibis.this @C 
// CHECK-NEXT:    %1 = ibis.port.input @C_in : i1
// CHECK-NEXT:    %2 = ibis.port.output @C_out : i1
// CHECK-NEXT:    %3 = ibis.instance @a, @A 
// CHECK-NEXT:    %4 = ibis.path [#ibis.step<parent : !ibis.scoperef> : !ibis.scoperef, #ibis.step<child, @a : !ibis.scoperef<@A>> : !ibis.scoperef<@A>]
// CHECK-NEXT:    %5 = ibis.get_port %4, @A_in : !ibis.scoperef<@A> -> !ibis.portref<i1>
// CHECK-NEXT:    ibis.container @D {
// CHECK-NEXT:      %6 = ibis.this @D 
// CHECK-NEXT:      %7 = ibis.path [#ibis.step<parent : !ibis.scoperef<@C>> : !ibis.scoperef<@C>]
// CHECK-NEXT:      %8 = ibis.get_port %7, @C_in : !ibis.scoperef<@C> -> !ibis.portref<i1>
// CHECK-NEXT:      %9 = ibis.get_port %7, @C_out : !ibis.scoperef<@C> -> !ibis.portref<i1>
// CHECK-NEXT:      %true = hw.constant true
// CHECK-NEXT:      ibis.port.write %8, %true : i1
// CHECK-NEXT:      %10 = ibis.port.read %9 : !ibis.portref<i1>
// CHECK-NEXT:      %11 = ibis.path [#ibis.step<parent : !ibis.scoperef> : !ibis.scoperef, #ibis.step<child, @a : !ibis.scoperef<@A>> : !ibis.scoperef<@A>]
// CHECK-NEXT:      %12 = ibis.get_port %11, @A_in : !ibis.scoperef<@A> -> !ibis.portref<i1>
// CHECK-NEXT:      %13 = ibis.get_port %11, @A_out : !ibis.scoperef<@A> -> !ibis.portref<i1>
// CHECK-NEXT:      ibis.port.write %12, %10 : i1
// CHECK-NEXT:      %14 = ibis.port.read %13 : !ibis.portref<i1>
// CHECK-NEXT:    }
// CHECK-NEXT:  }

ibis.class @A {
  %this = ibis.this @A
  ibis.port.input @A_in : i1
  ibis.port.output @A_out : i1
}

ibis.class @C {
  %this = ibis.this @C
  ibis.port.input @C_in : i1
  ibis.port.output @C_out : i1

  // Instantiation
  %a = ibis.instance @a, @A

  %some_child = ibis.path [
    #ibis.step<parent : !ibis.scoperef>,
    #ibis.step<child , @a : !ibis.scoperef<@A>>]

  // Test get parent/child
  %a_in_cp = ibis.get_port %some_child, @A_in : !ibis.scoperef<@A> -> !ibis.portref<i1>

  ibis.container @D {
    %this_d = ibis.this @D
    %parent_C = ibis.path [
      #ibis.step<parent : !ibis.scoperef<@C>>
    ]
    // Test local read/writes
    %c_in_p = ibis.get_port %parent_C, @C_in : !ibis.scoperef<@C> -> !ibis.portref<i1>
    %c_out_p = ibis.get_port %parent_C, @C_out : !ibis.scoperef<@C> -> !ibis.portref<i1>
    %true = hw.constant true
    ibis.port.write %c_in_p, %true : i1
    %c_out = ibis.port.read %c_out_p : !ibis.portref<i1>

    // Test cross-container read/writes
    %a_in_parent = ibis.path [
      #ibis.step<parent : !ibis.scoperef>,
      #ibis.step<child , @a : !ibis.scoperef<@A>>
    ]
    %a_in_p = ibis.get_port %a_in_parent, @A_in : !ibis.scoperef<@A> -> !ibis.portref<i1>
    %a_out_p = ibis.get_port %a_in_parent, @A_out : !ibis.scoperef<@A> -> !ibis.portref<i1>
    ibis.port.write %a_in_p, %c_out : i1
    %a_out = ibis.port.read %a_out_p : !ibis.portref<i1>
  }
}
