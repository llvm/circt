// RUN: circt-opt %s | circt-opt | FileCheck %s

// CHECK-LABEL:  ibis.class @A(%this) {
// CHECK-NEXT:    ibis.port.input @A_in : i1
// CHECK-NEXT:    ibis.port.output @A_out : i1
// CHECK-NEXT:  }

// CHECK-LABEL:  ibis.class @C(%this) {
// CHECK-NEXT:    ibis.port.input @C_in : i1
// CHECK-NEXT:    ibis.port.output @C_out : i1
// CHECK-NEXT:    %0 = ibis.instance @a, @A 
// CHECK-NEXT:    %1 = ibis.get_parent_of_ref %0 : !ibis.classref<@A>
// CHECK-NEXT:    %2 = ibis.get_instance_in_ref @a : @A in %this : !ibis.classref<@C> 
// CHECK-NEXT:    %3 = ibis.get_port %0, @A_in : !ibis.classref<@A> -> !ibis.portref<i1>
// CHECK-NEXT:    %4 = ibis.get_instance_in_ref @a : @A in %1 : !ibis.classref 
// CHECK-NEXT:    ibis.container @D {
// CHECK-NEXT:      %5 = ibis.get_port %this, @C_in : !ibis.classref<@C> -> !ibis.portref<i1>
// CHECK-NEXT:      %6 = ibis.get_port %this, @C_out : !ibis.classref<@C> -> !ibis.portref<i1>
// CHECK-NEXT:      %true = hw.constant true
// CHECK-NEXT:      ibis.port.write %6, %true : i1
// CHECK-NEXT:      %7 = ibis.port.read %5 : !ibis.portref<i1>
// CHECK-NEXT:      %8 = ibis.get_port %0, @A_in : !ibis.classref<@A> -> !ibis.portref<i1>
// CHECK-NEXT:      %9 = ibis.get_port %0, @A_out : !ibis.classref<@A> -> !ibis.portref<i1>
// CHECK-NEXT:      ibis.port.write %8, %7 : i1
// CHECK-NEXT:      %10 = ibis.port.read %9 : !ibis.portref<i1>
// CHECK-NEXT:    }
// CHECK-NEXT:  }

ibis.class @A(%this) {
  ibis.port.input @A_in : i1
  ibis.port.output @A_out : i1
}

ibis.class @C(%this) {
  ibis.port.input @C_in : i1
  ibis.port.output @C_out : i1

  // Instantiation
  %a = ibis.instance @a, @A

  // Test get parent/child
  %parent = ibis.get_parent_of_ref %a : !ibis.classref<@A>
  %child = ibis.get_instance_in_ref @a : @A in %this : !ibis.classref<@C>
  %a_in_cp = ibis.get_port %a, @A_in : !ibis.classref<@A> -> !ibis.portref<i1>

  // Test siblings
  %sibling = ibis.get_instance_in_ref @a : @A in %parent : !ibis.classref

  ibis.container @D {
    // Test local read/writes
    %c_in_p = ibis.get_port %this, @C_in : !ibis.classref<@C> -> !ibis.portref<i1>
    %c_out_p = ibis.get_port %this, @C_out : !ibis.classref<@C> -> !ibis.portref<i1>
    %true = hw.constant true
    ibis.port.write %c_out_p, %true : i1
    %c_out = ibis.port.read %c_in_p : !ibis.portref<i1>

    // Test cross-container read/writes
    %a_in_p = ibis.get_port %a, @A_in : !ibis.classref<@A> -> !ibis.portref<i1>
    %a_out_p = ibis.get_port %a, @A_out : !ibis.classref<@A> -> !ibis.portref<i1>
    ibis.port.write %a_in_p, %c_out : i1
    %a_out = ibis.port.read %a_out_p : !ibis.portref<i1>
  }
}
