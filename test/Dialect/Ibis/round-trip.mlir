// RUN: circt-opt %s | circt-opt | FileCheck %s

ibis.design @foo {

// CHECK-LABEL:  ibis.class @HighLevel {
// CHECK-NEXT:    %this = ibis.this <@HighLevel> 
// CHECK-NEXT:    ibis.var @single : memref<i32>
// CHECK-NEXT:    ibis.var @array : memref<10xi32>
// CHECK-NEXT:    ibis.method @foo() -> (i32, i32) {
// CHECK-NEXT:      %parent = ibis.path [#ibis.step<parent : !ibis.scoperef<@HighLevel>> : !ibis.scoperef<@HighLevel>]
// CHECK-NEXT:      %single = ibis.get_var %parent, @single : !ibis.scoperef<@HighLevel> -> memref<i32>
// CHECK-NEXT:      %array = ibis.get_var %parent, @array : !ibis.scoperef<@HighLevel> -> memref<10xi32>
// CHECK-NEXT:      %alloca = memref.alloca() : memref<i32>
// CHECK-NEXT:      %c32_i32 = hw.constant 32 : i32
// CHECK-NEXT:      %0:2 = ibis.sblock (%arg0 : i32 = %c32_i32) -> (i32, i32) attributes {schedule = 1 : i64} {
// CHECK-NEXT:        %1 = memref.load %alloca[] : memref<i32>
// CHECK-NEXT:        memref.store %arg0, %alloca[] : memref<i32>
// CHECK-NEXT:        ibis.sblock.return %1, %1 : i32, i32
// CHECK-NEXT:      }
// CHECK-NEXT:      ibis.return %0#0, %0#1 : i32, i32
// CHECK-NEXT:    }
// CHECK-NEXT:    ibis.method.df @bar(%arg0: none) -> none {
// CHECK-NEXT:      %0 = handshake.join %arg0 : none
// CHECK-NEXT:      ibis.return %0 : none
// CHECK-NEXT:    }
// CHECK-NEXT:  }


ibis.class @HighLevel {
  %this = ibis.this <@HighLevel>
  ibis.var @single : memref<i32>
  ibis.var @array : memref<10xi32>

  ibis.method @foo() -> (i32, i32)  {
    %parent = ibis.path [
      #ibis.step<parent : !ibis.scoperef<@HighLevel>>
    ]
    %single = ibis.get_var %parent, @single : !ibis.scoperef<@HighLevel> -> memref<i32>
    %array = ibis.get_var %parent, @array : !ibis.scoperef<@HighLevel> -> memref<10xi32>
    %local = memref.alloca() : memref<i32>
    %c32 = hw.constant 32 : i32
    %out1, %out2 = ibis.sblock(%arg : i32 = %c32) -> (i32, i32) attributes {schedule = 1} {
      %v = memref.load %local[] : memref<i32>
      memref.store %arg, %local[] : memref<i32>
      ibis.sblock.return %v, %v : i32, i32
    }
    ibis.return %out1, %out2 : i32, i32
  }

  ibis.method.df @bar(%arg0 : none) -> (none) {
    %0 = handshake.join %arg0 : none
    ibis.return %0 : none
  }
}


// CHECK-LABEL:  ibis.class @A {
// CHECK-NEXT:    %this = ibis.this <@A> 
// CHECK-NEXT:    %in = ibis.port.input "in" sym @in : i1
// CHECK-NEXT:    %out = ibis.port.output "out" sym @out : i1
// CHECK-NEXT:    %AnonymousPort = ibis.port.input sym @AnonymousPort : i1
// CHECK-NEXT:  }

// CHECK-LABEL:  ibis.class @LowLevel {
// CHECK-NEXT:    %this = ibis.this <@LowLevel> 
// CHECK-NEXT:    %LowLevel_in = ibis.port.input "LowLevel_in" sym @LowLevel_in : i1
// CHECK-NEXT:    %LowLevel_out = ibis.port.output "LowLevel_out" sym @LowLevel_out : i1
// CHECK-NEXT:    %in_wire, %in_wire.out = ibis.wire.input @in_wire : i1
// CHECK-NEXT:    %true = hw.constant true
// CHECK-NEXT:    %out_wire = ibis.wire.output @out_wire, %true : i1
// CHECK-NEXT:    %a = ibis.instance @a, <@A> 
// CHECK-NEXT:    ibis.container @D {
// CHECK-NEXT:      %this_0 = ibis.this <@LowLevel::@D> 
// CHECK-NEXT:      %parent = ibis.path [#ibis.step<parent : !ibis.scoperef<@LowLevel>> : !ibis.scoperef<@LowLevel>]
// CHECK-NEXT:      %parent.LowLevel_in.ref = ibis.get_port %parent, @LowLevel_in : !ibis.scoperef<@LowLevel> -> !ibis.portref<in i1>
// CHECK-NEXT:      %parent.LowLevel_out.ref = ibis.get_port %parent, @LowLevel_out : !ibis.scoperef<@LowLevel> -> !ibis.portref<out i1>
// CHECK-NEXT:      %true_1 = hw.constant true
// CHECK-NEXT:      ibis.port.write %parent.LowLevel_in.ref, %true_1 : !ibis.portref<in i1>
// CHECK-NEXT:      %parent.LowLevel_out.ref.val = ibis.port.read %parent.LowLevel_out.ref : !ibis.portref<out i1>
// CHECK-NEXT:      %parent.a = ibis.path [#ibis.step<parent : !ibis.scoperef> : !ibis.scoperef, #ibis.step<child, @a : !ibis.scoperef<@A>> : !ibis.scoperef<@A>]
// CHECK-NEXT:      %parent.a.in.ref = ibis.get_port %parent.a, @in : !ibis.scoperef<@A> -> !ibis.portref<in i1>
// CHECK-NEXT:      %parent.a.out.ref = ibis.get_port %parent.a, @out : !ibis.scoperef<@A> -> !ibis.portref<out i1>
// CHECK-NEXT:      ibis.port.write %parent.a.in.ref, %parent.LowLevel_out.ref.val : !ibis.portref<in i1>
// CHECK-NEXT:      %parent.a.out.ref.val = ibis.port.read %parent.a.out.ref : !ibis.portref<out i1>
// CHECK-NEXT:      hw.instance "foo" @externModule() -> ()
// CHECK-NEXT:    }
// CHECK-NEXT:  }

ibis.class @A {
  %this = ibis.this <@A>
  ibis.port.input "in" sym @in : i1
  ibis.port.output "out" sym @out : i1
  ibis.port.input sym @AnonymousPort : i1
}

ibis.class @LowLevel {
  %this = ibis.this <@LowLevel>
  ibis.port.input "LowLevel_in" sym @LowLevel_in : i1
  ibis.port.output "LowLevel_out" sym @LowLevel_out : i1

  %in_wire, %in_wire.val = ibis.wire.input @in_wire : i1
  %true = hw.constant 1 : i1
  %out_wire = ibis.wire.output @out_wire, %true : i1

  // Instantiation
  %a = ibis.instance @a, <@A>

  ibis.container @D {
    %this_d = ibis.this <@LowLevel::@D>
    %parent_C = ibis.path [
      #ibis.step<parent : !ibis.scoperef<@LowLevel>>
    ]
    // Test local read/writes
    %LowLevel_in_p = ibis.get_port %parent_C, @LowLevel_in : !ibis.scoperef<@LowLevel> -> !ibis.portref<in i1>
    %LowLevel_out_p = ibis.get_port %parent_C, @LowLevel_out : !ibis.scoperef<@LowLevel> -> !ibis.portref<out i1>
    %t = hw.constant true
    ibis.port.write %LowLevel_in_p, %t : !ibis.portref<in i1>
    %LowLevel_out = ibis.port.read %LowLevel_out_p : !ibis.portref<out i1>

    // Test cross-container read/writes
    %A.in_parent = ibis.path [
      #ibis.step<parent : !ibis.scoperef>,
      #ibis.step<child , @a : !ibis.scoperef<@A>>
    ]
    %A.in_p = ibis.get_port %A.in_parent, @in : !ibis.scoperef<@A> -> !ibis.portref<in i1>
    %A.out_p = ibis.get_port %A.in_parent, @out : !ibis.scoperef<@A> -> !ibis.portref<out i1>
    ibis.port.write %A.in_p, %LowLevel_out : !ibis.portref<in i1>
    %A.out = ibis.port.read %A.out_p : !ibis.portref<out i1>

    // Test hw.instance ops inside a container (symbol table usage)
    hw.instance "foo" @externModule() -> ()
  }
}

}

hw.module.extern @externModule()
