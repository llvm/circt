// RUN: circt-opt %s | circt-opt | FileCheck %s

kanagawa.design @foo {

// CHECK-LABEL:  kanagawa.class sym @HighLevel {
// CHECK-NEXT:    kanagawa.var @single : memref<i32>
// CHECK-NEXT:    kanagawa.var @array : memref<10xi32>
// CHECK-NEXT:    kanagawa.method @foo() -> (i32, i32) {
// CHECK-NEXT:      %parent = kanagawa.path [#kanagawa.step<parent : !kanagawa.scoperef<@foo::@HighLevel>> : !kanagawa.scoperef<@foo::@HighLevel>]
// CHECK-NEXT:      %single = kanagawa.get_var %parent, @single : !kanagawa.scoperef<@foo::@HighLevel> -> memref<i32>
// CHECK-NEXT:      %array = kanagawa.get_var %parent, @array : !kanagawa.scoperef<@foo::@HighLevel> -> memref<10xi32>
// CHECK-NEXT:      %alloca = memref.alloca() : memref<i32>
// CHECK-NEXT:      %c32_i32 = hw.constant 32 : i32
// CHECK-NEXT:      %0:2 = kanagawa.sblock (%arg0 : i32 = %c32_i32) -> (i32, i32) attributes {schedule = 1 : i64} {
// CHECK-NEXT:        %1 = memref.load %alloca[] : memref<i32>
// CHECK-NEXT:        memref.store %arg0, %alloca[] : memref<i32>
// CHECK-NEXT:        kanagawa.sblock.return %1, %1 : i32, i32
// CHECK-NEXT:      }
// CHECK-NEXT:      kanagawa.return %0#0, %0#1 : i32, i32
// CHECK-NEXT:    }
// CHECK-NEXT:    kanagawa.method.df @bar(%arg0: none) -> none {
// CHECK-NEXT:      %0 = handshake.join %arg0 : none
// CHECK-NEXT:      kanagawa.return %0 : none
// CHECK-NEXT:    }
// CHECK-NEXT:  }


kanagawa.class sym @HighLevel {
  kanagawa.var @single : memref<i32>
  kanagawa.var @array : memref<10xi32>

  kanagawa.method @foo() -> (i32, i32)  {
    %parent = kanagawa.path [
      #kanagawa.step<parent : !kanagawa.scoperef<@foo::@HighLevel>>
    ]
    %single = kanagawa.get_var %parent, @single : !kanagawa.scoperef<@foo::@HighLevel> -> memref<i32>
    %array = kanagawa.get_var %parent, @array : !kanagawa.scoperef<@foo::@HighLevel> -> memref<10xi32>
    %local = memref.alloca() : memref<i32>
    %c32 = hw.constant 32 : i32
    %out1, %out2 = kanagawa.sblock(%arg : i32 = %c32) -> (i32, i32) attributes {schedule = 1} {
      %v = memref.load %local[] : memref<i32>
      memref.store %arg, %local[] : memref<i32>
      kanagawa.sblock.return %v, %v : i32, i32
    }
    kanagawa.return %out1, %out2 : i32, i32
  }

  kanagawa.method.df @bar(%arg0 : none) -> (none) {
    %0 = handshake.join %arg0 : none
    kanagawa.return %0 : none
  }
}


// CHECK-LABEL:  kanagawa.class sym @A {
// CHECK-NEXT:    %in = kanagawa.port.input "in" sym @in : i1
// CHECK-NEXT:    %out = kanagawa.port.output "out" sym @out : i1
// CHECK-NEXT:    %AnonymousPort = kanagawa.port.input sym @AnonymousPort : i1
// CHECK-NEXT:  }

// CHECK-LABEL:  kanagawa.class sym @LowLevel {
// CHECK-NEXT:    %LowLevel_in = kanagawa.port.input "LowLevel_in" sym @LowLevel_in : i1
// CHECK-NEXT:    %LowLevel_out = kanagawa.port.output "LowLevel_out" sym @LowLevel_out : i1
// CHECK-NEXT:    %in_wire, %in_wire.out = kanagawa.wire.input @in_wire : i1
// CHECK-NEXT:    %true = hw.constant true
// CHECK-NEXT:    %out_wire = kanagawa.wire.output @out_wire, %true : i1
// CHECK-NEXT:    %a = kanagawa.instance @a, <@foo::@A> 
// CHECK-NEXT:    kanagawa.container sym @D {
// CHECK-NEXT:      %parent = kanagawa.path [#kanagawa.step<parent : !kanagawa.scoperef<@foo::@LowLevel>> : !kanagawa.scoperef<@foo::@LowLevel>]
// CHECK-NEXT:      %parent.LowLevel_in.ref = kanagawa.get_port %parent, @LowLevel_in : !kanagawa.scoperef<@foo::@LowLevel> -> !kanagawa.portref<in i1>
// CHECK-NEXT:      %parent.LowLevel_out.ref = kanagawa.get_port %parent, @LowLevel_out : !kanagawa.scoperef<@foo::@LowLevel> -> !kanagawa.portref<out i1>
// CHECK-NEXT:      %true_0 = hw.constant true
// CHECK-NEXT:      kanagawa.port.write %parent.LowLevel_in.ref, %true_0 : !kanagawa.portref<in i1>
// CHECK-NEXT:      %parent.LowLevel_out.ref.val = kanagawa.port.read %parent.LowLevel_out.ref : !kanagawa.portref<out i1>
// CHECK-NEXT:      %parent.a = kanagawa.path [#kanagawa.step<parent : !kanagawa.scoperef> : !kanagawa.scoperef, #kanagawa.step<child, @a : !kanagawa.scoperef<@foo::@A>> : !kanagawa.scoperef<@foo::@A>]
// CHECK-NEXT:      %parent.a.in.ref = kanagawa.get_port %parent.a, @in : !kanagawa.scoperef<@foo::@A> -> !kanagawa.portref<in i1>
// CHECK-NEXT:      %parent.a.out.ref = kanagawa.get_port %parent.a, @out : !kanagawa.scoperef<@foo::@A> -> !kanagawa.portref<out i1>
// CHECK-NEXT:      kanagawa.port.write %parent.a.in.ref, %parent.LowLevel_out.ref.val : !kanagawa.portref<in i1>
// CHECK-NEXT:      %parent.a.out.ref.val = kanagawa.port.read %parent.a.out.ref : !kanagawa.portref<out i1>
// CHECK-NEXT:      hw.instance "foo" @externModule() -> ()
// CHECK-NEXT:    }

// CHECK-NEXT:      kanagawa.container "ThisName" sym @ThisSymbol {
// CHECK-NEXT:      }
// CHECK-NEXT:  }

kanagawa.class sym @A {
  kanagawa.port.input "in" sym @in : i1
  kanagawa.port.output "out" sym @out : i1
  kanagawa.port.input sym @AnonymousPort : i1
}

kanagawa.class sym @LowLevel {
  kanagawa.port.input "LowLevel_in" sym @LowLevel_in : i1
  kanagawa.port.output "LowLevel_out" sym @LowLevel_out : i1

  %in_wire, %in_wire.val = kanagawa.wire.input @in_wire : i1
  %true = hw.constant 1 : i1
  %out_wire = kanagawa.wire.output @out_wire, %true : i1

  // Instantiation
  %a = kanagawa.instance @a, <@foo::@A>

  kanagawa.container sym @D {
    %parent_C = kanagawa.path [
      #kanagawa.step<parent : !kanagawa.scoperef<@foo::@LowLevel>>
    ]
    // Test local read/writes
    %LowLevel_in_p = kanagawa.get_port %parent_C, @LowLevel_in : !kanagawa.scoperef<@foo::@LowLevel> -> !kanagawa.portref<in i1>
    %LowLevel_out_p = kanagawa.get_port %parent_C, @LowLevel_out : !kanagawa.scoperef<@foo::@LowLevel> -> !kanagawa.portref<out i1>
    %t = hw.constant true
    kanagawa.port.write %LowLevel_in_p, %t : !kanagawa.portref<in i1>
    %LowLevel_out = kanagawa.port.read %LowLevel_out_p : !kanagawa.portref<out i1>

    // Test cross-container read/writes
    %A.in_parent = kanagawa.path [
      #kanagawa.step<parent : !kanagawa.scoperef>,
      #kanagawa.step<child , @a : !kanagawa.scoperef<@foo::@A>>
    ]
    %A.in_p = kanagawa.get_port %A.in_parent, @in : !kanagawa.scoperef<@foo::@A> -> !kanagawa.portref<in i1>
    %A.out_p = kanagawa.get_port %A.in_parent, @out : !kanagawa.scoperef<@foo::@A> -> !kanagawa.portref<out i1>
    kanagawa.port.write %A.in_p, %LowLevel_out : !kanagawa.portref<in i1>
    %A.out = kanagawa.port.read %A.out_p : !kanagawa.portref<out i1>

    // Test hw.instance ops inside a container (symbol table usage)
    hw.instance "foo" @externModule() -> ()
  }

  kanagawa.container "ThisName" sym @ThisSymbol {
  }
}

}

hw.module.extern @externModule()
