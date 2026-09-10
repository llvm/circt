// RUN: circt-opt --allow-unregistered-dialect --split-input-file --kanagawa-clean-selfdrivers %s | FileCheck %s

kanagawa.design @D {
// CHECK-LABEL:   kanagawa.container sym @C {
// CHECK:           %[[VAL_1:.*]] = hw.wire %[[VAL_2:.*]]  : i1
// CHECK:           %[[VAL_3:.*]] = kanagawa.port.output "out" sym @out : i1
// CHECK:           %[[VAL_2]] = hw.constant true
// CHECK:           kanagawa.port.write %[[VAL_3]], %[[VAL_1]] : !kanagawa.portref<out i1>
// CHECK:         }
kanagawa.container sym @C {
    %in = kanagawa.port.input "in" sym @in : i1
    %out = kanagawa.port.output "out" sym @out : i1
    %true = hw.constant 1 : i1
    kanagawa.port.write %in, %true : !kanagawa.portref<in i1>
    %v = kanagawa.port.read %in : !kanagawa.portref<in i1>
    kanagawa.port.write %out, %v : !kanagawa.portref<out i1>
}

}

// -----

kanagawa.design @D {
// CHECK-LABEL:   kanagawa.container sym @Selfdriver {
// CHECK:           %[[VAL_1:.*]] = hw.wire %[[VAL_2:.*]]  : i1
// CHECK:           %[[VAL_3:.*]] = kanagawa.port.output "myIn" sym @in : i1
// CHECK:           kanagawa.port.write %[[VAL_3]], %[[VAL_1]] : !kanagawa.portref<out i1>
// CHECK:           %[[VAL_2]] = hw.constant true
// CHECK:         }

// CHECK-LABEL:   kanagawa.container sym @ParentReader {
// CHECK:           %[[VAL_1:.*]] = kanagawa.container.instance @selfdriver, <@D::@Selfdriver>
// CHECK:           %[[VAL_2:.*]] = kanagawa.get_port %[[VAL_1]], @in : !kanagawa.scoperef<@D::@Selfdriver> -> !kanagawa.portref<out i1>
// CHECK:           %[[VAL_3:.*]] = kanagawa.port.read %[[VAL_2]] : !kanagawa.portref<out i1>
// CHECK:         }

kanagawa.container sym @Selfdriver {
  %in = kanagawa.port.input "myIn" sym @in : i1
  %true = hw.constant 1 : i1
  kanagawa.port.write %in, %true : !kanagawa.portref<in i1>
}

kanagawa.container sym @ParentReader {
  %selfdriver = kanagawa.container.instance @selfdriver, <@D::@Selfdriver>
  %in_ref = kanagawa.get_port %selfdriver, @in : !kanagawa.scoperef<@D::@Selfdriver> -> !kanagawa.portref<out i1>
  %in = kanagawa.port.read %in_ref : !kanagawa.portref<out i1>
}

}

// -----

kanagawa.design @D {

kanagawa.container sym @Foo {
  %in = kanagawa.port.input "in" sym @in : i1
}

// CHECK-LABEL:   kanagawa.container sym @ParentReaderWriter {
// CHECK:           %[[VAL_1:.*]] = kanagawa.container.instance @f, <@D::@Foo>
// CHECK:           %[[VAL_2:.*]] = kanagawa.get_port %[[VAL_1]], @in : !kanagawa.scoperef<@D::@Foo> -> !kanagawa.portref<in i1>
// CHECK:           "foo.bar"(%[[VAL_3:.*]]) : (i1) -> ()
// CHECK:           %[[VAL_3]] = hw.constant true
// CHECK:           kanagawa.port.write %[[VAL_2]], %[[VAL_3]] : !kanagawa.portref<in i1>
// CHECK:         }
kanagawa.container sym @ParentReaderWriter {
  %f = kanagawa.container.instance @f, <@D::@Foo>
  %in_wr_ref = kanagawa.get_port %f, @in : !kanagawa.scoperef<@D::@Foo> -> !kanagawa.portref<in i1>
  %in_rd_ref = kanagawa.get_port %f, @in : !kanagawa.scoperef<@D::@Foo> -> !kanagawa.portref<out i1>
  %v = kanagawa.port.read %in_rd_ref : !kanagawa.portref<out i1>
  "foo.bar"(%v) : (i1) -> ()
  %true = hw.constant 1 : i1
  kanagawa.port.write %in_wr_ref, %true : !kanagawa.portref<in i1>
}

}
