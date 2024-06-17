// RUN: circt-opt --allow-unregistered-dialect --split-input-file --ibis-clean-selfdrivers %s | FileCheck %s

ibis.design @D {
// CHECK-LABEL:   ibis.container sym @C {
// CHECK:           %[[VAL_0:.*]] = ibis.this <@D::@C>
// CHECK:           %[[VAL_1:.*]] = hw.wire %[[VAL_2:.*]]  : i1
// CHECK:           %[[VAL_3:.*]] = ibis.port.output "out" sym @out : i1
// CHECK:           %[[VAL_2]] = hw.constant true
// CHECK:           ibis.port.write %[[VAL_3]], %[[VAL_1]] : !ibis.portref<out i1>
// CHECK:         }
ibis.container sym @C {
    %this = ibis.this <@D::@C>
    %in = ibis.port.input "in" sym @in : i1
    %out = ibis.port.output "out" sym @out : i1
    %true = hw.constant 1 : i1
    ibis.port.write %in, %true : !ibis.portref<in i1>
    %v = ibis.port.read %in : !ibis.portref<in i1>
    ibis.port.write %out, %v : !ibis.portref<out i1>
}

}

// -----

ibis.design @D {
// CHECK-LABEL:   ibis.container sym @Selfdriver {
// CHECK:           %[[VAL_0:.*]] = ibis.this <@D::@Selfdriver>
// CHECK:           %[[VAL_1:.*]] = hw.wire %[[VAL_2:.*]]  : i1
// CHECK:           %[[VAL_3:.*]] = ibis.port.output "myIn" sym @in : i1
// CHECK:           ibis.port.write %[[VAL_3]], %[[VAL_1]] : !ibis.portref<out i1>
// CHECK:           %[[VAL_2]] = hw.constant true
// CHECK:         }

// CHECK-LABEL:   ibis.container sym @ParentReader {
// CHECK:           %[[VAL_0:.*]] = ibis.this <@D::@ParentReader>
// CHECK:           %[[VAL_1:.*]] = ibis.container.instance @selfdriver, <@D::@Selfdriver>
// CHECK:           %[[VAL_2:.*]] = ibis.get_port %[[VAL_1]], @in : !ibis.scoperef<@D::@Selfdriver> -> !ibis.portref<out i1>
// CHECK:           %[[VAL_3:.*]] = ibis.port.read %[[VAL_2]] : !ibis.portref<out i1>
// CHECK:         }

ibis.container sym @Selfdriver {
  %this = ibis.this <@D::@Selfdriver>
  %in = ibis.port.input "myIn" sym @in : i1
  %true = hw.constant 1 : i1
  ibis.port.write %in, %true : !ibis.portref<in i1>
}

ibis.container sym @ParentReader {
  %this = ibis.this <@D::@ParentReader>
  %selfdriver = ibis.container.instance @selfdriver, <@D::@Selfdriver>
  %in_ref = ibis.get_port %selfdriver, @in : !ibis.scoperef<@D::@Selfdriver> -> !ibis.portref<out i1>
  %in = ibis.port.read %in_ref : !ibis.portref<out i1>
}

}

// -----

ibis.design @D {

ibis.container sym @Foo {
  %this = ibis.this <@D::@Foo>
  %in = ibis.port.input "in" sym @in : i1
}

// CHECK-LABEL:   ibis.container sym @ParentReaderWriter {
// CHECK:           %[[VAL_0:.*]] = ibis.this <@D::@ParentReaderWriter>
// CHECK:           %[[VAL_1:.*]] = ibis.container.instance @f, <@D::@Foo>
// CHECK:           %[[VAL_2:.*]] = ibis.get_port %[[VAL_1]], @in : !ibis.scoperef<@D::@Foo> -> !ibis.portref<in i1>
// CHECK:           "foo.bar"(%[[VAL_3:.*]]) : (i1) -> ()
// CHECK:           %[[VAL_3]] = hw.constant true
// CHECK:           ibis.port.write %[[VAL_2]], %[[VAL_3]] : !ibis.portref<in i1>
// CHECK:         }
ibis.container sym @ParentReaderWriter {
  %this = ibis.this <@D::@ParentReaderWriter>
  %f = ibis.container.instance @f, <@D::@Foo>
  %in_wr_ref = ibis.get_port %f, @in : !ibis.scoperef<@D::@Foo> -> !ibis.portref<in i1>
  %in_rd_ref = ibis.get_port %f, @in : !ibis.scoperef<@D::@Foo> -> !ibis.portref<out i1>
  %v = ibis.port.read %in_rd_ref : !ibis.portref<out i1>
  "foo.bar"(%v) : (i1) -> ()
  %true = hw.constant 1 : i1
  ibis.port.write %in_wr_ref, %true : !ibis.portref<in i1>
}

}
