// RUN: circt-opt --allow-unregistered-dialect --split-input-file --ibis-clean-selfdrivers %s | FileCheck %s

ibis.design @D {
// CHECK-LABEL:   ibis.container @C {
// CHECK:           %[[VAL_0:.*]] = ibis.this <@C>
// CHECK:           %[[VAL_1:.*]] = hw.wire %[[VAL_2:.*]]  : i1
// CHECK:           %[[VAL_3:.*]] = ibis.port.output "out" sym @out : i1
// CHECK:           %[[VAL_2]] = hw.constant true
// CHECK:           ibis.port.write %[[VAL_3]], %[[VAL_1]] : !ibis.portref<out i1>
// CHECK:         }
ibis.container @C {
    %this = ibis.this <@C>
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
// CHECK-LABEL:   ibis.container @Selfdriver {
// CHECK:           %[[VAL_0:.*]] = ibis.this <@Selfdriver>
// CHECK:           %[[VAL_1:.*]] = hw.wire %[[VAL_2:.*]]  : i1
// CHECK:           %[[VAL_3:.*]] = ibis.port.output "in" sym @in : i1
// CHECK:           ibis.port.write %[[VAL_3]], %[[VAL_1]] : !ibis.portref<out i1>
// CHECK:           %[[VAL_2]] = hw.constant true
// CHECK:         }

// CHECK-LABEL:   ibis.container @ParentReader {
// CHECK:           %[[VAL_0:.*]] = ibis.this <@ParentReader>
// CHECK:           %[[VAL_1:.*]] = ibis.container.instance @selfdriver, <@Selfdriver>
// CHECK:           %[[VAL_2:.*]] = ibis.get_port %[[VAL_1]], @in : !ibis.scoperef<@Selfdriver> -> !ibis.portref<out i1>
// CHECK:           %[[VAL_3:.*]] = ibis.port.read %[[VAL_2]] : !ibis.portref<out i1>
// CHECK:         }

ibis.container @Selfdriver {
  %this = ibis.this <@Selfdriver>
  %in = ibis.port.input "in" sym @in : i1
  %true = hw.constant 1 : i1
  ibis.port.write %in, %true : !ibis.portref<in i1>
}

ibis.container @ParentReader {
  %this = ibis.this <@ParentReader>
  %selfdriver = ibis.container.instance @selfdriver, <@Selfdriver>
  %in_ref = ibis.get_port %selfdriver, @in : !ibis.scoperef<@Selfdriver> -> !ibis.portref<out i1>
  %in = ibis.port.read %in_ref : !ibis.portref<out i1>
}

}

// -----

ibis.design @D {

ibis.container @Foo {
  %this = ibis.this <@Foo>
  %in = ibis.port.input "in" sym @in : i1
}

// CHECK-LABEL:   ibis.container @ParentReaderWriter {
// CHECK:           %[[VAL_0:.*]] = ibis.this <@ParentReaderWriter>
// CHECK:           %[[VAL_1:.*]] = ibis.container.instance @f, <@Foo>
// CHECK:           %[[VAL_2:.*]] = ibis.get_port %[[VAL_1]], @in : !ibis.scoperef<@Foo> -> !ibis.portref<in i1>
// CHECK:           "foo.bar"(%[[VAL_3:.*]]) : (i1) -> ()
// CHECK:           %[[VAL_3]] = hw.constant true
// CHECK:           ibis.port.write %[[VAL_2]], %[[VAL_3]] : !ibis.portref<in i1>
// CHECK:         }
ibis.container @ParentReaderWriter {
  %this = ibis.this <@ParentReaderWriter>
  %f = ibis.container.instance @f, <@Foo>
  %in_wr_ref = ibis.get_port %f, @in : !ibis.scoperef<@Foo> -> !ibis.portref<in i1>
  %in_rd_ref = ibis.get_port %f, @in : !ibis.scoperef<@Foo> -> !ibis.portref<out i1>
  %v = ibis.port.read %in_rd_ref : !ibis.portref<out i1>
  "foo.bar"(%v) : (i1) -> ()
  %true = hw.constant 1 : i1
  ibis.port.write %in_wr_ref, %true : !ibis.portref<in i1>
}

}
