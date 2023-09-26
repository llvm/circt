// RUN: circt-opt --ibis-clean-selfdrivers %s | FileCheck %s

// CHECK-LABEL:   ibis.container @C {
// CHECK:           %[[VAL_0:.*]] = ibis.this @C
// CHECK:           %[[VAL_1:.*]] = hw.wire %[[VAL_2:.*]]  : i1
// CHECK:           %[[VAL_3:.*]] = ibis.port.output @out : i1
// CHECK:           %[[VAL_2]] = hw.constant true
// CHECK:           ibis.port.write %[[VAL_3]], %[[VAL_1]] : !ibis.portref<out i1>
// CHECK:         }
ibis.container @C {
    %this = ibis.this @C
    %in = ibis.port.input @in : i1
    %out = ibis.port.output @out : i1
    %true = hw.constant 1 : i1
    ibis.port.write %in, %true : !ibis.portref<in i1>
    %v = ibis.port.read %in : !ibis.portref<in i1>
    ibis.port.write %out, %v : !ibis.portref<out i1>
}
