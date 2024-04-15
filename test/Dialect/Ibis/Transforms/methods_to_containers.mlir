// RUN: circt-opt --pass-pipeline='builtin.module(ibis.design(ibis.class(ibis-convert-methods-to-containers)))' %s | FileCheck %s

// CHECK-LABEL:   ibis.class @ToContainers {
// CHECK:           %[[VAL_0:.*]] = ibis.this <@ToContainers>
// CHECK:           ibis.container @foo {
// CHECK:             %[[VAL_1:.*]] = ibis.this <@ToContainers::@foo>
// CHECK:             %[[VAL_2:.*]] = ibis.port.input "arg0" sym @arg0 : !dc.value<i32>
// CHECK:             %[[VAL_3:.*]] = ibis.port.read %[[VAL_2]] : !ibis.portref<in !dc.value<i32>>
// CHECK:             %[[VAL_4:.*]] = ibis.port.output "out0" sym @out0 : !dc.value<i32>
// CHECK:             ibis.port.write %[[VAL_4]], %[[VAL_5:.*]] : !ibis.portref<out !dc.value<i32>>
// CHECK:             %[[VAL_6:.*]], %[[VAL_7:.*]] = dc.unpack %[[VAL_3]] : !dc.value<i32>
// CHECK:             %[[VAL_8:.*]]:2 = dc.fork [2] %[[VAL_6]]
// CHECK:             %[[VAL_9:.*]] = dc.pack %[[VAL_8]]#0, %[[VAL_7]] : i32
// CHECK:             %[[VAL_10:.*]] = dc.pack %[[VAL_8]]#1, %[[VAL_7]] : i32
// CHECK:             %[[VAL_5]] = ibis.sblock.dc (%[[VAL_11:.*]] : !dc.value<i32> = %[[VAL_9]], %[[VAL_12:.*]] : !dc.value<i32> = %[[VAL_10]]) -> !dc.value<i32> {
// CHECK:               %[[VAL_13:.*]] = arith.addi %[[VAL_11]], %[[VAL_12]] : i32
// CHECK:               ibis.sblock.return %[[VAL_13]] : i32
// CHECK:             }
// CHECK:           }
// CHECK:         }

ibis.design @foo {
ibis.class @ToContainers {
  %this = ibis.this <@ToContainers> 
  ibis.method.df @foo(%arg0: !dc.value<i32>) -> !dc.value<i32> {
    %token, %output = dc.unpack %arg0 : !dc.value<i32>
    %0:2 = dc.fork [2] %token 
    %1 = dc.pack %0#0, %output : i32
    %2 = dc.pack %0#1, %output : i32
    %3 = ibis.sblock.dc (%arg1 : !dc.value<i32> = %1, %arg2 : !dc.value<i32> = %2) -> !dc.value<i32> {
      %4 = arith.addi %arg1, %arg2 : i32
      ibis.sblock.return %4 : i32
    }
    ibis.return %3 : !dc.value<i32>
  }
}
}
