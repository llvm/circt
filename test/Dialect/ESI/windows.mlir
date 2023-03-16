// RUN: circt-opt %s -verify-diagnostics | circt-opt -verify-diagnostics | FileCheck %s

!typeA = !hw.struct<header1: i6, header2: i1, header3: i16>

// CHECK-LABEL:   esi.window @TypeAwin1 into !hw.struct<header1: i6, header2: i1, header3: i16> {
// CHECK-NEXT:      esi.window.frame {
// CHECK-NEXT:        esi.window.field "header1" : i6
// CHECK-NEXT:        esi.window.field "header2" : i1
// CHECK-NEXT:      }
// CHECK-NEXT:      esi.window.frame {
// CHECK-NEXT:        esi.window.field "header3" : i16
// CHECK-NEXT:      }
// CHECK-NEXT:    }
esi.window @TypeAwin1 into !typeA {
  esi.window.frame {
    esi.window.field "header1" : i6
    esi.window.field "header2" : i1
  }
  esi.window.frame {
    esi.window.field "header3" : i16
  }
}


// CHECK-LABEL:   hw.module.extern @TypeAModuleDst(%windowed: !esi.window<@typeAwin1>)
hw.module.extern @TypeAModuleDst(%windowed: !esi.window<@typeAwin1>) 
// CHECK-LABEL:   hw.module.extern @TypeAModuleSrc() -> (windowed: !esi.window<@typeAwin1>)
hw.module.extern @TypeAModuleSrc() -> (windowed: !esi.window<@typeAwin1>) 
