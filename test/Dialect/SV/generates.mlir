// RUN: circt-opt %s | FileCheck %s
// RUN: circt-opt %s | circt-opt | FileCheck %s

hw.module @Case1<NUM : i8> () -> () {
  %fd = hw.constant 0x80000002 : i32
  sv.generate.case "foo_case" : (#hw.param.decl.ref<"NUM">) [
    case (0) {
      sv.initial {
        sv.fwrite %fd, "case 0\n"
      }
    }
    case (1 : i64) {}
  ]
}

// CHECK-LABEL: hw.module @Case1
// CHECK:         [[FD:%.+]] = hw.constant -2147483646 : i32
// CHECK:         sv.generate.case "foo_case" : (#hw.param.decl.ref<"NUM">)[
// CHECK:         case (0 : i64) {
// CHECK:           sv.initial {
// CHECK:             sv.fwrite [[FD]], "case 0\0A"
// CHECK:           }
// CHECK:         }
// CHECK:         case (1 : i64) {
// CHECK:         }]
