// RUN: circt-opt %s -verify-diagnostics | circt-opt -verify-diagnostics | FileCheck %s

module {
  rtl.module @B(%a: !esi.channel<i1> {rtl.direction = "in"}) {
  }

  // CHECK-LABEL: rtl.module @B(%arg0: !esi.channel<i1> {rtl.direction = "in", rtl.name = "a"})
}
