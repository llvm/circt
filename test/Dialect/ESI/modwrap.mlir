// RUN: esi-tester %s --test-mod-wrap

rtl.module @Foo(%clk: i1, %foo: i23, %foo_valid: i1) -> (%foo_ready: i1) {
  %rdy = rtl.constant 1 : i1
  rtl.output %rdy : i1
}
