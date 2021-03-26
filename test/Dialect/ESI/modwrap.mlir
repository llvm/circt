// RUN: esi-tester %s --test-mod-wrap | FileCheck %s

rtl.module.extern @OutputChannel(%clk: i1, %bar_ready: i1) -> (%bar: i42, %bar_valid: i1)

// CHECK:  rtl.module @OutputChannel_esi(%clk: i1) -> (%bar: !esi.channel<i42>) {
// CHECK:    %chanOutput, %ready = esi.wrap.vr %wrapped.bar, %wrapped.bar_valid : i42
// CHECK:    %wrapped.bar, %wrapped.bar_valid = rtl.instance "wrapped" @OutputChannel(%clk, %ready) : (i1, i1) -> (i42, i1)
// CHECK:    rtl.output %chanOutput : !esi.channel<i42>

rtl.module.extern @InputChannel(%clk: i1, %foo: i23, %foo_valid: i1) -> (%foo_ready: i1, %rawOut: i99) 

// CHECK:  rtl.module @InputChannel_esi(%clk: i1, %foo: !esi.channel<i23>) -> (%rawOut: i99) {
// CHECK:    %rawOutput, %valid = esi.unwrap.vr %foo, %wrapped.foo_ready : i23
// CHECK:    %wrapped.foo_ready, %wrapped.rawOut = rtl.instance "wrapped" @InputChannel(%clk, %rawOutput, %valid) : (i1, i23, i1) -> (i1, i99)
// CHECK:    rtl.output %wrapped.rawOut : i99

rtl.module.extern @Mixed(%clk: i1, %foo: i23, %foo_valid: i1, %bar_ready: i1) ->
                          (%bar: i42, %bar_valid: i1, %foo_ready: i1, %rawOut: i99)

// CHECK:  rtl.module @Mixed_esi(%clk: i1, %foo: !esi.channel<i23>) -> (%bar: !esi.channel<i42>, %rawOut: i99) {
// CHECK:    %rawOutput, %valid = esi.unwrap.vr %foo, %wrapped.foo_ready : i23
// CHECK:    %chanOutput, %ready = esi.wrap.vr %wrapped.bar, %wrapped.bar_valid : i42
// CHECK:    %wrapped.bar, %wrapped.bar_valid, %wrapped.foo_ready, %wrapped.rawOut = rtl.instance "wrapped" @Mixed(%clk, %rawOutput, %valid, %ready) : (i1, i23, i1, i1) -> (i42, i1, i1, i99)
// CHECK:    rtl.output %chanOutput, %wrapped.rawOut : !esi.channel<i42>, i99
