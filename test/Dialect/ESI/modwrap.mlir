// RUN: esi-tester %s --test-mod-wrap

rtl.module.extern @OutputChannel(%clk: i1, %bar_ready: i1) -> (%bar: i42, %bar_valid: i1)
rtl.module.extern @InputChannel(%clk: i1, %foo: i23, %foo_valid: i1) -> (%foo_ready: i1, %rawOut: i99) 
rtl.module.extern @Mixed(%clk: i1, %foo: i23, %foo_valid: i1, %bar_ready: i1) ->
                          (%bar: i42, %bar_valid: i1, %foo_ready: i1, %rawOut: i99)
