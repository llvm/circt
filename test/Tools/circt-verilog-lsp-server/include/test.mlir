module {
    hw.module @test(out out: i1) {
        %0 = hw.constant true
        hw.output %0 : i1
    }
}
