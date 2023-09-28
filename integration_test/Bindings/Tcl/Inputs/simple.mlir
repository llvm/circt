hw.module.extern @ichi(in %a: i2, in %b: i3, out c: i4, out d: i5)

hw.module @owo(out owo_result : i32) {
  %0 = hw.constant 3 : i32
  hw.output %0 : i32
}
hw.module @uwu() { }

hw.module @nya(in %nya_input : i32) {
  hw.instance "uwu1" @uwu() -> ()
}

hw.module @test(out test_result : i32) {
  %myArray1 = sv.wire : !hw.inout<array<42 x i8>>
  %0 = hw.instance "owo1" @owo() -> ("owo_result" : i32)
  hw.instance "nya1" @nya("nya_input": %0 : i32) -> ()
  hw.output %0 : i32
}

hw.module @always() {
  %clock = hw.constant 1 : i1
  %3 = sv.reg : !hw.inout<i1>
  %false = hw.constant 0 : i1
  sv.alwaysff(posedge %clock)  {
    sv.passign %3, %false : i1
  }
  hw.output
}
