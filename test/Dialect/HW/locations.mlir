// RUN: circt-opt %s | circt-opt | FileCheck %s -check-prefix BASIC
// RUN: circt-opt %s --mlir-print-debuginfo | circt-opt --mlir-print-debuginfo --mlir-print-local-scope | FileCheck %s -check-prefix DEBUG

// Basic IR parser+printer round tripping test. The debug locations should only
// be printed when the required, and they should be accurately parsed back in.

hw.module @test0(in %input: i7, out output: i7) { hw.output %input : i7 }
hw.module @test1(in %input: i7 {hw.arg = "arg"}, out output: i7 {hw.res = "res"}) { hw.output %input : i7 }
hw.module @test2(in %input: i7 loc("arg"), out output: i7 loc("res")) { hw.output %input : i7 }
hw.module @test3(in %input: i7 {hw.arg = "arg"} loc("arg"), out output: i7 {hw.res = "res"} loc("res")) { hw.output %input : i7 }
hw.module.extern @test4(in %input: i7, out output: i7)
hw.module.extern @test5(in %input: i7 {hw.arg = "arg"}, out output: i7 {hw.res = "res"})
hw.module.extern @test6(in %input: i7 loc("arg"), out output: i7 loc("res"))
hw.module.extern @test7(in %input: i7 {hw.arg = "arg"} loc("arg"), out output: i7 {hw.res = "res"} loc("res"))
sv.func @test8(in %input: i7, out output: i7) { sv.return %input : i7 }
sv.func @test9(in %input: i7 {hw.arg = "arg"}, out output: i7 {hw.res = "res"}) { sv.return %input : i7 }
sv.func @test10(in %input: i7 loc("arg"), out output: i7 loc("res")) { sv.return %input : i7 }
sv.func @test11(in %input: i7 {hw.arg = "arg"} loc("arg"), out output: i7 {hw.res = "res"} loc("res")) { sv.return %input : i7 }
sv.func private @test12(in %input: i7, out output: i7)
sv.func private @test13(in %input: i7 {hw.arg = "arg"}, out output: i7 {hw.res = "res"})
sv.func private @test14(in %input: i7 loc("arg"), out output: i7 loc("res"))
sv.func private @test15(in %input: i7 {hw.arg = "arg"} loc("arg"), out output: i7 {hw.res = "res"} loc("res"))

// BASIC: hw.module @test0(in %input : i7, out output : i7)
// BASIC: hw.module @test1(in %input : i7 {hw.arg = "arg"}, out output : i7 {hw.res = "res"})
// BASIC: hw.module @test2(in %input : i7, out output : i7)
// BASIC: hw.module @test3(in %input : i7 {hw.arg = "arg"}, out output : i7 {hw.res = "res"})
// BASIC: hw.module.extern @test4(in %input : i7, out output : i7)
// BASIC: hw.module.extern @test5(in %input : i7 {hw.arg = "arg"}, out output : i7 {hw.res = "res"})
// BASIC: hw.module.extern @test6(in %input : i7, out output : i7)
// BASIC: hw.module.extern @test7(in %input : i7 {hw.arg = "arg"}, out output : i7 {hw.res = "res"})
// BASIC: sv.func @test8(in %input : i7, out output : i7)
// BASIC: sv.func @test9(in %input : i7 {hw.arg = "arg"}, out output : i7 {hw.res = "res"})
// BASIC: sv.func @test10(in %input : i7, out output : i7)
// BASIC: sv.func @test11(in %input : i7 {hw.arg = "arg"}, out output : i7 {hw.res = "res"})
// BASIC: sv.func private @test12(in %input : i7, out output : i7)
// BASIC: sv.func private @test13(in %input : i7 {hw.arg = "arg"}, out output : i7 {hw.res = "res"})
// BASIC: sv.func private @test14(in %input : i7, out output : i7)
// BASIC: sv.func private @test15(in %input : i7 {hw.arg = "arg"}, out output : i7 {hw.res = "res"})

// DEBUG: hw.module @test0(in %input : i7 loc({{.+}}), out output : i7 loc({{.+}}))
// DEBUG: hw.module @test1(in %input : i7 {hw.arg = "arg"} loc({{.+}}), out output : i7 {hw.res = "res"} loc({{.+}}))
// DEBUG: hw.module @test2(in %input : i7 loc("arg"), out output : i7 loc("res"))
// DEBUG: hw.module @test3(in %input : i7 {hw.arg = "arg"} loc("arg"), out output : i7 {hw.res = "res"} loc("res"))
// DEBUG: hw.module.extern @test4(in %input : i7 loc({{.+}}), out output : i7 loc({{.+}}))
// DEBUG: hw.module.extern @test5(in %input : i7 {hw.arg = "arg"} loc({{.+}}), out output : i7 {hw.res = "res"} loc({{.+}}))
// DEBUG: hw.module.extern @test6(in %input : i7 loc("arg"), out output : i7 loc("res"))
// DEBUG: hw.module.extern @test7(in %input : i7 {hw.arg = "arg"} loc("arg"), out output : i7 {hw.res = "res"} loc("res"))
// DEBUG: sv.func @test8(in %input : i7 loc({{.+}}), out output : i7 loc({{.+}}))
// DEBUG: sv.func @test9(in %input : i7 {hw.arg = "arg"} loc({{.+}}), out output : i7 {hw.res = "res"} loc({{.+}}))
// DEBUG: sv.func @test10(in %input : i7 loc("arg"), out output : i7 loc("res"))
// DEBUG: sv.func @test11(in %input : i7 {hw.arg = "arg"} loc("arg"), out output : i7 {hw.res = "res"} loc("res"))
// DEBUG: sv.func private @test12(in %input : i7 loc({{.+}}), out output : i7 loc({{.+}}))
// DEBUG: sv.func private @test13(in %input : i7 {hw.arg = "arg"} loc({{.+}}), out output : i7 {hw.res = "res"} loc({{.+}}))
// DEBUG: sv.func private @test14(in %input : i7 loc("arg"), out output : i7 loc("res"))
// DEBUG: sv.func private @test15(in %input : i7 {hw.arg = "arg"} loc("arg"), out output : i7 {hw.res = "res"} loc("res"))
