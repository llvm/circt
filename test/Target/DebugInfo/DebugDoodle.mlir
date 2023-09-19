// RUN: true

#dbg1 = #dbg.hierarchy { name: "Foo", kind: "module" }
#dbg2 = #dbg.integer_type(32)
#dbg3 = #dbg.variable { name: "a", kind: "port", type: #dbg2, scope: #dbg5 }
#dbg4 = #dbg.variable { name: "b", kind: "port", type: #dbg2, scope: #dbg5 }
#dbg5 = #dbg.scope { name: "io", scope: #dbg1 }
#dbg6 = #dbg.expr { source: "tail(mul(a, a), 32)" }
#dbg7 = #dbg.assign { var: #dbg4, source: "b <= tail(mul(a, a), 32)" }
#dbg8 = #dbg.variable { name: "dummy", kind: "val", type: #dbg2, scope: #dbg1 }

#loc1 = loc(fused<#dbg1>["test/DebugDoodle.fir":2:10])
#loc2 = loc(fused<#dbg3>["test/DebugDoodle.fir":3:11])
#loc3 = loc(fused<#dbg4>["test/DebugDoodle.fir":4:12])
#loc4 = loc(fused<#dbg6>["test/DebugDoodle.fir":6:15])
#loc5 = loc(fused<#dbg7>["test/DebugDoodle.fir":2:10])

hw.module @Foo(%a: i32 loc(#loc2)) -> (b: i32 loc(#loc3)) {
  %0 = comb.mul %a, %a : i32 loc(#loc4)
  %1 = comb.add %0, %c1_32 : i32
  dbg.value %1 loc(#dbg8)
  hw.output %0 : i32 loc(#loc5)
} loc(#loc1)

// ----- 8< ----- output ----- 8< -----

hierarchy(module, Foo) {
  scope(io) {
    var(port, a, int32, svexpr = "Foo.a")
    var(port, b, int32, svexpr = "Foo.b")
  }
  var()
  assign(b, svexpr = "Foo.a * Foo.b")
}

scala line -> sv line
scala line <- sv line

HGL2HDL

HDL2HGL

// How does this look without debug info?
// How can this be given custom debug info?
// How can this represent inlining?
// How does this work with port removal?
