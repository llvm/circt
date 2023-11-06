// RUN: true
// RUasdfN: circt-opt %s --verify-diagnostics | circt-opt | FileCheck %s

//===----------------------------------------------------------------------===//
// Input
//===----------------------------------------------------------------------===//

hw.module @Foo(in %a: i42, out b: i42) {
  %bar.y = hw.instance "bar" @Bar(x: %a: i42) -> (y: i42)
  dbg.variable "a", %a : i42
  dbg.variable "b", %bar.y : i42
  hw.output %bar.y : i42
}

hw.module @Bar(in %x: i42, out y: i42) {
  %c1_i42 = hw.constant 1 : i42
  %0 = comb.mul %x, %x : i42
  %1 = comb.add %0, %c1_i42 : i42
  dbg.variable "x", %x : i42
  dbg.variable "squared", %0 : i42
  dbg.variable "y", %1 : i42
  hw.output %1 : i42
}

// - Foo
//   - a => `Foo.a`
//   - b => `Foo.b`
//   - bar Bar
//     - x => `Bar.x`
//     - squared => `Bar.x*Bar.x`
//     - y => `Bar.y`

//===----------------------------------------------------------------------===//
// Inlining @Bar
//===----------------------------------------------------------------------===//

hw.module @Foo2(in %a: i42, out b: i42) {
  // ----- 8< ----- inlined Bar ----- 8< -----
  %c1_i42 = hw.constant 1 : i42
  %0 = comb.mul %a, %a : i42
  %1 = comb.add %0, %c1_i42 : i42
  %bar_hier = dbg.scope "bar", "Bar2"              // <--- maybe?
  dbg.variable "x", %a scope %bar_hier : i42       // <--- maybe?
  dbg.variable "squared", %0 scope %bar_hier : i42 // <--- maybe?
  dbg.variable %1, "y" scope %bar_hier : i42       // <--- maybe?
  // ----- 8< ----- inlined Bar ----- 8< -----
  dbg.variable "a", %a : i42
  dbg.variable "b", %1 : i42
  hw.output %1 : i42
}

// - Foo2
//   - a => `Foo2.a`
//   - b => `Foo2.b`
//   - bar Bar2
//     - x => `Foo2.a`
//     - squared => `Foo2.a*Foo2.a`
//     - y => `Foo2.a*Foo2.a+1`

//===----------------------------------------------------------------------===//
// Optimizing across ports (comb.add pulled out of @Bar)
//===----------------------------------------------------------------------===//

hw.module @Foo3(in %a: i42, out b: i42) {
  // ----- 8< ----- updated Bar ----- 8< -----
  %bar_hier = dbg.scope "bar", "Bar3" // <--- maybe?
  %bar.T0 = hw.instance "bar" @Bar3(x: %a: i42,
    scope: %bar_hier: !dbg.scope) -> (out T0: i42) // <--- maybe?
  %c1_i42 = hw.constant 1 : i42
  %0 = comb.add %bar.T0, %c1_i42 : i42
  dbg.variable %0, "y" scope %bar_hier : i42 // <--- maybe?
  // ----- 8< ----- updated Bar ----- 8< -----
  dbg.variable "a", %a : i42
  dbg.variable "b", %0 : i42
  hw.output %0 : i42
}

hw.module @Bar3(in %x: i42, out T0: i42, in %scope: !dbg.scope) {
  %0 = comb.mul %x, %x : i42
  dbg.variable "x", %x scope %scope : i42       // <--- maybe?
  dbg.variable "squared", %0 scope %scope : i42 // <--- maybe?
  hw.output %0 : i42
}

// - Foo3
//   - a => `Foo3.a`
//   - b => `Foo3.b`
//   - bar Bar3
//     - x => `Bar3.x`
//     - squared => `Bar3.x*Bar3.x`
//     - y => `Foo3.T0+1`

//===----------------------------------------------------------------------===//
// Chisel Call Stacks
//===----------------------------------------------------------------------===//

hw.module @Foo4(in %a: i42, out b: i42) {
  %scopeIf = dbg.scope "if (enablePow)"
  %scopeForI0 = dbg.scope "for @ i=0" parent %scopeIf
  %scopeForI1 = dbg.scope "for @ i=1" parent %scopeIf
  %scopeIfTail = dbg.scope "if (tailInc)" parent %scopeForI1
  %0 = comb.mul %a, %a : i42
  %1 = comb.mul %0, %a : i42
  %c1_i42 = hw.constant 1 : i42
  %2 = comb.add %1, %c1_i42 : i42
  dbg.variable "x", %0 scope %scopeForI0 : i42
  dbg.variable "x", %1 scope %scopeForI1 : i42
  dbg.variable "inc", %2 scope %scopeIfTail : i42
  hw.output %2 : i42
}

// - Foo4
//   - "if (enablePow)"
//     - "for @ i=0"
//       - x => `Foo4.a*Foo4.a`
//     - "for @ i=1"
//       - x => `(Foo4.a*Foo4.a)*Foo4.a`
//       - "if (tailInc)"
//         - inc => `(Foo4.a*Foo4.a)*Foo4.a+1`
