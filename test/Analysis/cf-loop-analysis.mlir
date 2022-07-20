// RUN: circt-opt --test-cf-loop-analysis %s --allow-unregistered-dialect --split-input-file | FileCheck %s

// CHECK-LABEL: func.func @simple_loop
func.func @simple_loop(%n: i64) {
  %c0 = arith.constant 1 : i64
  cf.br ^0(%c0 : i64)
^0(%i: i64):
// CHECK: "block.info"() {loopInfo = ["header", "inLoop"]}
  %cond = arith.cmpi eq, %i, %n : i64
  cf.cond_br %cond, ^2, ^1
^1:
// CHECK: "block.info"() {loopInfo = ["latch", "inLoop"]}
  %c1 = arith.constant 1 : i64
  %ni = arith.addi %i, %c1 : i64
  cf.br ^0(%ni: i64)
^2:
// CHECK: "block.info"() {loopInfo = ["exit"]}
  return
}

// -----

// CHECK-LABEL: func.func @multi_latch
func.func @multi_latch(%n: i64) {
  %c0 = arith.constant 1 : i64
  cf.br ^0(%c0 : i64)
^0(%i: i64):
// CHECK: "block.info"() {loopInfo = ["header", "inLoop"]}
  %cond = arith.cmpi eq, %i, %n : i64
  cf.cond_br %cond, ^3, ^1
^1:
// CHECK: "block.info"() {loopInfo = ["latch", "inLoop"]}
  %c1 = arith.constant 1 : i64
  %ni = arith.addi %i, %c1 : i64
  cf.cond_br %cond, ^0(%ni: i64), ^2
^2:
// CHECK: "block.info"() {loopInfo = ["latch", "inLoop"]}
  cf.br ^0(%ni: i64)
^3:
// CHECK: "block.info"() {loopInfo = ["exit"]}
  return
}

// -----

// CHECK-LABEL: func.func @multiple_loops
func.func @multiple_loops(%n: i64) {
  %c0 = arith.constant 1 : i64
  %c1 = arith.constant 1 : i64
  cf.br ^0(%c0 : i64)
^0(%i: i64):
// CHECK: "block.info"() {loopInfo = ["header", "inLoop"]}
  %cond = arith.cmpi eq, %i, %n : i64
  cf.cond_br %cond, ^2(%c0: i64), ^1
^1:
// CHECK: "block.info"() {loopInfo = ["latch", "inLoop"]}
  %ni = arith.addi %i, %c1 : i64
  cf.br ^0(%ni: i64)
^2(%j: i64):
// CHECK: "block.info"() {loopInfo = ["exit", "header", "inLoop"]}
  %cond2 = arith.cmpi eq, %j, %n : i64
  cf.cond_br %cond2, ^end, ^3
^3:
// CHECK: "block.info"() {loopInfo = ["latch", "inLoop"]}
  %nj = arith.addi %j, %c1 : i64
  cf.br ^2(%nj: i64)
^end:
// CHECK: "block.info"() {loopInfo = ["exit"]}
  return
}

// -----

// CHECK-LABEL: func.func @nested
func.func @nested(%n: i64) {
  %c0 = arith.constant 1 : i64
  %c1 = arith.constant 1 : i64
  cf.br ^0(%c0 : i64)
^0(%i: i64):
// CHECK: "block.info"() {loopInfo = ["header", "inLoop"]}
  %cond = arith.cmpi eq, %i, %n : i64
  cf.cond_br %cond, ^end, ^1
^1:
// CHECK: "block.info"() {loopInfo = ["inLoop"]}
  %ni = arith.addi %i, %c1 : i64
  cf.br ^2(%ni: i64)
^2(%j: i64):
// CHECK: "block.info"() {loopInfo = ["latch", "inLoop"]}
  %cond2 = arith.cmpi eq, %j, %n : i64
  cf.cond_br %cond2, ^0(%ni: i64), ^3
^3:
// CHECK: "block.info"() {loopInfo = ["inLoop"]}
  %nj = arith.addi %j, %c1 : i64
  cf.br ^2(%nj: i64)
^end:
// CHECK: "block.info"() {loopInfo = ["exit"]}
  return
}
