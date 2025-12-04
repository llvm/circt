// RUN: circt-opt %s --split-input-file --pass-pipeline='builtin.module(synth-print-longest-path-analysis{output-file="-" show-top-k-percent=100})' | FileCheck %s
// RUN: circt-opt %s --split-input-file --pass-pipeline='builtin.module(synth-print-longest-path-analysis{output-file="-" show-top-k-percent=100 emit-json=true})' | FileCheck %s --check-prefix JSON

// CHECK:      # Longest Path Analysis result for "parent"
// CHECK-NEXT: Found 4 paths
// CHECK-NEXT: Found 2 unique end points
// CHECK-NEXT: Maximum path delay: 2
// CHECK-NEXT: ## Showing Levels
// CHECK-NEXT: Level = 1         . Count = 1         . 50.00     %
// CHECK-NEXT: Level = 2         . Count = 1         . 100.00    %
// CHECK-NEXT: ## Top 2 (out of 2) end points
// CHECK:      ==============================================
// CHECK-NEXT: #1: Distance=2
// CHECK-NEXT: EndPoint=Object($root.y[0])
// CHECK-NEXT: StartPoint=Object($root.a[0])
// CHECK-NEXT: == History Start (closer to end point) ==
// CHECK-NEXT: <--- (logic delay 0) ---
// CHECK-NEXT: Object($root.c2.x[0], delay=2, comment="output port")
// CHECK-NEXT: <--- (logic delay 0) ---
// CHECK-NEXT: Object($root/c2:child.r[0], delay=2, comment="namehint")
// CHECK-NEXT: <--- (logic delay 1) ---
// CHECK-NEXT: Object($root/c2:child.a[0], delay=1, comment="input port")
// CHECK-NEXT: <--- (logic delay 0) ---
// CHECK-NEXT: Object($root.c1.x[0], delay=1, comment="output port")
// CHECK-NEXT: <--- (logic delay 0) ---
// CHECK-NEXT: Object($root/c1:child.r[0], delay=1, comment="namehint")
// CHECK-NEXT: <--- (logic delay 1) ---
// CHECK-NEXT: Object($root/c1:child.a[0], delay=0, comment="input port")
// CHECK-NEXT: <--- (logic delay 0) ---
// CHECK-NEXT: Object($root.a[0], delay=0, comment="input port")
// CHECK-NEXT: == History End (closer to start point) ==
// CHECK-NEXT: ==============================================
// Make sure the second path is reported.
// CHECK-NEXT: #2: Distance=1
// JSON:               [{"module_name":"parent",
// JSON-SAME{LITERAL}:   "timing_levels":
// JSON-SAME{LITERAL}:    [{"level":1,"count":1,"percentage":50},
// JSON-SAME{LITERAL}:     {"level":2,"count":1,"percentage":100}]
// JSON-SAME{LITERAL}:    "top_paths":[
// JSON-SAME{LITERAL}:     {"end_point":{"bit_pos":0,"instance_path":[],"name":"y"}
// JSON-SAME{LITERAL}:      "path":{"delay":2,"history":[{"comment":"output port","delay":2,"object":{"bit_pos":0,"instance_path":[],"name":"c2.x"}}
// JSON-SAME{LITERAL}:      "start_point":{"bit_pos":0,"instance_path":[],"name":"a"}
// JSON-SAME{LITERAL}:      "root":"parent"}
// Make sure the second path is reported.
// JSON-SAME{LITERAL}:     {"end_point":{"bit_pos":0,"instance_path":[],"name":"x"},

hw.module private @child(in %a : i1, in %b : i1, out x : i1) {
  %r = synth.aig.and_inv %a, %b {sv.namehint = "r"} : i1 // r[0] := max(a[0], b[0]) + 1 = 1
  hw.output %r : i1
}

hw.module private @parent(in %a : i1, in %b : i1, out x : i1, out y : i1) {
  %0 = hw.instance "c1" @child(a: %a: i1, b: %b: i1) -> (x: i1)
  %1 = hw.instance "c2" @child(a: %0: i1, b: %b: i1) -> (x: i1)
  hw.output %0, %1 : i1, i1
}
