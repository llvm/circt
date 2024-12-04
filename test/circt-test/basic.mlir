// RUN: circt-test -l %s | FileCheck %s
// RUN: circt-test -l --json %s | FileCheck --check-prefix=JSON %s
// RUN: circt-as %s -o - | circt-test -l | FileCheck %s
// RUN: circt-test -l %s --list-ignored | FileCheck --check-prefix=CHECK-WITH-IGNORED %s

// JSON: [

// JSON-NEXT: {
// JSON-NEXT:   "name": "Some.TestA"
// JSON-NEXT:   "kind": "formal"
// JSON-NEXT: }
// JSON-NEXT: {
// JSON-NEXT:   "name": "Some.TestB"
// JSON-NEXT:   "kind": "formal"
// JSON-NEXT: }
// CHECK: Some.TestA formal {}
// CHECK: Some.TestB formal {}
verif.formal @Some.TestA {} {}
verif.formal @Some.TestB {} {}

// JSON-NEXT: {
// JSON-NEXT:   "name": "Attrs"
// JSON-NEXT:   "kind": "formal"
// JSON-NEXT:   "attrs": {
// JSON-NEXT:     "awesome": true
// JSON-NEXT:     "engine": "bmc"
// JSON-NEXT:     "ignore": false
// JSON-NEXT:     "offset": 42
// JSON-NEXT:     "tags": [
// JSON-NEXT:       "sby"
// JSON-NEXT:       "induction"
// JSON-NEXT:     ]
// JSON-NEXT:     "wow": false
// JSON-NEXT:   }
// JSON-NEXT: }
// CHECK: Attrs formal {awesome = true, engine = "bmc", ignore = false, offset = 42 : i64, tags = ["sby", "induction"], wow = false}
verif.formal @Attrs {
    awesome = true,
    engine = "bmc",
    offset = 42 : i64,
    tags = ["sby", "induction"],
    wow = false,
    ignore = false
} {}

// CHECK-NOT: "name": "Ignore"
// JSON-NOT: "name": "Ignore"
// CHECK-WITH-IGNORED: Ignore formal {another = "attr", ignore = true}
verif.formal @Ignore {
    ignore = true,
    another = "attr"
} {}

// JSON: ]
