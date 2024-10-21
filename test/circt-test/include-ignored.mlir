// RUN: circt-test %s --include-ignored | FileCheck %s
// RUN: circt-test %s --json --include-ignored | FileCheck --check-prefix=JSON %s
// RUN: circt-as %s -o - | circt-test --include-ignored | FileCheck %s

// JSON: [
// JSON-NEXT: {
// JSON-NEXT:   "name": "Attrs"
// JSON-NEXT:   "kind": "formal"
// JSON-NEXT:   "attrs": {
// JSON-NEXT:     "awesome": true
// JSON-NEXT:     "engine": "bmc"
// JSON-NEXT:     "ignore": true,
// JSON-NEXT:     "offset": 42
// JSON-NEXT:     "tags": [
// JSON-NEXT:       "sby"
// JSON-NEXT:       "induction"
// JSON-NEXT:     ]
// JSON-NEXT:     "wow": false
// JSON-NEXT:   }
// JSON-NEXT: }
// CHECK: Attrs formal {awesome = true, engine = "bmc", ignore = true, offset = 42 : i64, tags = ["sby", "induction"], wow = false}
verif.formal @Attrs attributes {
    awesome = true,
    engine = "bmc",
    offset = 42 : i64,
    tags = ["sby", "induction"],
    wow = false,
    ignore = true
} {}

// JSON: ]
