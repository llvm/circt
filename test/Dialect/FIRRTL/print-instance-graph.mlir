// RUN: circt-opt -firrtl-print-instance-graph %s -o %t 2>&1 | FileCheck %s

// CHECK: digraph "Top"
// CHECK:   label="Top";
// CHECK:   [[TOP:.*]] [shape=record,label="{Top}"];
// CHECK:   [[TOP]] -> [[ALLIGATOR:.*]][label=alligator];
// CHECK:   [[TOP]] -> [[CAT:.*]][label=cat];
// CHECK:   [[ALLIGATOR]] [shape=record,label="{Alligator}"];
// CHECK:   [[ALLIGATOR]] -> [[BEAR:.*]][label=bear];
// CHECK:   [[CAT]] [shape=record,label="{Cat}"];
// CHECK:   [[BEAR]] [shape=record,label="{Bear}"];
// CHECK:   [[BEAR]] -> [[CAT]][label=cat];

firrtl.circuit "Top" {

firrtl.module @Top() {
  firrtl.instance @Alligator {name = "alligator" }
  firrtl.instance @Cat {name = "cat"}
}

firrtl.module @Alligator() {
  firrtl.instance @Bear {name = "bear"}
}

firrtl.module @Bear() {
  firrtl.instance @Cat {name = "cat" }
}

firrtl.module @Cat() { }

}
