// RUN: circt-opt %s -export-verilog | FileCheck %s

module attributes {circt.loweringOptions = "locationInfoStyle=wrapInAtSquareBracket"} {

// CHECK-LABEL: module Callstack(
// CHECK-SAME:    // @[{'foo'(fooSource.x:10:8) <- {'bar'(barSource.x:20:8) <- 'baz'(bazSource.x:30:8)}}]

// Emulate a callstack-like location info.
hw.module @Callstack(in %a: i1 loc("")) {
  hw.output
} loc(callsite(
    "foo"("fooSource.x":10:8)
    at callsite("bar"("barSource.x":20:8) 
    at "baz"("bazSource.x":30:8))))


// Check location merging logic.

// CHECK-LABEL: module MergedLocations(
hw.module @MergedLocations(in %clock: i1, in %flag1 : i1, in %flag2: i1, in %flag3: i1) {
  %true = hw.constant 1 : i1 loc("")
  %false = hw.constant 0 : i1
  %r1 = sv.reg : !hw.inout<i1>
  %r2 = sv.reg : !hw.inout<i1>
  sv.always posedge %clock {
    
    // Induce FileLineColLoc merging.
// CHECK:     r1 <= flag1 + flag2 + 1'h1 + flag3; // @[myFile.x:9:3, :10:6, myOtherFile.x:9:4, :10:5]    
    %0 = comb.add %flag1, %flag2 : i1 loc("myFile.x":10:6)
    %1 = comb.add %true, %flag3 : i1 loc("myOtherFile.x":10:5)
    %mergedFLCLoc = comb.add %0, %1 : i1 loc("myOtherFile.x":9:4)
    sv.passign %r1, %mergedFLCLoc : i1 loc("myFile.x":9:3)

    // Now add in a callstack and named location.
    // Note: We duplicate the operations here to avoid wire spilling.
// CHECK:     r2 <= flag1 + flag2 + 1'h1 + flag3 - 1'h1;  // @['MyNamedLocation'(), {'foo'(myFile.x:10:8) <- {'bar'(myFile.x:20:8) <- 'baz'(myFile.x:30:8)}}, myFile.x:9:4, :10:{5,6}]
    %callstack_0 = comb.add %flag1, %flag2 : i1 loc("myFile.x":10:6)
    %callstack_1 = comb.add %true, %flag3 : i1 loc("myFile.x":10:5)
    %callstack_mergedFLCLoc = comb.add %callstack_0, %callstack_1 : i1 loc("myFile.x":9:4)
    %callstackLoc = comb.add %callstack_mergedFLCLoc, %true : i1 loc(callsite(
    "foo"("myFile.x":10:8)
    at callsite("bar"("myFile.x":20:8) 
    at "baz"("myFile.x":30:8))))
    sv.passign %r2, %callstackLoc : i1 loc("MyNamedLocation")
  }
}
}
