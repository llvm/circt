// RUN: circt-opt -lower-std-to-handshake %s -split-input-file -verify-diagnostics

// Create a loop with two exit points (task pipelining loops requires unified
// exit blocks). This is also a special situation since the loop header is also
// an exit block.

// expected-error @+1 {{Multiple exits detected within a loop. Loop task pipelining is only supported for loops with unified loop exit blocks.}}
func @main(%cond : i1) -> index {
  %c1 = arith.constant 1 : index
  %c42 = arith.constant 42 : index
  %c1_0 = arith.constant 1 : index
  br ^bb1(%c1 : index)
^bb1(%0: index):	// 2 preds: ^bb0, ^bb2
  %1 = arith.cmpi slt, %0, %c42 : index
  cond_br %1, ^bb2, ^bb3
^bb2:	
  %2 = arith.addi %0, %c1_0 : index
  cond_br %cond, ^bb1(%2 : index), ^bb3
^bb3:	
  return %0 : index
}

// -----


// Create a loop with two exit points (task pipelining loops requires unified
// exit blocks). Like the above, but the loop header is not an exit point.

// expected-error @+1 {{Multiple exits detected within a loop. Loop task pipelining is only supported for loops with unified loop exit blocks.}}
func @main(%cond : i1) -> index {
  %c1 = arith.constant 1 : index
  %c42 = arith.constant 42 : index
  %c1_0 = arith.constant 1 : index
  br ^bb1(%c1 : index)
^bb1(%0: index):
  %1 = arith.cmpi slt, %0, %c42 : index
  br ^bb2
^bb2:	
  %2 = arith.addi %0, %c1_0 : index
  cond_br %cond, ^bb3, ^bb4
^bb3:
  %3 = arith.addi %0, %c1_0 : index
  cond_br %cond, ^bb1(%2 : index), ^bb4
^bb4:	
  return %0 : index
}


// -----

// A loop with a single exit point but with multiple loop latches.

// expected-error @+1 {{Multiple loop latches detected (backedges from within the loop to the loop header). Loop task pipelining is only supported for loops with unified loop latches.}}
func @main(%cond : i1) -> index {
  %c1 = arith.constant 1 : index
  %c42 = arith.constant 42 : index
  %c1_0 = arith.constant 1 : index
  br ^bb1(%c1 : index)
^bb1(%0: index):
  %1 = arith.cmpi slt, %0, %c42 : index
  br ^bb2
^bb2:	
  %2 = arith.addi %0, %c1_0 : index
  cond_br %cond, ^bb1(%2 : index), ^bb3
^bb3:
  %3 = arith.addi %0, %c1_0 : index
  cond_br %cond, ^bb1(%2 : index), ^bb4
^bb4:	
  return %0 : index
}
