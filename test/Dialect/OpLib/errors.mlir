// RUN: circt-opt %s -split-input-file -verify-diagnostics

oplib.library @lib0 {
  oplib.operator @addi latency<0>, incDelay<0.2>, outDelay<0.2> {
    oplib.target @target0(%l: i32, %r: i32) -> i32 {
      %o = oplib.operation "addi" in "arith"(%l, %r : i32, i32) : i32
      oplib.output %o : i32
    }
    oplib.calyx_match(@target0 : (i32, i32) -> i32) produce {
      %left, %right, %out = calyx.std_add @add : i32, i32, i32
      oplib.yield ins(%left, %right : i32, i32), outs(%out : i32)
    }
  }
}

// -----

// expected-error @+1 {{can only contain OperatorOps}}
oplib.library @lib0 {
  func.func @main() {
    func.return
  }
}

// -----

oplib.library @lib0 {
  // expected-error @+1 {{region terminator must be supported match op}}
  oplib.operator @addi latency<0>, incDelay<0.2>, outDelay<0.2> {
    oplib.target @target0(%l: i32, %r: i32) -> i32 {
      %o = oplib.operation "addi" in "arith"(%l, %r : i32, i32) : i32
      oplib.output %o : i32
    }
    func.return
  }
}

// -----

oplib.library @lib0 {
  // expected-error @+1 {{must have either both incDelay and outDelay or neither}}
  oplib.operator @addi latency<0>, incDelay<0.2> {
    oplib.target @target0(%l: i32, %r: i32) -> i32 {
      %o = oplib.operation "addi" in "arith"(%l, %r : i32, i32) : i32
      oplib.output %o : i32
    }
    oplib.calyx_match(@target0 : (i32, i32) -> i32) produce {
      %left, %right, %out = calyx.std_add @add : i32, i32, i32
      oplib.yield ins(%left, %right : i32, i32), outs(%out : i32)
    }
  }
}

// -----

oplib.library @lib0 {
  // expected-error @+1 {{incDelay and outDelay of combinational operators must be the same}}
  oplib.operator @addi latency<0>, incDelay<0.2>, outDelay<0.5> {
    oplib.target @target0(%l: i32, %r: i32) -> i32 {
      %o = oplib.operation "addi" in "arith"(%l, %r : i32, i32) : i32
      oplib.output %o : i32
    }
    oplib.calyx_match(@target0 : (i32, i32) -> i32) produce {
      %left, %right, %out = calyx.std_add @add : i32, i32, i32
      oplib.yield ins(%left, %right : i32, i32), outs(%out : i32)
    }
  }
}
