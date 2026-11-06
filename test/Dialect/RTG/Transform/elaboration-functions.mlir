// RUN: circt-opt --rtg-elaborate=seed=0 --split-input-file --verify-diagnostics %s | FileCheck %s

func.func @add_one(%value: index) -> index {
  %one = index.constant 1
  %result = index.add %value, %one
  return %result : index
}

func.func @forward(%value: index) -> index {
  %result = func.call @add_one(%value) : (index) -> index
  return %result : index
}

func.func @no_inline(%value: index) -> index attributes {no_inline} {
  return %value : index
}

func.func @recursive(%value: index) -> index {
  %zero = index.constant 0
  %done = index.cmp eq(%value, %zero)
  %result = scf.if %done -> index {
    scf.yield %zero : index
  } else {
    %one = index.constant 1
    %next = index.sub %value, %one
    %result = func.call @recursive(%next) : (index) -> index
    scf.yield %result : index
  }
  return %result : index
}

// Empty void calls execute and leave no operation behind.
func.func @empty() {
  return
}

// The comment makes the evaluated argument observable in the output.
func.func @consume(%value: index) attributes {no_inline} {
  %text = rtg.int_format %value
  rtg.comment %text
  return
}

rtg.target @target : !rtg.dict<> {
  rtg.yield
}

// CHECK-LABEL: rtg.test @inline_function_calls
rtg.test @inline_function_calls() {
  %value = index.constant 41
  %inlined = func.call @forward(%value) : (index) -> index
  func.call @consume(%inlined) : (index) -> ()
  rtg.yield
}
// The nested @forward -> @add_one calls are both inlined.
// CHECK: [[FORTY_TWO:%.+]] = rtg.constant "42" : !rtg.string
// CHECK: rtg.comment [[FORTY_TWO]]
// CHECK-NOT: func.call

// Calls with no_inline attributes and recursive calls are evaluated too.
// CHECK-LABEL: rtg.test @evaluated_function_calls
rtg.test @evaluated_function_calls() {
  %value = index.constant 41
  %marked = func.call @no_inline(%value) : (index) -> index
  func.call @consume(%marked) : (index) -> ()

  %call_marked = func.call @add_one(%value) <no_inline> : (index) -> index
  func.call @consume(%call_marked) : (index) -> ()

  %depth = index.constant 2
  %recursive = func.call @recursive(%depth) : (index) -> index
  func.call @consume(%recursive) : (index) -> ()

  func.call @empty() : () -> ()
  rtg.yield
}
// CHECK: [[FORTY_ONE:%.+]] = rtg.constant "41" : !rtg.string
// CHECK: rtg.comment [[FORTY_ONE]]
// CHECK: [[FORTY_TWO_AGAIN:%.+]] = rtg.constant "42" : !rtg.string
// CHECK: rtg.comment [[FORTY_TWO_AGAIN]]
// CHECK: [[ZERO:%.+]] = rtg.constant "0" : !rtg.string
// CHECK: rtg.comment [[ZERO]]
// CHECK-NOT: func.call

// -----

rtg.effect @choose : () -> index

func.func @choose_in_helper() -> index {
  %value = rtg.perform @choose() : () -> index
  return %value : index
}

// Leave an output trace of each resumed value.
func.func @consume_choice(%value: index) attributes {no_inline} {
  %text = rtg.int_format %value
  rtg.comment %text
  return
}

rtg.target @target : !rtg.dict<> {
  rtg.yield
}

// CHECK-LABEL: rtg.test @effect_through_call
rtg.test @effect_through_call() {
  rtg.with_handlers {
    handle @choose(%continuation: !rtg.continuation<index>) {
      %seven = index.constant 7
      rtg.resume %continuation, %seven : !rtg.continuation<index>, index
      %thirteen = index.constant 13
      rtg.resume %continuation, %thirteen : !rtg.continuation<index>, index
      rtg.yield
    }
    do {
      %value = func.call @choose_in_helper() : () -> index
      func.call @consume_choice(%value) : (index) -> ()
      rtg.yield
    }
  }
}
// The captured continuation crosses the inlined function boundary, returns
// each resumed value to the call site, and replays the caller's remaining ops.
// CHECK-NOT: rtg.with_handlers
// CHECK-NOT: rtg.perform
// CHECK-NOT: rtg.resume
// CHECK: [[SEVEN:%.+]] = rtg.constant "7" : !rtg.string
// CHECK: rtg.comment [[SEVEN]]
// CHECK: [[THIRTEEN:%.+]] = rtg.constant "13" : !rtg.string
// CHECK: rtg.comment [[THIRTEEN]]
// CHECK-NOT: func.call
