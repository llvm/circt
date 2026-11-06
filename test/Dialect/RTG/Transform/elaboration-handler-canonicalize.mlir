// RUN: circt-opt --rtg-elaborate=seed=0 %s | FileCheck %s --check-prefix=ELAB
// RUN: circt-opt --canonicalize %s | FileCheck %s --check-prefix=FOLDED
// RUN: circt-opt --canonicalize --rtg-elaborate=seed=0 %s | FileCheck %s --check-prefix=CANON

rtg.effect @canonicalized_if_effect : () -> index
func.func @record_canonicalized_if(%value: index) {
  %text = rtg.int_format %value
  rtg.comment %text
  return
}

rtg.target @t : !rtg.dict<> {
  rtg.yield
}

rtg.test @canonicalized_if_continuation() {
  %zero = index.constant 0
  %one = index.constant 1
  %ref = rtg.mut_create %zero : index
  %true = arith.constant true
  rtg.with_handlers {
    handle @canonicalized_if_effect(%k: !rtg.continuation<index>) {
      %five = index.constant 5
      rtg.resume %k, %five : !rtg.continuation<index>, index
      rtg.yield
    }
    do {
      scf.if %true {
        %value = rtg.perform @canonicalized_if_effect() : () -> index
        func.call @record_canonicalized_if(%value) : (index) -> ()
      }
      rtg.mut_write %ref, %one : !rtg.mut<index>
      %observed = rtg.mut_read %ref : !rtg.mut<index>
      func.call @record_canonicalized_if(%observed) : (index) -> ()
      rtg.yield
    }
  }
}

// Folding the constant-condition scf.if must not shorten the handler-delimited
// continuation; both runs produce the same branch and outer-body suffix.
// FOLDED-LABEL: rtg.test @canonicalized_if_continuation
// FOLDED-NOT: scf.if
// FOLDED: rtg.perform @canonicalized_if_effect
// ELAB: [[FIVE:%.+]] = rtg.constant "5" : !rtg.string
// ELAB: rtg.comment [[FIVE]]
// ELAB: [[ONE:%.+]] = rtg.constant "1" : !rtg.string
// ELAB: rtg.comment [[ONE]]
// CANON: [[FIVE:%.+]] = rtg.constant "5" : !rtg.string
// CANON: rtg.comment [[FIVE]]
// CANON: [[ONE:%.+]] = rtg.constant "1" : !rtg.string
// CANON: rtg.comment [[ONE]]
