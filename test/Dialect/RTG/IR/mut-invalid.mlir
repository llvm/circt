// RUN: circt-opt %s --split-input-file --verify-diagnostics

// Mutable cells and continuations cannot be constructed from typed attributes.
// expected-error @below {{cannot encode value of type '!rtg.mut<index>' in a constant attribute}}
%ref = rtg.constant "not-a-cell" : !rtg.mut<index>

// -----

// expected-error @below {{cannot encode value of type '!rtg.continuation<index>' in a constant attribute}}
%k = rtg.constant "not-a-continuation" : !rtg.continuation<index>

// -----

// expected-error @below {{cannot encode value of type '!rtg.continuation<none>' in a constant attribute}}
%k = "rtg.constant"() <{value = "not-a-continuation" : !rtg.continuation<none>}> : () -> !rtg.continuation<none>

// -----

// Actual identity-valued elements must also be rejected inside aggregates.
// expected-error @below {{cannot encode value of type '!rtg.mut<index>' in a constant attribute}}
%tuple = rtg.constant #rtg.tuple<0 : index, "not-a-cell" : !rtg.mut<index>>

// -----

// expected-error @below {{cannot encode value of type '!rtg.continuation<index>' in a constant attribute}}
%set = rtg.constant #rtg.set<"not-a-continuation" : !rtg.continuation<index>>

// -----

// Map keys and values both need checking.
// expected-error @below {{cannot encode value of type '!rtg.mut<index>' in a constant attribute}}
%map = rtg.constant #rtg.map<"not-a-cell" : !rtg.mut<index> -> 0 : index>

// -----

// expected-error @below {{cannot encode value of type '!rtg.continuation<none>' in a constant attribute}}
%map = rtg.constant #rtg.map<0 : index -> "not-a-continuation" : !rtg.continuation<none>>

// -----

// Recurse through mixed aggregates, not just their immediate elements.
// expected-error @below {{cannot encode value of type '!rtg.mut<index>' in a constant attribute}}
%nested = rtg.constant #rtg.set<#rtg.tuple<#rtg.map<0 : index -> "not-a-cell" : !rtg.mut<index>>>>

// -----

// Identity types in aggregate type parameters are legal when no such value is
// encoded. In particular, empty sets can be used to stash continuations.
rtg.test @empty_identity_containers() {
  %cells = rtg.constant #rtg.set<> : !rtg.set<!rtg.mut<index>>
  %continuations = rtg.constant #rtg.set<> : !rtg.set<!rtg.continuation<index>>
  %map = rtg.constant #rtg.map<> : !rtg.map<!rtg.mut<index> -> !rtg.continuation<none>>
  %tuple = rtg.constant #rtg.tuple<#rtg.set<> : !rtg.set<!rtg.continuation<index>>>
  %nested = rtg.constant #rtg.set<#rtg.map<> : !rtg.map<!rtg.continuation<none> -> !rtg.mut<index>>>
}

// -----

// Generic assembly bypasses type inference in the custom assembly formats.
func.func @create_type_mismatch(%init: index) {
  // expected-error @below {{failed to verify that ref element type must match initialValue type}}
  %ref = "rtg.mut_create"(%init) : (index) -> !rtg.mut<i32>
  return
}

// -----

func.func @read_type_mismatch(%ref: !rtg.mut<index>) {
  // expected-error @below {{failed to verify that result type must match ref element type}}
  %value = "rtg.mut_read"(%ref) : (!rtg.mut<index>) -> i32
  return
}

// -----

func.func @write_type_mismatch(%ref: !rtg.mut<index>, %value: i32) {
  // expected-error @below {{failed to verify that newValue type must match ref element type}}
  "rtg.mut_write"(%ref, %value) : (!rtg.mut<index>, i32) -> ()
  return
}
