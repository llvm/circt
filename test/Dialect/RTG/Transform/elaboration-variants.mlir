// RUN: circt-opt --rtg-elaborate=seed=0 --split-input-file --verify-diagnostics %s | FileCheck %s

rtg.variant.decl @Option {
  None: !rtg.tuple,
  Some: index
}

func.func @consume_index(%value: index) attributes {no_inline} {
  %text = rtg.int_format %value
  rtg.comment %text
  return
}

rtg.target @target : !rtg.dict<> {
  rtg.yield
}

// CHECK-LABEL: rtg.test @variant_create_and_match_target
rtg.test @variant_create_and_match() {
  %five = index.constant 5
  %some = rtg.variant_create Some(%five) : (index) -> !rtg.variant<@Option>
  %matched = rtg.match %some : !rtg.variant<@Option> -> index {
    case None(%unit: !rtg.tuple) {
      %zero = index.constant 0
      rtg.yield %zero : index
    }
    case Some(%payload: index) {
      rtg.yield %payload : index
    }
  }
  // CHECK: [[FIVE:%.+]] = rtg.constant "5" : !rtg.string
  // CHECK-NEXT: rtg.comment [[FIVE]]
  func.call @consume_index(%matched) : (index) -> ()
}

// -----

rtg.variant.decl @Option {
  None: !rtg.tuple,
  Some: index
}

func.func @consume_index(%value: index) attributes {no_inline} {
  %text = rtg.int_format %value
  rtg.comment %text
  return
}

rtg.target @target : !rtg.dict<> {
  rtg.yield
}

// CHECK-LABEL: rtg.test @nullary_variant_match_target
rtg.test @nullary_variant_match() {
  %none = rtg.variant_create None() : () -> !rtg.variant<@Option>
  %matched = rtg.match %none : !rtg.variant<@Option> -> index {
    case None(%unit: !rtg.tuple) {
      %zero = index.constant 0
      rtg.yield %zero : index
    }
    case Some(%payload: index) {
      rtg.yield %payload : index
    }
  }
  // CHECK: [[ZERO:%.+]] = rtg.constant "0" : !rtg.string
  // CHECK-NEXT: rtg.comment [[ZERO]]
  func.call @consume_index(%matched) : (index) -> ()
}

// -----

rtg.variant.decl @Option {
  None: !rtg.tuple,
  Some: index
}

func.func @consume_index(%value: index) attributes {no_inline} {
  %text = rtg.int_format %value
  rtg.comment %text
  return
}

rtg.target @target : !rtg.dict<> {
  rtg.yield
}

// CHECK-LABEL: rtg.test @variant_in_tuple_target
rtg.test @variant_in_tuple() {
  %five = index.constant 5
  %some = rtg.variant_create Some(%five) : (index) -> !rtg.variant<@Option>
  %tuple = rtg.tuple_create %some, %five : !rtg.variant<@Option>, index
  %variant = rtg.tuple_extract %tuple at 0 : !rtg.tuple<!rtg.variant<@Option>, index>
  %payload = rtg.match %variant : !rtg.variant<@Option> -> index {
    case None(%unit: !rtg.tuple) {
      %zero = index.constant 0
      rtg.yield %zero : index
    }
    case Some(%value: index) {
      rtg.yield %value : index
    }
  }
  // CHECK: [[FIVE:%.+]] = rtg.constant "5" : !rtg.string
  // CHECK-NEXT: rtg.comment [[FIVE]]
  func.call @consume_index(%payload) : (index) -> ()
}

// -----

rtg.variant.decl @Tree {
  Leaf: index,
  Branch: !rtg.variant<@Tree>
}

func.func @consume_index(%value: index) attributes {no_inline} {
  %text = rtg.int_format %value
  rtg.comment %text
  return
}

rtg.target @target : !rtg.dict<> {
  rtg.yield
}

// CHECK-LABEL: rtg.test @nested_variant_payload_target
rtg.test @nested_variant_payload() {
  %thirteen = index.constant 13
  %leaf = rtg.variant_create Leaf(%thirteen) : (index) -> !rtg.variant<@Tree>
  %branch = rtg.variant_create Branch(%leaf) : (!rtg.variant<@Tree>) -> !rtg.variant<@Tree>
  %value = rtg.match %branch : !rtg.variant<@Tree> -> index {
    case Leaf(%leaf_payload: index) {
      rtg.yield %leaf_payload : index
    }
    case Branch(%branch_payload: !rtg.variant<@Tree>) {
      %nested = rtg.match %branch_payload : !rtg.variant<@Tree> -> index {
        case Leaf(%nested_leaf: index) {
          rtg.yield %nested_leaf : index
        }
        case Branch(%nested_branch: !rtg.variant<@Tree>) {
          %zero = index.constant 0
          rtg.yield %zero : index
        }
      }
      rtg.yield %nested : index
    }
  }
  // CHECK: [[THIRTEEN:%.+]] = rtg.constant "13" : !rtg.string
  // CHECK-NEXT: rtg.comment [[THIRTEEN]]
  func.call @consume_index(%value) : (index) -> ()
}

// -----

rtg.struct.decl @Payload { value: index, valid: i1 }
rtg.variant.decl @Envelope {
  Record: !rtg.struct<@Payload>,
  Pair: !rtg.tuple<index, i1>
}

func.func @consume_index(%value: index) attributes {no_inline} {
  %text = rtg.int_format %value
  rtg.comment %text
  return
}

rtg.target @target : !rtg.dict<> {
  rtg.yield
}

// CHECK-LABEL: rtg.test @variant_struct_payload_target
rtg.test @variant_struct_payload() {
  %thirteen = index.constant 13
  %true = index.bool.constant true
  %record = rtg.struct_create %thirteen, %true : (index, i1) -> !rtg.struct<@Payload>
  %envelope = rtg.variant_create Record(%record) : (!rtg.struct<@Payload>) -> !rtg.variant<@Envelope>
  %value = rtg.match %envelope : !rtg.variant<@Envelope> -> index {
    case Record(%record_payload: !rtg.struct<@Payload>) {
      %field = rtg.struct_extract %record_payload[value] : !rtg.struct<@Payload> -> index
      rtg.yield %field : index
    }
    case Pair(%pair: !rtg.tuple<index, i1>) {
      %field = rtg.tuple_extract %pair at 0 : !rtg.tuple<index, i1>
      rtg.yield %field : index
    }
  }
  // CHECK: [[THIRTEEN:%.+]] = rtg.constant "13" : !rtg.string
  // CHECK-NEXT: rtg.comment [[THIRTEEN]]
  func.call @consume_index(%value) : (index) -> ()
}

// -----

rtg.variant.decl @Option {
  None: !rtg.tuple,
  Some: index
}

func.func @identity_variant(%value: !rtg.variant<@Option>) -> !rtg.variant<@Option> {
  return %value : !rtg.variant<@Option>
}

func.func @consume_index(%value: index) attributes {no_inline} {
  %text = rtg.int_format %value
  rtg.comment %text
  return
}

rtg.target @target : !rtg.dict<> {
  rtg.yield
}

// CHECK-LABEL: rtg.test @variant_through_function_target
rtg.test @variant_through_function() {
  %constant = rtg.constant #rtg.variant<Some, 9 : index> : !rtg.variant<@Option>
  %through_function = func.call @identity_variant(%constant) : (!rtg.variant<@Option>) -> !rtg.variant<@Option>
  %payload = rtg.match %through_function : !rtg.variant<@Option> -> index {
    case None(%unit: !rtg.tuple) {
      %zero = index.constant 0
      rtg.yield %zero : index
    }
    case Some(%value: index) {
      rtg.yield %value : index
    }
  }
  // CHECK: [[NINE:%.+]] = rtg.constant "9" : !rtg.string
  // CHECK-NEXT: rtg.comment [[NINE]]
  func.call @consume_index(%payload) : (index) -> ()
}

// -----

rtg.variant.decl @Option {
  None: !rtg.tuple,
  Some: index
}

func.func @consume_index(%value: index) attributes {no_inline} {
  %text = rtg.int_format %value
  rtg.comment %text
  return
}

rtg.target @target : !rtg.dict<> {
  rtg.yield
}

// The constructor is known even though the payload crosses an opaque region.
// CHECK-LABEL: rtg.test @symbolic_variant_payload_target
rtg.test @symbolic_variant_payload() {
  %opaque_payload = scf.execute_region -> index {
    %seventeen = index.constant 17
    scf.yield %seventeen : index
  }
  %some = rtg.variant_create Some(%opaque_payload) : (index) -> !rtg.variant<@Option>
  %payload = rtg.match %some : !rtg.variant<@Option> -> index {
    case None(%unit: !rtg.tuple) {
      %zero = index.constant 0
      rtg.yield %zero : index
    }
    case Some(%value: index) {
      rtg.yield %value : index
    }
  }
  // CHECK: scf.execute_region -> index {
  // CHECK: {{.*}} = rtg.int_format {{.*}}
  // CHECK-NEXT: rtg.comment {{.*}}
  func.call @consume_index(%payload) : (index) -> ()
}

// -----

rtg.variant.decl @Option {
  None: !rtg.tuple,
  Some: index
}
rtg.target @target : !rtg.dict<> { rtg.yield }

// Continuations captured inside a selected match region replay through the
// match frame and continue in that same region after each resume.
rtg.effect @choose : () -> index
func.func @consume(%value: index) attributes {no_inline} {
  %text = rtg.int_format %value
  rtg.comment %text
  return
}

// CHECK-LABEL: rtg.test @match_continuation
rtg.test @match_continuation() {
  %seven = index.constant 7
  %some = rtg.variant_create Some(%seven) : (index) -> !rtg.variant<@Option>
  rtg.with_handlers {
    handle @choose(%continuation: !rtg.continuation<index>) {
      %eleven = index.constant 11
      rtg.resume %continuation, %eleven : !rtg.continuation<index>, index
      %thirteen = index.constant 13
      rtg.resume %continuation, %thirteen : !rtg.continuation<index>, index
      rtg.yield
    }
    do {
      rtg.match %some : !rtg.variant<@Option> {
        case None(%unit: !rtg.tuple) {
          rtg.yield
        }
        case Some(%payload: index) {
          %value = rtg.perform @choose() : () -> index
          // CHECK: [[ELEVEN:%.+]] = rtg.constant "11" : !rtg.string
          // CHECK: rtg.comment [[ELEVEN]]
          // CHECK: [[THIRTEEN:%.+]] = rtg.constant "13" : !rtg.string
          // CHECK: rtg.comment [[THIRTEEN]]
          func.call @consume(%value) : (index) -> ()
          rtg.yield
        }
      }
      rtg.yield
    }
  }
}

// -----

rtg.variant.decl @Option {
  None: !rtg.tuple,
  Some: index
}
rtg.target @target : !rtg.dict<> { rtg.yield }

// CHECK-LABEL: rtg.test @match_constant_variant
rtg.test @match_constant_variant() {
  %constant = rtg.constant #rtg.variant<Some, 9 : index> : !rtg.variant<@Option>
  rtg.match %constant : !rtg.variant<@Option> {
    case None(%unit: !rtg.tuple) {
      rtg.yield
    }
    case Some(%payload: index) {
      %text = rtg.int_format %payload
      // CHECK: [[NINE:%.+]] = rtg.constant "9" : !rtg.string
      // CHECK-NEXT: rtg.comment [[NINE]]
      rtg.comment %text
      rtg.yield
    }
  }
}

// -----

rtg.variant.decl @Option {
  None: !rtg.tuple,
  Some: index
}
rtg.target @target : !rtg.dict<> { rtg.yield }

rtg.test @non_concrete_selector() {
  %bit = index.bool.constant true
  %selector = builtin.unrealized_conversion_cast %bit : i1 to !rtg.variant<@Option>
  // expected-error @below {{cannot evaluate match with a non-concrete variant selector}}
  rtg.match %selector : !rtg.variant<@Option> {
    case None(%unit: !rtg.tuple) { rtg.yield }
    case Some(%payload: index) { rtg.yield }
  }
}
