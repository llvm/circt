// RUN: circt-opt --cse --canonicalize %s | FileCheck %s
// RUN: circt-opt --cse --canonicalize='top-down=false' %s | FileCheck %s

om.class @Foo() {
  om.class.fields
}

// CHECK-LABEL: @ObjectsMustNotCSE
func.func @ObjectsMustNotCSE() -> (!om.class.type<@Foo>, !om.class.type<@Foo>) {
  // CHECK-NEXT: [[OBJ1:%.+]] = om.object @Foo
  // CHECK-NEXT: [[OBJ2:%.+]] = om.object @Foo
  // CHECK-NEXT: return [[OBJ1]], [[OBJ2]]
  %obj1 = om.object @Foo() : () -> !om.class.type<@Foo>
  %obj2 = om.object @Foo() : () -> !om.class.type<@Foo>
  return %obj1, %obj2 : !om.class.type<@Foo>, !om.class.type<@Foo>
}

om.class @FooWithAssert() {
  %0 = om.constant false
  %1 = om.constant "foo" : !om.string
  om.property_assert %0, %1 : i1
  om.class.fields
}

// An object instantiation of a class containing a property_assert must not be
// canonicalized away, because the assert has observable side effects.
// CHECK-LABEL: @ObjectWithAssertMustNotDCE
om.class @ObjectWithAssertMustNotDCE() {
  // CHECK: om.object @FooWithAssert
  %0 = om.object @FooWithAssert() : () -> !om.class.type<@FooWithAssert>
  om.class.fields
}

om.class @StringConcatCanonicalization(%str1: !om.string, %str2: !om.string) -> (out1: !om.string, out2: !om.string,
                                                                                  out3: !om.string, out4: !om.string,
                                                                                  out5: !om.string, out6: !om.string,
                                                                                  out7: !om.string, out8: !om.string) {
  %s1 = om.constant "Hello" : !om.string
  %s2 = om.constant "World" : !om.string
  %s3 = om.constant "!" : !om.string
  %empty = om.constant "" : !om.string

  // CHECK-DAG: [[EMPTY:%.+]] = om.constant "" : !om.string
  // CHECK-DAG: [[HELLO:%.+]] = om.constant "Hello" : !om.string
  // CHECK-DAG: [[HELLOWORLD:%.+]] = om.constant "HelloWorld!" : !om.string
  // CHECK-DAG: [[CONST:%.+]] = om.constant "!"

  // Merge all constants
  %0 = om.string.concat %s1, %s2, %s3 : !om.string

  // Drop empty string
  %1 = om.string.concat %s1, %empty : !om.string

  // Single operand replaced with operand
  %2 = om.string.concat %s1 : !om.string

  // Single constant operand folds to the attribute.
  %singleConst = om.string.concat %s3 : !om.string

  // Empty concat
  %3 = om.string.concat %empty, %empty : !om.string

  // Flatten nested concat (single use)
  %4 = om.string.concat %s1, %s2 : !om.string
  %5 = om.string.concat %4, %s3 : !om.string

  // Nested concat with multiple uses should NOT be flattened
  // to avoid fighting with DCE.
  // CHECK-DAG: [[NESTED:%.+]] = om.string.concat %str1, %str2
  // CHECK-DAG: [[CONCAT1:%.+]] = om.string.concat [[NESTED]], [[CONST]]
  %nested = om.string.concat %str1, %str2 : !om.string
  %concat1 = om.string.concat %nested, %s3 : !om.string

  // CHECK: om.class.fields [[HELLOWORLD]], [[HELLO]], [[HELLO]], [[CONST]], [[EMPTY]], [[HELLOWORLD]], [[CONCAT1]], [[NESTED]]
  om.class.fields %0, %1, %2, %singleConst, %3, %5, %concat1, %nested : !om.string, !om.string, !om.string, !om.string, !om.string, !om.string, !om.string, !om.string
}

// CHECK-LABEL: @IntegerBinaryArithmeticFold
om.class @IntegerBinaryArithmeticFold(%x: !om.integer) -> (out1: !om.integer, out2: !om.integer,
                                                           out3: !om.integer, out4: !om.integer,
                                                           out5: !om.integer, out6: !om.integer) {
  %i3 = om.constant #om.integer<3 : si4> : !om.integer
  %i4 = om.constant #om.integer<4 : si4> : !om.integer
  %i2 = om.constant #om.integer<2 : si4> : !om.integer
  %neg1 = om.constant #om.integer<-1 : si4> : !om.integer
  %i1 = om.constant #om.integer<1 : si4> : !om.integer
  %wide = om.constant #om.integer<7 : si6> : !om.integer

  // CHECK-DAG: [[ADD:%.+]] = om.constant #om.integer<7 : si4> : !om.integer
  // Arithmetic uses APSInt semantics at the operands' folded bit width, so
  // 3 * 4 folds to si4 -4 here.
  // CHECK-DAG: [[MUL:%.+]] = om.constant #om.integer<-4 : si4> : !om.integer
  // CHECK-DAG: [[SHR:%.+]] = om.constant #om.integer<1 : si4> : !om.integer
  // CHECK-DAG: [[SHL:%.+]] = om.constant #om.integer<-2 : si5> : !om.integer
  // CHECK-DAG: [[WIDEADD:%.+]] = om.constant #om.integer<9 : si6> : !om.integer
  // CHECK: [[DYN:%.+]] = om.integer.add %x, %{{.+}} : !om.integer
  %0 = om.integer.add %i3, %i4 : !om.integer
  %1 = om.integer.mul %i3, %i4 : !om.integer
  %2 = om.integer.shr %i4, %i2 : !om.integer
  %3 = om.integer.shl %neg1, %i1 : !om.integer

  // Mixed bit widths should still fold after extending operands.
  %4 = om.integer.add %i2, %wide : !om.integer

  // Non-constant operands should remain.
  %5 = om.integer.add %x, %i1 : !om.integer

  // CHECK: om.class.fields [[ADD]], [[MUL]], [[SHR]], [[SHL]], [[WIDEADD]], [[DYN]]
  om.class.fields %0, %1, %2, %3, %4, %5 : !om.integer, !om.integer, !om.integer, !om.integer, !om.integer, !om.integer
}

// CHECK-LABEL: @IntegerBitwiseOpsFold
om.class @IntegerBitwiseOpsFold(%x: !om.integer) -> (out1: !om.integer, out2: !om.integer,
                                                     out3: !om.integer, out4: !om.integer,
                                                     out5: !om.integer) {
  %i3 = om.constant #om.integer<3 : si4> : !om.integer
  %i5 = om.constant #om.integer<5 : si4> : !om.integer
  %neg2 = om.constant #om.integer<-2 : si4> : !om.integer
  %wide = om.constant #om.integer<16 : si6> : !om.integer
  %zero = om.constant #om.integer<0 : si4> : !om.integer

  // CHECK-DAG: [[AND:%.+]] = om.constant #om.integer<1 : si4> : !om.integer
  // CHECK-DAG: [[OR:%.+]] = om.constant #om.integer<7 : si4> : !om.integer
  // CHECK-DAG: [[NOT:%.+]] = om.constant #om.integer<-1 : si4> : !om.integer
  // -2 is sign-extended to the 6-bit width of %wide before the operation.
  // CHECK-DAG: [[WIDEOR:%.+]] = om.constant #om.integer<-2 : si6> : !om.integer
  // CHECK: [[DYN:%.+]] = om.integer.bitand %x, %{{.+}} : !om.integer
  %0 = om.integer.bitand %i3, %i5 : !om.integer
  %1 = om.integer.bitor %i3, %i5 : !om.integer
  %2 = om.integer.bitnot %zero : !om.integer

  // Mixed bit widths should still fold after extending operands.
  %3 = om.integer.bitor %neg2, %wide : !om.integer

  // Non-constant operands should remain.
  %4 = om.integer.bitand %x, %i3 : !om.integer

  // CHECK: om.class.fields [[AND]], [[OR]], [[NOT]], [[WIDEOR]], [[DYN]]
  om.class.fields %0, %1, %2, %3, %4 : !om.integer, !om.integer, !om.integer, !om.integer, !om.integer
}

// CHECK-LABEL: @PropEqFold
om.class @PropEqFold(%str: !om.string, %b: i1, %n: !om.integer,
                     %ref: !om.class.type<@Foo>, %other: !om.class.type<@Foo>,
                     %any: !om.any) -> (
    out1: i1, out2: i1, out3: i1, out4: i1, out5: i1,
    out6: i1, out7: i1, out8: i1, out9: i1, out10: i1,
    equal: i1, not_equal: i1, any_equal: i1, any_not_equal: i1,
    parameter: i1, may_alias: i1, unknown: i1, any_parameter: i1,
    unknown_alias: i1, unknown_distinct: i1) {
  %hello1 = om.constant "hello" : !om.string
  %hello2 = om.constant "hello" : !om.string
  %world  = om.constant "world" : !om.string

  // CHECK-DAG: [[TRUE:%.+]] = om.constant true
  // CHECK-DAG: [[FALSE:%.+]] = om.constant false

  // Equal constant strings fold to true.
  %0 = om.prop.eq %hello1, %hello2 : !om.string

  // Unequal constant strings fold to false.
  %1 = om.prop.eq %hello1, %world : !om.string

  // Non-constant string operands do not fold.
  // CHECK: [[EQ:%.+]] = om.prop.eq %str, %str : !om.string
  %2 = om.prop.eq %str, %str : !om.string

  %true  = om.constant true
  %false = om.constant false

  // Equal constant booleans fold to true.
  %3 = om.prop.eq %true, %true : i1

  // Unequal constant booleans fold to false.
  %4 = om.prop.eq %true, %false : i1

  // Non-constant bool operands do not fold.
  // CHECK: [[BEQ:%.+]] = om.prop.eq %b, %b : i1
  %5 = om.prop.eq %b, %b : i1

  %i42a = om.constant #om.integer<42 : si64> : !om.integer
  %i42b = om.constant #om.integer<42 : si64> : !om.integer
  %i42_signless = om.constant #om.integer<42 : i64> : !om.integer
  %i0   = om.constant #om.integer<0 : si64> : !om.integer

  // Equal constant integers fold to true.
  %6 = om.prop.eq %i42a, %i42b : !om.integer

  // Equal constant integers with different signedness fold to true.
  %7 = om.prop.eq %i42a, %i42_signless : !om.integer

  // Unequal constant integers fold to false.
  %8 = om.prop.eq %i42a, %i0 : !om.integer

  // Non-constant integer operands do not fold.
  // CHECK: [[IEQ:%.+]] = om.prop.eq %n, %n : !om.integer
  %9 = om.prop.eq %n, %n : !om.integer

  %a = om.elaborated_object @Foo() : () -> !om.class.type<@Foo>
  %otherObject = om.elaborated_object @Foo() : () -> !om.class.type<@Foo>
  %aAny = om.any_cast %a : (!om.class.type<@Foo>) -> !om.any
  %bAny = om.any_cast %otherObject : (!om.class.type<@Foo>) -> !om.any

  // Compare identities, not fields, looking through any casts.
  %equal = om.prop.eq %a, %a : !om.class.type<@Foo>
  %notEqual = om.prop.eq %a, %otherObject : !om.class.type<@Foo>
  %anyEqual = om.prop.eq %aAny, %aAny : !om.any
  %anyNotEqual = om.prop.eq %aAny, %bAny : !om.any

  // Self-comparisons fold even when the parameter's identity is unknown.
  %parameter = om.prop.eq %ref, %ref : !om.class.type<@Foo>
  %anyParameter = om.prop.eq %any, %any : !om.any

  // Distinct parameters may alias other references, so do not fold them.
  // CHECK: [[MAY_ALIAS:%.+]] = om.prop.eq %ref, %other : !om.class.type<@Foo>
  %mayAlias = om.prop.eq %ref, %other : !om.class.type<@Foo>

  // Unknown references are equal to themselves, including through any casts.
  // Distinct unknown SSA values may alias, so their comparison must not fold.
  // CHECK: [[UNKNOWN_REF:%.+]] = om.unknown : !om.class.type<@Foo>
  %unknown = om.unknown : !om.class.type<@Foo>
  %unknownAny = om.any_cast %unknown : (!om.class.type<@Foo>) -> !om.any
  %unknownEqual = om.prop.eq %unknown, %unknown : !om.class.type<@Foo>
  %unknownAlias = om.prop.eq %unknownAny, %unknownAny : !om.any
  // CHECK: [[OTHER_UNKNOWN_REF:%.+]] = om.unknown : !om.class.type<@Foo>
  %otherUnknown = om.unknown : !om.class.type<@Foo>
  // CHECK: [[UNKNOWN_DISTINCT:%.+]] = om.prop.eq [[UNKNOWN_REF]], [[OTHER_UNKNOWN_REF]] : !om.class.type<@Foo>
  %unknownDistinct = om.prop.eq %unknown, %otherUnknown : !om.class.type<@Foo>
  // CHECK: om.class.fields [[TRUE]], [[FALSE]], [[EQ]], [[TRUE]], [[FALSE]], [[BEQ]], [[TRUE]], [[TRUE]], [[FALSE]], [[IEQ]],
  // CHECK-SAME: [[TRUE]], [[FALSE]], [[TRUE]], [[FALSE]], [[TRUE]], [[MAY_ALIAS]], [[TRUE]], [[TRUE]], [[TRUE]], [[UNKNOWN_DISTINCT]]
  om.class.fields %0, %1, %2, %3, %4, %5, %6, %7, %8, %9,
      %equal, %notEqual, %anyEqual, %anyNotEqual, %parameter, %mayAlias,
      %unknownEqual, %anyParameter, %unknownAlias, %unknownDistinct :
      i1, i1, i1, i1, i1, i1, i1, i1, i1, i1,
      i1, i1, i1, i1, i1, i1, i1, i1, i1, i1
}

// CHECK-LABEL: @AnyCastFold
om.class @AnyCastFold(%ref: !om.class.type<@Foo>, %any: !om.any) -> (
    concrete: !om.any, identity: !om.any, nested: !om.any) {
  // CHECK-NEXT: [[CAST:%.+]] = om.any_cast %ref : (!om.class.type<@Foo>) -> !om.any
  %cast = om.any_cast %ref : (!om.class.type<@Foo>) -> !om.any
  %identity = om.any_cast %any : (!om.any) -> !om.any
  %nested = om.any_cast %cast : (!om.any) -> !om.any
  // CHECK-NEXT: om.class.fields [[CAST]], %any, [[CAST]] : !om.any, !om.any, !om.any
  om.class.fields %cast, %identity, %nested : !om.any, !om.any, !om.any
}

// Strip casts only when the resulting operands still have matching types.
// CHECK-LABEL: @PropEqAnyCast
om.class @PropEqAnyCast(%ref: !om.class.type<@Foo>,
                        %other: !om.class.type<@Foo>, %any: !om.any,
                        %different: !om.class.type<@FooWithAssert>) -> (
    lhs: i1, rhs: i1, both: i1, different: i1) {
  // CHECK-DAG: [[CAST:%.+]] = om.any_cast %ref : (!om.class.type<@Foo>) -> !om.any
  // CHECK-DAG: [[DIFFERENT_CAST:%.+]] = om.any_cast %different : (!om.class.type<@FooWithAssert>) -> !om.any
  %cast = om.any_cast %ref : (!om.class.type<@Foo>) -> !om.any
  %otherCast = om.any_cast %other : (!om.class.type<@Foo>) -> !om.any
  %differentCast = om.any_cast %different : (!om.class.type<@FooWithAssert>) -> !om.any
  // CHECK: [[LHS:%.+]] = om.prop.eq [[CAST]], %any : !om.any
  %lhs = om.prop.eq %cast, %any : !om.any
  // CHECK: [[RHS:%.+]] = om.prop.eq %any, [[CAST]] : !om.any
  %rhs = om.prop.eq %any, %cast : !om.any
  // CHECK: [[BOTH:%.+]] = om.prop.eq %ref, %other : !om.class.type<@Foo>
  %both = om.prop.eq %cast, %otherCast : !om.any
  // CHECK: [[DIFFERENT:%.+]] = om.prop.eq [[CAST]], [[DIFFERENT_CAST]] : !om.any
  %differentEq = om.prop.eq %cast, %differentCast : !om.any
  // CHECK: om.class.fields [[LHS]], [[RHS]], [[BOTH]], [[DIFFERENT]] : i1, i1, i1, i1
  om.class.fields %lhs, %rhs, %both, %differentEq : i1, i1, i1, i1
}

// Cyclic any casts canonicalize to the same SSA reference. Self-equality folds
// to true, but stripping casts from an opaque comparison must not loop.
// CHECK-LABEL: @PropEqCastCycle
om.class @PropEqCastCycle(%any: !om.any) -> (equal: i1, opaque: i1) {
  // CHECK-DAG: [[TRUE:%.+]] = om.constant true
  // CHECK-DAG: [[CAST:%.+]] = om.any_cast %{{.+}} : (!om.any) -> !om.any
  %a = om.any_cast %b : (!om.any) -> !om.any
  %b = om.any_cast %a : (!om.any) -> !om.any
  %equal = om.prop.eq %a, %b : !om.any
  // CHECK: [[OPAQUE:%.+]] = om.prop.eq [[CAST]], %any : !om.any{{$}}
  %opaque = om.prop.eq %a, %any : !om.any
  // CHECK: om.class.fields [[TRUE]], [[OPAQUE]] : i1, i1
  om.class.fields %equal, %opaque : i1, i1
}

// CHECK-LABEL: @IntegerBitwiseFold
om.class @IntegerBitwiseFold(%b: i8) -> (out1: i8, out2: i8, out3: i8,
                                          out4: i8, out5: i8, out6: i8,
                                          out7: i8, out8: i8) {
  // CHECK-DAG: [[ZERO:%.+]] = om.constant 0 : i8
  // CHECK-DAG: [[ONES:%.+]] = om.constant -1 : i8
  %zero = om.constant 0 : i8
  %ones = om.constant -1 : i8

  // 0xFF AND 0x00 = 0x00.
  %0 = om.integer.and %ones, %zero : i8

  // 0xFF OR 0x00 = 0xFF.
  %1 = om.integer.or %ones, %zero : i8

  // 0xFF XOR 0x00 = 0xFF.
  %2 = om.integer.xor %ones, %zero : i8

  // 0xFF XOR 0xFF = 0x00.
  %3 = om.integer.xor %ones, %ones : i8

  // Non-constant AND does not fold.
  // CHECK: [[AND:%.+]] = om.integer.and %b, %b : i8
  %4 = om.integer.and %b, %b : i8

  // Non-constant OR does not fold.
  // CHECK: [[OR:%.+]] = om.integer.or %b, %b : i8
  %5 = om.integer.or %b, %b : i8

  // Non-constant XOR does not fold.
  // CHECK: [[XOR:%.+]] = om.integer.xor %b, %b : i8
  %6 = om.integer.xor %b, %b : i8

  // XOR with all-zeros is identity.
  %7 = om.integer.xor %b, %zero : i8

  // CHECK: om.class.fields [[ZERO]], [[ONES]], [[ONES]], [[ZERO]], [[AND]], [[OR]], [[XOR]], %b
  om.class.fields %0, %1, %2, %3, %4, %5, %6, %7 : i8, i8, i8, i8, i8, i8, i8, i8
}

// CHECK-LABEL: @ListCreateFold
// CHECK-SAME: (%0: !om.integer)
om.class @ListCreateFold(%0: !om.integer) -> (
    strings: !om.list<!om.string>, booleans: !om.list<i1>,
    integers: !om.list<!om.integer>, empty: !om.list<!om.string>,
    dynamic: !om.list<!om.integer>, mixed: !om.list<!om.integer>) {
  // Integer lists preserve arbitrary-precision signed values and types.
  %1 = om.constant #om.integer<-36755551979133953793 : si66> : !om.integer
  %2 = om.constant #om.integer<36755551979133953793 : si66> : !om.integer
  // CHECK: [[NEGATIVE:%[0-9]+]] = om.constant #om.integer<-36755551979133953793 : si66> : !om.integer
  // CHECK: [[INTEGERS:%[0-9]+]] = om.constant #om.list<!om.integer, [#om.integer<-36755551979133953793 : si66> : !om.integer, #om.integer<36755551979133953793 : si66> : !om.integer]> : !om.list<!om.integer>
  %3 = om.list_create %1, %2 : !om.integer

  // String lists fold to a list attribute.
  %4 = om.constant "a" : !om.string
  %5 = om.constant "b" : !om.string
  // CHECK: [[STRINGS:%[0-9]+]] = om.constant #om.list<!om.string, ["a" : !om.string, "b" : !om.string]> : !om.list<!om.string>
  %6 = om.list_create %4, %5 : !om.string

  // Boolean lists fold to a list attribute.
  %7 = om.constant true
  %8 = om.constant false
  // CHECK: [[BOOLEANS:%[0-9]+]] = om.constant #om.list<i1, [true, false]> : !om.list<i1>
  %9 = om.list_create %7, %8 : i1

  // Empty string lists fold to an empty list attribute.
  // CHECK: [[EMPTY:%[0-9]+]] = om.constant #om.list<!om.string, []> : !om.list<!om.string>
  %10 = om.list_create : !om.string

  // Nonconstant list elements keep the list_create operation.
  // CHECK: [[DYNAMIC:%[0-9]+]] = om.list_create [[ARG:%_0]] : !om.integer
  %11 = om.list_create %0 : !om.integer

  // Mixed constant and nonconstant inputs do not fold.
  // CHECK: [[MIXED:%[0-9]+]] = om.list_create [[NEGATIVE]], [[ARG]] : !om.integer
  %12 = om.list_create %1, %0 : !om.integer

  // CHECK: om.class.fields [[STRINGS]], [[BOOLEANS]], [[INTEGERS]], [[EMPTY]], [[DYNAMIC]], [[MIXED]]
  om.class.fields %6, %9, %3, %10, %11, %12 :
      !om.list<!om.string>, !om.list<i1>, !om.list<!om.integer>,
      !om.list<!om.string>, !om.list<!om.integer>, !om.list<!om.integer>
}

// CHECK-LABEL: @ListConcatFold
om.class @ListConcatFold(
    %0: !om.list<!om.string>, %1: !om.list<!om.string>,
    %a: !om.class.type<@Foo>, %b: !om.class.type<@Foo>, %c: !om.class.type<@Foo>,
    %x: !om.string, %y: !om.string) -> (
    joined: !om.list<!om.string>, empty: !om.list<!om.string>,
    single_constant: !om.list<!om.string>, single_dynamic: !om.list<!om.string>,
    nested: !om.list<!om.string>, dynamic: !om.list<!om.string>,
    class_joined: !om.list<!om.class.type<@Foo>>,
    class_nested: !om.list<!om.class.type<@Foo>>,
    original: !om.list<!om.class.type<@Foo>>,
    string_joined: !om.list<!om.string>, mixed: !om.list<!om.string>) {
  // Fold constant lists without relying on list_create.
  %2 = om.constant #om.list<!om.string, ["a" : !om.string]>
      : !om.list<!om.string>
  %3 = om.constant #om.list<!om.string, ["b" : !om.string]>
      : !om.list<!om.string>
  %4 = om.constant #om.list<!om.string, ["c" : !om.string]>
      : !om.list<!om.string>
  // CHECK-DAG: [[LIST_A:%[0-9]+]] = om.constant #om.list<!om.string, ["a" : !om.string]> : !om.list<!om.string>{{$}}
  // CHECK-DAG: [[LIST_B:%[0-9]+]] = om.constant #om.list<!om.string, ["b" : !om.string]> : !om.list<!om.string>{{$}}
  // CHECK-DAG: [[JOINED:%[0-9]+]] = om.constant #om.list<!om.string, ["a" : !om.string, "b" : !om.string, "c" : !om.string]> : !om.list<!om.string>{{$}}
  %5 = om.list_concat %2, %3, %4 : !om.list<!om.string>

  // Empty lists concatenate to an empty list.
  %6 = om.constant #om.list<!om.string, []> : !om.list<!om.string>
  %7 = om.constant #om.list<!om.string, []> : !om.list<!om.string>
  // CHECK-DAG: [[EMPTY:%[0-9]+]] = om.constant #om.list<!om.string, []> : !om.list<!om.string>{{$}}
  %8 = om.list_concat %6, %7 : !om.list<!om.string>

  // A single constant list is unchanged.
  %9 = om.list_concat %2 : !om.list<!om.string>

  // A single list parameter is unchanged.
  %10 = om.list_concat %0 : !om.list<!om.string>

  // Fold nested constant concatenations.
  %11 = om.list_concat %2, %3 : !om.list<!om.string>
  %12 = om.list_concat %11, %4 : !om.list<!om.string>

  // Keep concatenations of list parameters.
  // CHECK: [[DYNAMIC_CONCAT:%[0-9]+]] = om.list_concat [[ARG0:%_0]], [[ARG1:%_1]] : <!om.string>{{$}}
  %13 = om.list_concat %0, %1 : !om.list<!om.string>

  // Flatten list_create operands without changing element order.
  // CHECK-DAG: [[ORIGINAL:%.+]] = om.list_create %a : !om.class.type<@Foo>{{$}}
  // CHECK-DAG: [[CLASS_JOINED:%.+]] = om.list_create %a, %b : !om.class.type<@Foo>{{$}}
  // CHECK-DAG: [[CLASS_NESTED:%.+]] = om.list_create %a, %b, %c, %a, %b : !om.class.type<@Foo>{{$}}
  %head = om.list_create %a : !om.class.type<@Foo>
  %middle = om.list_create %b : !om.class.type<@Foo>
  %tail = om.list_create %c : !om.class.type<@Foo>
  // Ignore empty lists, whether constructed or constant.
  %empty = om.list_create : !om.class.type<@Foo>
  %emptyConstant = om.constant #om.list<!om.class.type<@Foo>, []> : !om.list<!om.class.type<@Foo>>
  %joined = om.list_concat %empty, %head, %emptyConstant, %middle, %empty : !om.list<!om.class.type<@Foo>>
  // Flatten a reused concat without dropping repeated elements.
  %nested = om.list_concat %joined, %tail, %joined : !om.list<!om.class.type<@Foo>>

  // Flatten lists of nonconstant strings.
  // CHECK-DAG: [[STRING_HEAD:%.+]] = om.list_create %x : !om.string{{$}}
  // CHECK-DAG: [[STRING_JOINED:%.+]] = om.list_create %x, %y : !om.string{{$}}
  %stringHead = om.list_create %x : !om.string
  %stringTail = om.list_create %y : !om.string
  %stringJoined = om.list_concat %stringHead, %6, %stringTail : !om.list<!om.string>
  // A nonempty constant prevents flattening.
  // CHECK: [[MIXED:%.+]] = om.list_concat [[STRING_HEAD]], [[LIST_B]] : <!om.string>{{$}}
  %mixed = om.list_concat %stringHead, %3 : !om.list<!om.string>

  // CHECK: om.class.fields [[JOINED]], [[EMPTY]], [[LIST_A]], [[ARG0]], [[JOINED]], [[DYNAMIC_CONCAT]],
  // CHECK-SAME: [[CLASS_JOINED]], [[CLASS_NESTED]], [[ORIGINAL]], [[STRING_JOINED]], [[MIXED]]
  // CHECK-SAME: : !om.list<!om.string>, !om.list<!om.string>, !om.list<!om.string>,
  // CHECK-SAME: !om.list<!om.string>, !om.list<!om.string>, !om.list<!om.string>,
  // CHECK-SAME: !om.list<!om.class.type<@Foo>>, !om.list<!om.class.type<@Foo>>, !om.list<!om.class.type<@Foo>>,
  // CHECK-SAME: !om.list<!om.string>, !om.list<!om.string>
  om.class.fields %5, %8, %9, %10, %12, %13, %joined, %nested, %head,
      %stringJoined, %mixed :
      !om.list<!om.string>, !om.list<!om.string>, !om.list<!om.string>,
      !om.list<!om.string>, !om.list<!om.string>, !om.list<!om.string>,
      !om.list<!om.class.type<@Foo>>, !om.list<!om.class.type<@Foo>>,
      !om.list<!om.class.type<@Foo>>, !om.list<!om.string>, !om.list<!om.string>
}

// CHECK-LABEL: @ListContainsFold
om.class @ListContainsFold(%empty_search: !om.string,
                           %other: !om.class.type<@Foo>, %opaque: !om.any,
                           %x: !om.integer, %y: !om.integer, %z: !om.integer) -> (
    string_found: i1, string_absent: i1, bool_found: i1, bool_absent: i1,
    integer_found: i1, integer_absent: i1, empty: i1,
    class_found: i1, class_absent: i1, class_unknown: i1,
    any_found: i1, any_absent: i1, any_unknown: i1, any_empty: i1,
    dynamic_found: i1, dynamic_unknown: i1) {
  %string_list = om.constant #om.list<!om.string, ["a" : !om.string, "b" : !om.string]> : !om.list<!om.string>
  %string_found_value = om.constant "b" : !om.string
  %string_absent_value = om.constant "absent" : !om.string
  %string_found = om.list_contains %string_list, %string_found_value : !om.list<!om.string>
  %string_absent = om.list_contains %string_list, %string_absent_value : !om.list<!om.string>

  %bool_list = om.constant #om.list<i1, [true]> : !om.list<i1>
  %true = om.constant true
  %false = om.constant false
  %bool_found = om.list_contains %bool_list, %true : !om.list<i1>
  %bool_absent = om.list_contains %bool_list, %false : !om.list<i1>

  %integer_list = om.constant #om.list<!om.integer, [#om.integer<42 : si64>]> : !om.list<!om.integer>
  %integer_found_value = om.constant #om.integer<42 : si64> : !om.integer
  %integer_absent_value = om.constant #om.integer<0 : si64> : !om.integer
  %integer_found = om.list_contains %integer_list, %integer_found_value : !om.list<!om.integer>
  %integer_absent = om.list_contains %integer_list, %integer_absent_value : !om.list<!om.integer>

  // An empty list returns false for a nonconstant search value.
  %empty_list = om.constant #om.list<!om.string, []> : !om.list<!om.string>
  %empty = om.list_contains %empty_list, %empty_search : !om.list<!om.string>

  // Compare object identities, not fields.
  %a = om.elaborated_object @Foo() : () -> !om.class.type<@Foo>
  %b = om.elaborated_object @Foo() : () -> !om.class.type<@Foo>
  %aRef = om.any_cast %a : (!om.class.type<@Foo>) -> !om.any
  %bRef = om.any_cast %b : (!om.class.type<@Foo>) -> !om.any
  %objects = om.list_create %other, %a : !om.class.type<@Foo>
  %knownObjects = om.list_create %a : !om.class.type<@Foo>
  %refs = om.list_create %opaque, %aRef : !om.any
  %knownRefs = om.list_create %aRef : !om.any

  // CHECK-DAG: [[TRUE:%.+]] = om.constant true
  // CHECK-DAG: [[FALSE:%.+]] = om.constant false
  // CHECK-DAG: [[OBJECTS:%.+]] = om.list_create %other, %{{.+}} : !om.class.type<@Foo>{{$}}
  // CHECK-DAG: [[REFS:%.+]] = om.list_create %opaque, %{{.+}} : !om.any{{$}}
  // A match after an unresolved comparison still returns true.
  %classFound = om.list_contains %objects, %a : !om.list<!om.class.type<@Foo>>
  %classAbsent = om.list_contains %knownObjects, %b : !om.list<!om.class.type<@Foo>>
  // A parameter may alias the search object.
  // CHECK: [[CLASS_UNKNOWN:%.+]] = om.list_contains [[OBJECTS]], %{{.+}} : <!om.class.type<@Foo>>{{$}}
  %classUnknown = om.list_contains %objects, %b : !om.list<!om.class.type<@Foo>>
  %anyFound = om.list_contains %refs, %aRef : !om.list<!om.any>
  %anyAbsent = om.list_contains %knownRefs, %bRef : !om.list<!om.any>
  // CHECK: [[ANY_UNKNOWN:%.+]] = om.list_contains [[REFS]], %{{.+}} : <!om.any>{{$}}
  %anyUnknown = om.list_contains %refs, %bRef : !om.list<!om.any>

  // Empty lists return false for opaque references.
  %emptyRefs = om.list_create : !om.any
  %anyEmpty = om.list_contains %emptyRefs, %opaque : !om.list<!om.any>

  // An SSA-identical element matches even after an unresolved comparison.
  // CHECK: [[DYNAMIC_LIST:%.+]] = om.list_create %y, %x : !om.integer{{$}}
  %dynamic_list = om.list_create %y, %x : !om.integer
  %dynamic_found = om.list_contains %dynamic_list, %x : !om.list<!om.integer>
  // Distinct SSA values may be equal.
  // CHECK: [[DYNAMIC_UNKNOWN:%.+]] = om.list_contains [[DYNAMIC_LIST]], %z : <!om.integer>{{$}}
  %dynamic_unknown = om.list_contains %dynamic_list, %z : !om.list<!om.integer>

  // CHECK: om.class.fields [[TRUE]], [[FALSE]], [[TRUE]], [[FALSE]], [[TRUE]], [[FALSE]], [[FALSE]],
  // CHECK-SAME: [[TRUE]], [[FALSE]], [[CLASS_UNKNOWN]], [[TRUE]], [[FALSE]], [[ANY_UNKNOWN]], [[FALSE]],
  // CHECK-SAME: [[TRUE]], [[DYNAMIC_UNKNOWN]]
  // CHECK-SAME: : i1, i1, i1, i1, i1, i1, i1, i1, i1, i1, i1, i1, i1, i1, i1, i1
  om.class.fields %string_found, %string_absent, %bool_found, %bool_absent,
      %integer_found, %integer_absent, %empty, %classFound, %classAbsent,
      %classUnknown, %anyFound, %anyAbsent, %anyUnknown, %anyEmpty,
      %dynamic_found, %dynamic_unknown :
      i1, i1, i1, i1, i1, i1, i1, i1, i1, i1, i1, i1, i1, i1, i1, i1
}
