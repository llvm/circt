// RUN: circt-opt --cse --canonicalize %s | FileCheck %s

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

// CHECK-LABEL: @IntegerPropertyBitwiseFold
om.class @IntegerPropertyBitwiseFold() -> (andResult: !om.integer, orResult: !om.integer,
                                  notResult: !om.integer) {
  %neg2 = om.constant #om.integer<-2 : si66> : !om.integer
  // Exercise values beyond 64 bits.
  %wide = om.constant #om.integer<18446744073709551621 : si66> : !om.integer

  // CHECK-DAG: [[AND:%.+]] = om.constant #om.integer<18446744073709551620 : si66> : !om.integer
  %and = om.integer.and %neg2, %wide : !om.integer

  // CHECK-DAG: [[OR:%.+]] = om.constant #om.integer<-1 : si66> : !om.integer
  %or = om.integer.or %neg2, %wide : !om.integer

  // CHECK-DAG: [[NOT:%.+]] = om.constant #om.integer<1 : si66> : !om.integer
  %not = om.integer.not %neg2 : !om.integer

  // CHECK: om.class.fields [[AND]], [[OR]], [[NOT]]
  om.class.fields %and, %or, %not : !om.integer, !om.integer, !om.integer
}

// CHECK-LABEL: @IntegerPropertyBitwiseDynamic
om.class @IntegerPropertyBitwiseDynamic(%input: !om.integer) ->
    (nestedResult: !om.integer, mixedResult: !om.integer,
     andZero: !om.integer, zeroAnd: !om.integer, andOnes: !om.integer, onesAnd: !om.integer,
     orZero: !om.integer, zeroOr: !om.integer, orOnes: !om.integer, onesOr: !om.integer) {
  %neg2 = om.constant #om.integer<-2 : si4> : !om.integer
  %five = om.constant #om.integer<5 : si8> : !om.integer
  // CHECK-DAG: [[ZERO:%.+]] = om.constant #om.integer<0 : si4> : !om.integer
  // CHECK-DAG: [[ONES:%.+]] = om.constant #om.integer<-1 : si4> : !om.integer
  %zero = om.constant #om.integer<0 : si4> : !om.integer
  %ones = om.constant #om.integer<-1 : si4> : !om.integer

  // Different stored widths must sign-extend before bitwise folding.
  // CHECK-DAG: [[MIXED:%.+]] = om.constant #om.integer<4 : si8> : !om.integer
  %mixed = om.integer.and %neg2, %five : !om.integer

  // CHECK: [[OR:%.+]] = om.integer.or %input, %{{.+}} : !om.integer
  %or = om.integer.or %input, %five : !om.integer
  // CHECK: [[NOT:%.+]] = om.integer.not %input : !om.integer
  %not = om.integer.not %input : !om.integer
  // CHECK: [[NESTED:%.+]] = om.integer.and [[OR]], [[NOT]] : !om.integer
  %nested = om.integer.and %or, %not : !om.integer

  // Zero/all-ones folds work with a non-constant operand on either side.
  %andZero = om.integer.and %input, %zero : !om.integer
  %zeroAnd = om.integer.and %zero, %input : !om.integer
  %andOnes = om.integer.and %input, %ones : !om.integer
  %onesAnd = om.integer.and %ones, %input : !om.integer
  %orZero = om.integer.or %input, %zero : !om.integer
  %zeroOr = om.integer.or %zero, %input : !om.integer
  %orOnes = om.integer.or %input, %ones : !om.integer
  %onesOr = om.integer.or %ones, %input : !om.integer

  // CHECK: om.class.fields [[NESTED]], [[MIXED]], [[ZERO]], [[ZERO]], %input, %input, %input, %input, [[ONES]], [[ONES]]
  om.class.fields %nested, %mixed, %andZero, %zeroAnd, %andOnes, %onesAnd, %orZero, %zeroOr, %orOnes, %onesOr : !om.integer, !om.integer, !om.integer, !om.integer, !om.integer, !om.integer, !om.integer, !om.integer, !om.integer, !om.integer
}

// CHECK-LABEL: @PropEqFold
om.class @PropEqFold(%str: !om.string, %b: i1, %n: !om.integer) -> (out1: i1, out2: i1,
                                                                     out3: i1, out4: i1,
                                                                     out5: i1, out6: i1,
                                                                     out7: i1, out8: i1,
                                                                     out9: i1, out10: i1) {
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

  // CHECK: om.class.fields [[TRUE]], [[FALSE]], [[EQ]], [[TRUE]], [[FALSE]], [[BEQ]], [[TRUE]], [[TRUE]], [[FALSE]], [[IEQ]]
  om.class.fields %0, %1, %2, %3, %4, %5, %6, %7, %8, %9 : i1, i1, i1, i1, i1, i1, i1, i1, i1, i1
}

// CHECK-LABEL: @IntegerBitwiseFold
om.class @IntegerBitwiseFold(%b: i8) -> (out1: i8, out2: i8, out3: i8,
                                          out4: i8, out5: i8, out6: i8,
                                          out7: i8, out8: i8, out9: i8, out10: i8,
                                          out11: i8, out12: i8, out13: i8, out14: i8,
                                          out15: i8, out16: i8) {
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

  // Zero/all-ones folds preserve fixed-width types on either side.
  %8 = om.integer.and %b, %zero : i8
  %9 = om.integer.and %zero, %b : i8
  %10 = om.integer.and %b, %ones : i8
  %11 = om.integer.and %ones, %b : i8
  %12 = om.integer.or %b, %zero : i8
  %13 = om.integer.or %zero, %b : i8
  %14 = om.integer.or %b, %ones : i8
  %15 = om.integer.or %ones, %b : i8

  // CHECK: om.class.fields [[ZERO]], [[ONES]], [[ONES]], [[ZERO]], [[AND]], [[OR]], [[XOR]], %b, [[ZERO]], [[ZERO]], %b, %b, %b, %b, [[ONES]], [[ONES]]
  om.class.fields %0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15 : i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8
}
