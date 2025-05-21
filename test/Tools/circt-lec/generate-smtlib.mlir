// RUN: circt-lec %s --c1 foo1 --c2 foo2 --emit-smtlib | FileCheck %s

hw.module @foo1(in %a : i8, in %b : i8, out c : i8) {
  %add = comb.add %a, %b: i8
  hw.output %add : i8
}

hw.module @foo2(in %a : i8, in %b : i8, out c : i8) {
  %add = comb.add %b, %a: i8
  hw.output %add : i8
}


// CHECK:      ; solver scope 0
// CHECK-NEXT: (declare-const [[TMP:.+]] (_ BitVec 8))
// CHECK-NEXT: (declare-const [[TMP0:.+]] (_ BitVec 8))
// CHECK-NEXT: (assert (let (([[TMP1:.+]] (bvadd [[TMP0]] [[TMP]])))
// CHECK-NEXT:         (let (([[TMP2:.+]] (bvadd tmp [[TMP0]])))
// CHECK-NEXT:         (let (([[TMP3:.+]] (distinct [[TMP2]] [[TMP1]])))
// CHECK-NEXT:         [[TMP3]]))))
// CHECK-NEXT: (check-sat)
// CHECK-NEXT: (reset)
