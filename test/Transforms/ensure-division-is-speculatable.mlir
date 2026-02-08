// RUN: circt-opt --ensure-division-is-speculatable %s | FileCheck %s

// CHECK-LABEL: @divu
// CHECK-DAG: %[[C0:.*]] = hw.constant 0 : i8
// CHECK-DAG: %[[C1:.*]] = hw.constant 1 : i8
// CHECK-DAG: %[[CMP:.*]] = comb.icmp eq %arg1, %[[C0]] : i8
// CHECK-DAG: %[[MUX:.*]] = comb.mux %[[CMP]], %[[C1]], %arg1 : i8
// CHECK: %[[RES:.*]] = comb.divu %arg0, %[[MUX]] {comb.speculatable} : i8
// CHECK: %[[MUX2:.*]] = comb.mux %[[CMP]], %[[C0]], %[[RES]] : i8
// CHECK: return %[[MUX2]] : i8
func.func @divu(%arg0: i8, %arg1: i8) -> i8 {
  %0 = comb.divu %arg0, %arg1 : i8
  return %0 : i8
}

// CHECK-LABEL: @divu_speculatable
// CHECK-DAG: %[[C2:.*]] = hw.constant 2 : i8
// CHECK: %[[RES:.*]] = comb.divu %arg0, %[[C2]] : i8
// CHECK: return %[[RES]] : i8
func.func @divu_speculatable(%arg0: i8) -> i8 {
  %two = hw.constant 2 : i8
  %0 = comb.divu %arg0, %two : i8
  return %0 : i8
}

// CHECK-LABEL: @divs
// CHECK-DAG: %[[C0:.*]] = hw.constant 0 : i8
// CHECK-DAG: %[[C1:.*]] = hw.constant 1 : i8
// CHECK-DAG: %[[CM1:.*]] = hw.constant -1 : i8
// CHECK-DAG: %[[INT_MIN:.*]] = hw.constant -128 : i8
// CHECK-DAG: %[[CMP1:.*]] = comb.icmp eq %arg1, %[[C0]] : i8
// CHECK-DAG: %[[CMP2:.*]] = comb.icmp eq %arg0, %[[INT_MIN]] : i8
// CHECK-DAG: %[[CMP3:.*]] = comb.icmp eq %arg1, %[[CM1]] : i8
// CHECK-DAG: %[[AND:.*]] = comb.and %[[CMP2]], %[[CMP3]] : i1
// CHECK-DAG: %[[OR:.*]] = comb.or %[[CMP1]], %[[AND]] : i1
// CHECK-DAG: %[[MUX:.*]] = comb.mux %[[OR]], %[[C1]], %arg1 : i8
// CHECK: %[[RES:.*]] = comb.divs %arg0, %[[MUX]] {comb.speculatable} : i8
// CHECK: %[[MUX2:.*]] = comb.mux %[[OR]], %[[C0]], %[[RES]] : i8
// CHECK: return %[[MUX2]] : i8
func.func @divs(%arg0: i8, %arg1: i8) -> i8 {
  %0 = comb.divs %arg0, %arg1 : i8
  return %0 : i8
}

// CHECK-LABEL: @modu
// CHECK-DAG: %[[C0:.*]] = hw.constant 0 : i8
// CHECK-DAG: %[[C1:.*]] = hw.constant 1 : i8
// CHECK-DAG: %[[CMP:.*]] = comb.icmp eq %arg1, %[[C0]] : i8
// CHECK-DAG: %[[MUX:.*]] = comb.mux %[[CMP]], %[[C1]], %arg1 : i8
// CHECK: %[[RES:.*]] = comb.modu %arg0, %[[MUX]] {comb.speculatable} : i8
// CHECK: %[[MUX2:.*]] = comb.mux %[[CMP]], %[[C0]], %[[RES]] : i8
// CHECK: return %[[MUX2]] : i8
func.func @modu(%arg0: i8, %arg1: i8) -> i8 {
  %0 = comb.modu %arg0, %arg1 : i8
  return %0 : i8
}

// CHECK-LABEL: @mods
// CHECK-DAG: %[[C0:.*]] = hw.constant 0 : i8
// CHECK-DAG: %[[C1:.*]] = hw.constant 1 : i8
// CHECK-DAG: %[[CM1:.*]] = hw.constant -1 : i8
// CHECK-DAG: %[[INT_MIN:.*]] = hw.constant -128 : i8
// CHECK-DAG: %[[CMP1:.*]] = comb.icmp eq %arg1, %[[C0]] : i8
// CHECK-DAG: %[[CMP2:.*]] = comb.icmp eq %arg0, %[[INT_MIN]] : i8
// CHECK-DAG: %[[CMP3:.*]] = comb.icmp eq %arg1, %[[CM1]] : i8
// CHECK-DAG: %[[AND:.*]] = comb.and %[[CMP2]], %[[CMP3]] : i1
// CHECK-DAG: %[[OR:.*]] = comb.or %[[CMP1]], %[[AND]] : i1
// CHECK-DAG: %[[MUX:.*]] = comb.mux %[[OR]], %[[C1]], %arg1 : i8
// CHECK: %[[RES:.*]] = comb.mods %arg0, %[[MUX]] {comb.speculatable} : i8
// CHECK: %[[MUX2:.*]] = comb.mux %[[OR]], %[[C0]], %[[RES]] : i8
// CHECK: return %[[MUX2]] : i8
func.func @mods(%arg0: i8, %arg1: i8) -> i8 {
  %0 = comb.mods %arg0, %arg1 : i8
  return %0 : i8
}
