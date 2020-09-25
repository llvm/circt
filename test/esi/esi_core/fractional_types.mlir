// RUN: esic %s | esic | FileCheck %s

// CHECK-LABEL: module
module {
    // CHECK-LABEL: func @fixed(%{{.*}}: !esi.fixed<true, 3, 4>)
    func @fixed(%A: !esi.fixed<true, 3, 4>) {
        // %0 = constant (true, 3, 10) : esi.compound
        %1 = "esi.cast_compound"(%A) : (!esi.fixed<true, 3, 4>) ->  i1
        return
    }

    // CHECK-LABEL: func @float(%{{.*}}: !esi.float<true, 3, 4>)
    func @float(%A: !esi.float<true, 3, 4>) {
        return
    }

    // CHECK-LABEL: func @float_unsigned(%{{.*}}: !esi.float<false, 1, 13>)
    func @float_unsigned(%A: !esi.float<false, 1, 13>) {
        return
    }
}
