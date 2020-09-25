// RUN: esic %s | esic | FileCheck %s

// CHECK-LABEL: module
module {
    // CHECK-LABEL: func @list1(%{{.*}}: !esi.list<ui1>)
    func @list1(%A: !esi.list<ui1>) {
        return
    }
}
