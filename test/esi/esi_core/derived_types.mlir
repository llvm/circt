// RUN: circt-opt %s | circt-opt | FileCheck %s

// CHECK-LABEL: module
module {
    // CHECK-LABEL: func @list1(%{{.*}}: !esi.list<ui1>)
    func @list1(%A: !esi.list<ui1>) {
        return
    }

    // CHECK-LABEL: func @ptr(%{{.*}}: !esi.ptr<i4, false>)
    func @ptr(%a: !esi.ptr<i4, false>) {
        return
    }

    // CHECK-LABEL: func @asciiStr(%{{.*}}: !esi.string<ascii>)
    func @asciiStr(%a: !esi.string<ascii>) {
        return
    }
    // CHECK-LABEL: func @utf8Str(%{{.*}}: !esi.string<utf8>)
    func @utf8Str(%a: !esi.string<UTF8>) {
        return
    }
    // CHECK-LABEL: func @utf16Str(%{{.*}}: !esi.string<utf16>)
    func @utf16Str(%a: !esi.string<Utf16>) {
        return
    }
    // CHECK-LABEL: func @utf32Str(%{{.*}}: !esi.string<utf32>)
    func @utf32Str(%a: !esi.string<utF32>) {
        return
    }
}
