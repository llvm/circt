// RUN: circt-opt %s | circt-opt | FileCheck %s

// CHECK: #moore.fvint<0 : 0>
hw.constant false {foo = #moore.fvint<0 : 0>}
// CHECK: #moore.fvint<42 : 32>
hw.constant false {foo = #moore.fvint<42 : 32>}
// CHECK: #moore.fvint<-42 : 32>
hw.constant false {foo = #moore.fvint<-42 : 32>}
// CHECK: #moore.fvint<1234567890123456789012345678901234567890 : 131>
hw.constant false {foo = #moore.fvint<1234567890123456789012345678901234567890 : 131>}
// CHECK: #moore.fvint<hABCDEFXZ0123456789 : 72>
hw.constant false {foo = #moore.fvint<hABCDEFXZ0123456789 : 72>}
// CHECK: #moore.fvint<b1010XZ01 : 8>
hw.constant false {foo = #moore.fvint<b1010XZ01 : 8>}
