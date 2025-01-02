// RUN: circt-opt %s | FileCheck %s

func.func @basic(%i : i1) {
    // CHECK: ice40.sb_lut4
    %a = ice40.sb_lut4 %i, %i, %i, %i, [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1]
    // CHECK: ice40.sb_carry
    %b = ice40.sb_carry %i, %i, %a
    // CHECK: ice40.sb_lut4_carry
    %c, %d = ice40.sb_lut4_carry %i, %i, %a, %a, %b, [1,0,1,1,0,1,1,1,0,1,1,1,1,0,1,1]
    func.return
}
