// RUN: circt-opt %s | FileCheck %s
// RUN: circt-opt %s --lower-ice40-to-blif | FileCheck --check-prefix=LOWER %s

func.func @basic(%i : i1) {
    // LOWER: blif.library_gate
    // LOWER-SAME: SB_LUT4
    // CHECK: ice40.sb_lut4
    %a = ice40.sb_lut4 %i, %i, %i, %i, [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1]
    // LOWER: blif.library_gate
    // LOWER-SAME: SB_CARRY
    // CHECK: ice40.sb_carry
    %b = ice40.sb_carry %i, %i, %a
    // LOWER: blif.library_gate
    // LOWER-SAME: SB_LUT4
    // LOWER: blif.library_gate
    // LOWER-SAME: SB_CARRY
    // CHECK: ice40.sb_lut4_carry
    %c, %d = ice40.sb_lut4_carry %i, %i, %a, %a, %b, [1,0,1,1,0,1,1,1,0,1,1,1,1,0,1,1]
    func.return
}
