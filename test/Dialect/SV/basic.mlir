// RUN: circt-opt %s | FileCheck %s
// RUN: circt-opt %s | circt-opt | FileCheck %s

// CHECK-LABEL: hw.module @test1(%arg0: i1, %arg1: i1, %arg8: i8) {
hw.module @test1(%arg0: i1, %arg1: i1, %arg8: i8) {

  // CHECK: %param_x = sv.localparam : i42 {value = 11 : i42}
  %param_x = sv.localparam : i42 {value = 11 : i42}


  // This corresponds to this block of system verilog code:
  //    always @(posedge arg0) begin
  //      `ifndef SYNTHESIS
  //         if (`PRINTF_COND_ && arg1) $fwrite(32'h80000002, "Hi\n");
  //      `endif
  //    end // always @(posedge)

  sv.always posedge  %arg0 {
    sv.ifdef.procedural "SYNTHESIS" {
    } else {
      %tmp = sv.verbatim.expr "PRINTF_COND_" : () -> i1
      %tmpx = sv.constantX : i1
      %tmpz = sv.constantZ : i1
      %tmp2 = comb.and %tmp, %tmpx, %tmpz, %arg1 : i1
      sv.if %tmp2 {
        sv.fwrite "Hi\n"
      }
      sv.if %tmp2 {
        // Test fwrite with operands.
        sv.fwrite "%x"(%tmp2) : i1
      } else {
        sv.fwrite "There\n"
      }
    }
  }

  // CHECK-NEXT: sv.always posedge %arg0 {
  // CHECK-NEXT:   sv.ifdef.procedural "SYNTHESIS" {
  // CHECK-NEXT:   } else {
  // CHECK-NEXT:     %PRINTF_COND_ = sv.verbatim.expr "PRINTF_COND_" : () -> i1
  // CHECK-NEXT:     %x_i1 = sv.constantX : i1
  // CHECK-NEXT:     %z_i1 = sv.constantZ : i1
  // CHECK-NEXT:     %0 = comb.and %PRINTF_COND_, %x_i1, %z_i1, %arg1 : i1
  // CHECK-NEXT:     sv.if %0 {
  // CHECK-NEXT:       sv.fwrite "Hi\0A"
  // CHECK-NEXT:     }
  // CHECK-NEXT:     sv.if %0 {
  // CHECK-NEXT:       sv.fwrite "%x"(%0) : i1
  // CHECK-NEXT:     } else {
  // CHECK-NEXT:       sv.fwrite "There\0A"
  // CHECK-NEXT:     }
  // CHECK-NEXT:   }
  // CHECK-NEXT: }

  sv.alwaysff(posedge %arg0) {
    sv.fwrite "Yo\n"
  }

  // CHECK-NEXT: sv.alwaysff(posedge %arg0)  {
  // CHECK-NEXT:   sv.fwrite "Yo\0A"
  // CHECK-NEXT: }

  sv.alwaysff(posedge %arg0) {
    sv.fwrite "Sync Main Block\n"
  } ( syncreset : posedge %arg1) {
    sv.fwrite "Sync Reset Block\n"
  }

  // CHECK-NEXT: sv.alwaysff(posedge %arg0) {
  // CHECK-NEXT:   sv.fwrite "Sync Main Block\0A"
  // CHECK-NEXT:  }(syncreset : posedge %arg1) {
  // CHECK-NEXT:   sv.fwrite "Sync Reset Block\0A"
  // CHECK-NEXT: }

  sv.alwaysff (posedge %arg0) {
    sv.fwrite "Async Main Block\n"
  } ( asyncreset : negedge %arg1) {
    sv.fwrite "Async Reset Block\n"
  }

  // CHECK-NEXT: sv.alwaysff(posedge %arg0) {
  // CHECK-NEXT:   sv.fwrite "Async Main Block\0A"
  // CHECK-NEXT:  }(asyncreset : negedge %arg1) {
  // CHECK-NEXT:   sv.fwrite "Async Reset Block\0A"
  // CHECK-NEXT: }

// Smoke test generic syntax.
  sv.initial {
    "sv.if"(%arg0) ( {
      ^bb0:
    }, {
    }) : (i1) -> ()
  }

  // CHECK-NEXT:     sv.initial {
  // CHECK-NEXT:       sv.if %arg0 {
  // CHECK-NEXT:       }
  // CHECK-NEXT:     }

  // CHECK-NEXT: sv.initial {
  // CHECK-NEXT:   sv.casez %arg8 : i8
  // CHECK-NEXT:   case b0000001x: {
  // CHECK-NEXT:     sv.fwrite "x"
  // CHECK-NEXT:   }
  // CHECK-NEXT:   case b000000x1: {
  // CHECK-NEXT:     sv.fwrite "y"
  // CHECK-NEXT:   }
  // CHECK-NEXT:   default: {
  // CHECK-NEXT:     sv.fwrite "z"
  // CHECK-NEXT:   }
  // CHECK-NEXT: }
  sv.initial {
    sv.casez %arg8 : i8
    case b0000001x: {
      sv.fwrite "x"
    }
    case b000000x1: {
      sv.fwrite "y"
    }
    default: {
      sv.fwrite "z"
    }
  }

  // CHECK-NEXT: sv.initial {
  // CHECK-NEXT:   sv.casez %arg1 : i1
  // CHECK-NEXT:   case b0: {
  // CHECK-NEXT:     sv.fwrite "zero"
  // CHECK-NEXT:   }
  // CHECK-NEXT:   case b1: {
  // CHECK-NEXT:     sv.fwrite "one"
  // CHECK-NEXT:   }
  // CHECK-NEXT: }
  sv.initial {
    sv.casez %arg1 : i1
    case b0: {
      sv.fwrite "zero"
    }
    case b1: {
      sv.fwrite "one"
    }
  }

  // CHECK-NEXT: %combWire = sv.reg : !hw.inout<i1>
  %combWire = sv.reg : !hw.inout<i1>
  // CHECK-NEXT: %selReg = sv.reg : !hw.inout<i10>
  %selReg = sv.reg : !hw.inout<i10>
  // CHECK-NEXT: %combWire2 = sv.wire : !hw.inout<i1>
  %combWire2 = sv.wire : !hw.inout<i1>
  // CHECK-NEXT: %regForce = sv.reg : !hw.inout<i1>
  %regForce = sv.reg  : !hw.inout<i1>
  // CHECK-NEXT: sv.alwayscomb {
  sv.alwayscomb {
    // CHECK-NEXT: %x_i1 = sv.constantX : i1
    %tmpx = sv.constantX : i1
    // CHECK-NEXT: sv.passign %combWire, %x_i1 : i1
    sv.passign %combWire, %tmpx : i1
    // CHECK-NEXT: %[[c2_i3:.+]] = hw.constant 2 : i3
    // CHECK-NEXT: %[[v0:.+]] = sv.indexed_part_select_inout %selReg[%[[c2_i3]] : 1] : !hw.inout<i10>, i3
    // CHECK-NEXT: sv.passign %[[v0]], %x_i1 : i1
    %c2 = hw.constant 2 : i3
    %xx1 = sv.indexed_part_select_inout %selReg[%c2:1] :  !hw.inout<i10>, i3
    sv.passign %xx1, %tmpx : i1
    // CHECK-NEXT: sv.force %combWire2, %x_i1 : i1
    sv.force %combWire2, %tmpx : i1
    // CHECK-NEXT: sv.force %regForce, %x_i1 : i1
    sv.force %regForce, %tmpx : i1
    sv.release %combWire2 : !hw.inout<i1>
    sv.release %regForce : !hw.inout<i1>
    // CHECK-NEXT: sv.release %combWire2 : !hw.inout<i1>
    // CHECK-NEXT: sv.release %regForce : !hw.inout<i1>
    // CHECK-NEXT: }
  }

  // CHECK-NEXT: %reg23 = sv.reg : !hw.inout<i23>
  // CHECK-NEXT: %regStruct23 = sv.reg : !hw.inout<struct<foo: i23>>
  // CHECK-NEXT: %reg24 = sv.reg sym @regSym1 : !hw.inout<i23>
  // CHECK-NEXT: %wire25 = sv.wire sym @wireSym1 : !hw.inout<i23>
  %reg23       = sv.reg  : !hw.inout<i23>
  %regStruct23 = sv.reg  : !hw.inout<struct<foo: i23>>
  %reg24       = sv.reg sym @regSym1 : !hw.inout<i23>
  %wire25      = sv.wire sym @wireSym1 : !hw.inout<i23>

  // Simulation Control Tasks
  // CHECK-NEXT: sv.initial {
  // CHECK-NEXT: sv.stop 1
  // CHECK-NEXT: sv.finish 1
  // CHECK-NEXT: sv.exit
  // CHECK-NEXT: }
  sv.initial {
    sv.stop 1
    sv.finish 1
    sv.exit
  }

  // Severity Message Tasks
  // CHECK-NEXT: sv.initial {
  // CHECK-NEXT: sv.fatal 1
  // CHECK-NEXT: sv.fatal 1, "hello"
  // CHECK-NEXT: sv.fatal 1, "hello %d"(%arg0) : i1
  // CHECK-NEXT: sv.error
  // CHECK-NEXT: sv.error "hello"
  // CHECK-NEXT: sv.error "hello %d"(%arg0) : i1
  // CHECK-NEXT: sv.warning
  // CHECK-NEXT: sv.warning "hello"
  // CHECK-NEXT: sv.warning "hello %d"(%arg0) : i1
  // CHECK-NEXT: sv.info
  // CHECK-NEXT: sv.info "hello"
  // CHECK-NEXT: sv.info "hello %d"(%arg0) : i1
  // CHECK-NEXT: }
  sv.initial {
    sv.fatal 1
    sv.fatal 1, "hello"
    sv.fatal 1, "hello %d"(%arg0) : i1
    sv.error
    sv.error "hello"
    sv.error "hello %d"(%arg0) : i1
    sv.warning
    sv.warning "hello"
    sv.warning "hello %d"(%arg0) : i1
    sv.info
    sv.info "hello"
    sv.info "hello %d"(%arg0) : i1
  }

  // CHECK-NEXT: hw.output
  hw.output
}

//CHECK-LABEL: sv.bind #hw.innerNameRef<@AB::@a1>
//CHECK-NEXT: sv.bind #hw.innerNameRef<@AB::@b1>
sv.bind #hw.innerNameRef<@AB::@a1>
sv.bind #hw.innerNameRef<@AB::@b1>


hw.module.extern @ExternDestMod(%a: i1, %b: i2)
hw.module @InternalDestMod(%a: i1, %b: i2) {}
//CHECK-LABEL: hw.module @AB(%a: i1, %b: i2) {
//CHECK-NEXT:   hw.instance "whatever" sym @a1 @ExternDestMod(a: %a: i1, b: %b: i2) -> () {doNotPrint = 1 : i64}
//CHECK-NEXT:   hw.instance "yo" sym @b1 @InternalDestMod(a: %a: i1, b: %b: i2) -> () {doNotPrint = 1 : i64}

hw.module @AB(%a: i1, %b: i2) {
  hw.instance "whatever" sym @a1 @ExternDestMod(a: %a: i1, b: %b: i2) -> () {doNotPrint=1}
  hw.instance "yo" sym @b1 @InternalDestMod(a: %a: i1, b: %b: i2) -> () {doNotPrint=1}
}

//CHECK-LABEL: hw.module @XMR_src
hw.module @XMR_src(%a : i23) {
  //CHECK-NEXT:   sv.xmr isRooted "a", "b", "c" : !hw.inout<i23>
  %xmr1 = sv.xmr isRooted a,b,c : !hw.inout<i23>
  //CHECK-NEXT:   sv.xmr "a", "b", "c" : !hw.inout<i3>
  %xmr2 = sv.xmr "a",b,c : !hw.inout<i3>
  %r = sv.read_inout %xmr1 : !hw.inout<i23>
  sv.assign %xmr1, %a : i23
}

  hw.module @part_select(%in4 : i4, %in8 : i8) -> (a : i3, b : i5) {
  // CHECK-LABEL: hw.module @part_select

    %myReg2 = sv.reg : !hw.inout<i18>
    %c2_i3 = hw.constant 7 : i4
    // CHECK: = sv.indexed_part_select_inout %myReg2[%[[c7_i4:.+]] : 8] : !hw.inout<i18>, i4
    %a1 = sv.indexed_part_select_inout %myReg2 [%c2_i3:8] : !hw.inout<i18>, i4
    sv.assign %a1, %in8 : i8
    // CHECK:  = sv.indexed_part_select_inout %myReg2[%[[c7_i4]] decrement : 8] : !hw.inout<i18>, i4
    %b1 = sv.indexed_part_select_inout %myReg2 [%c2_i3 decrement:8] : !hw.inout<i18>, i4
    sv.assign %b1, %in8 : i8
    %c3_i3 = hw.constant 3 : i4
    %rc = sv.read_inout %myReg2 : !hw.inout<i18>
    %c = sv.indexed_part_select %rc [%c3_i3:3] : i18, i4
    // CHECK: %[[v2:.+]] = sv.read_inout %myReg2 : !hw.inout<i18>
    // CHECK: = sv.indexed_part_select %[[v2]][%[[c3_i4:.+]] : 3] : i18, i4
    %rd = sv.read_inout %myReg2 : !hw.inout<i18>
    %d = sv.indexed_part_select %rd [%in4 decrement:5] : i18, i4
    // CHECK: %[[v4:.+]] = sv.read_inout %myReg2 : !hw.inout<i18>
    // CHECK: = sv.indexed_part_select %[[v4]][%[[in4:.+]]  decrement : 5] : i18, i4
    hw.output %c, %d : i3, i5
}
