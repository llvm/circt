// RUN: circt-opt %s --lower-esi-to-physical | circt-opt -verify-diagnostics | FileCheck %s
// RUN: circt-opt %s --lower-esi-to-physical --lower-esi-ports --hw-flatten-io --lower-esi-to-hw | FileCheck --check-prefix=HW %s

// Lowering of `esi.buffer` with data-delayed channels. The stages carry the
// input data delay.

// CHECK-LABEL:  hw.module @bufferFifoDelay(
// CHECK-NEXT:     %data, %empty = esi.unwrap.fifo %in, [[RDEN:%.+]] : !esi.channel<i16, FIFO(2)>
// CHECK-NEXT:     %chanOutput, %ready = esi.wrap.vr %data, [[VALID:%.+]] : !esi.channel<i16, ValidReady(2)>
// CHECK-NEXT:     [[RDEN]] = comb.and %ready, [[VALID]] : i1
// CHECK:          [[VALID]] = comb.xor %empty, %true : i1
// CHECK-NEXT:     [[STAGE:%.+]] = esi.stage %clk, %rst, %chanOutput {name = "buf_stage0"} : !esi.channel<i16, ValidReady(2)>
// CHECK-NEXT:     %rawOutput, %valid = esi.unwrap.vr [[STAGE]], %rden : !esi.channel<i16, ValidReady(2)>
// CHECK-NEXT:     %chanOutput_0, %rden = esi.wrap.fifo %rawOutput, [[EMPTY:%.+]] : !esi.channel<i16, FIFO(2)>
// CHECK:          [[EMPTY]] = comb.xor %valid, %true_1 : i1
// CHECK-NEXT:     hw.output %chanOutput_0 : !esi.channel<i16, FIFO(2)>
// HW-LABEL:     hw.module @bufferFifoDelay(
// HW:             hw.instance "buf_stage0" @ESI_PipelineStage<WIDTH: ui32 = 16, DATA_DELAY: ui32 = 2>(
hw.module @bufferFifoDelay(in %clk: !seq.clock, in %rst: i1, in %in: !esi.channel<i16, FIFO(2)>, out out: !esi.channel<i16, FIFO(2)>) {
  %0 = esi.buffer %clk, %rst, %in {name = "buf"} : !esi.channel<i16, FIFO(2)> -> !esi.channel<i16, FIFO(2)>
  hw.output %0 : !esi.channel<i16, FIFO(2)>
}

// CHECK-LABEL:  hw.module @bufferVRDelay(
// CHECK-NEXT:     [[S0:%.+]] = esi.stage %clk, %rst, %in : !esi.channel<i16, ValidReady(2)>
// CHECK-NEXT:     [[S1:%.+]] = esi.stage %clk, %rst, [[S0]] : !esi.channel<i16, ValidReady(2)>
// CHECK-NEXT:     hw.output [[S1]] : !esi.channel<i16, ValidReady(2)>
// HW-LABEL:     hw.module @bufferVRDelay(
// HW-NEXT:        hw.instance "pipelineStage" @ESI_PipelineStage<WIDTH: ui32 = 16, DATA_DELAY: ui32 = 2>(clk: %clk: !seq.clock, rst: %rst: i1, a: %in: i16, a_valid: %in_valid: i1,
// HW-NEXT:        hw.instance "pipelineStage" @ESI_PipelineStage<WIDTH: ui32 = 16, DATA_DELAY: ui32 = 2>(
hw.module @bufferVRDelay(in %clk: !seq.clock, in %rst: i1, in %in: !esi.channel<i16, ValidReady(2)>, out out: !esi.channel<i16, ValidReady(2)>) {
  %0 = esi.buffer %clk, %rst, %in {stages = 2} : !esi.channel<i16, ValidReady(2)> -> !esi.channel<i16, ValidReady(2)>
  hw.output %0 : !esi.channel<i16, ValidReady(2)>
}

// CHECK-LABEL:  hw.module @bufferVODelay(
// CHECK-NEXT:     %rawOutput, %valid = esi.unwrap.vo %in : !esi.channel<i16, ValidOnly(1)>
// CHECK-NEXT:     %chanOutput, %ready = esi.wrap.vr %rawOutput, %valid : !esi.channel<i16, ValidReady(1)>
// CHECK-NEXT:     [[STAGE:%.+]] = esi.stage %clk, %rst, %chanOutput : !esi.channel<i16, ValidReady(1)>
// CHECK-NEXT:     hw.output [[STAGE]] : !esi.channel<i16, ValidReady(1)>
// HW-LABEL:     hw.module @bufferVODelay(
// HW-NEXT:        hw.instance "pipelineStage" @ESI_PipelineStage<WIDTH: ui32 = 16, DATA_DELAY: ui32 = 1>(
hw.module @bufferVODelay(in %clk: !seq.clock, in %rst: i1, in %in: !esi.channel<i16, ValidOnly(1)>, out out: !esi.channel<i16, ValidReady(1)>) {
  %0 = esi.buffer %clk, %rst, %in : !esi.channel<i16, ValidOnly(1)> -> !esi.channel<i16, ValidReady(1)>
  hw.output %0 : !esi.channel<i16, ValidReady(1)>
}

// CHECK-LABEL:  hw.module @bufferVOOutputDelay(
// CHECK-NEXT:     [[STAGE:%.+]] = esi.stage %clk, %rst, %in : !esi.channel<i16, ValidReady(1)>
// CHECK-NEXT:     %true = hw.constant true
// CHECK-NEXT:     %rawOutput, %valid = esi.unwrap.vr [[STAGE]], %true : !esi.channel<i16, ValidReady(1)>
// CHECK-NEXT:     [[OUT:%.+]] = esi.wrap.vo %rawOutput, %valid : !esi.channel<i16, ValidOnly(1)>
// CHECK-NEXT:     hw.output [[OUT]] : !esi.channel<i16, ValidOnly(1)>
// HW-LABEL:     hw.module @bufferVOOutputDelay(
// HW-NEXT:        hw.instance "pipelineStage" @ESI_PipelineStage<WIDTH: ui32 = 16, DATA_DELAY: ui32 = 1>(
hw.module @bufferVOOutputDelay(in %clk: !seq.clock, in %rst: i1, in %in: !esi.channel<i16, ValidReady(1)>, out out: !esi.channel<i16, ValidOnly(1)>) {
  %0 = esi.buffer %clk, %rst, %in : !esi.channel<i16, ValidReady(1)> -> !esi.channel<i16, ValidOnly(1)>
  hw.output %0 : !esi.channel<i16, ValidOnly(1)>
}

// Changing the data delay. A larger output delay delays the data with
// registers.

// CHECK-LABEL:  hw.module @bufferIncreaseDelay(
// CHECK:          [[STAGE:%.+]] = esi.stage %clk, %rst, %chanOutput {name = "buf_stage0"} : !esi.channel<i16, ValidReady(1)>
// CHECK-NEXT:     %rawOutput, %valid = esi.unwrap.vr [[STAGE]], [[READY:%.+]] : !esi.channel<i16, ValidReady(1)>
// CHECK-NEXT:     %buf_data_dly0 = seq.compreg %rawOutput, %clk : i16
// CHECK-NEXT:     %buf_data_dly1 = seq.compreg %buf_data_dly0, %clk : i16
// CHECK-NEXT:     %chanOutput_0, [[READY]] = esi.wrap.vr %buf_data_dly1, %valid : !esi.channel<i16, ValidReady(3)>
// CHECK-NEXT:     hw.output %chanOutput_0 : !esi.channel<i16, ValidReady(3)>
hw.module @bufferIncreaseDelay(in %clk: !seq.clock, in %rst: i1, in %in: !esi.channel<i16, FIFO(1)>, out out: !esi.channel<i16, ValidReady(3)>) {
  %0 = esi.buffer %clk, %rst, %in {name = "buf"} : !esi.channel<i16, FIFO(1)> -> !esi.channel<i16, ValidReady(3)>
  hw.output %0 : !esi.channel<i16, ValidReady(3)>
}

// CHECK-LABEL:  hw.module @bufferIncreaseDelayFIFO(
// CHECK-NEXT:     [[STAGE:%.+]] = esi.stage %clk, %rst, %in : !esi.channel<i16, ValidReady>
// CHECK-NEXT:     %rawOutput, %valid = esi.unwrap.vr [[STAGE]], %ready : !esi.channel<i16, ValidReady>
// CHECK-NEXT:     %data_dly0 = seq.compreg %rawOutput, %clk : i16
// CHECK-NEXT:     %chanOutput, %ready = esi.wrap.vr %data_dly0, %valid : !esi.channel<i16, ValidReady(1)>
// CHECK-NEXT:     %rawOutput_0, %valid_1 = esi.unwrap.vr %chanOutput, %rden : !esi.channel<i16, ValidReady(1)>
// CHECK-NEXT:     %chanOutput_2, %rden = esi.wrap.fifo %rawOutput_0, {{%.+}} : !esi.channel<i16, FIFO(1)>
// CHECK:          hw.output %chanOutput_2 : !esi.channel<i16, FIFO(1)>
hw.module @bufferIncreaseDelayFIFO(in %clk: !seq.clock, in %rst: i1, in %in: !esi.channel<i16, ValidReady>, out out: !esi.channel<i16, FIFO(1)>) {
  %0 = esi.buffer %clk, %rst, %in : !esi.channel<i16, ValidReady> -> !esi.channel<i16, FIFO(1)>
  hw.output %0 : !esi.channel<i16, FIFO(1)>
}

// A smaller ValidOnly output delay delays the valid.

// CHECK-LABEL:  hw.module @bufferDecreaseDelayVO(
// CHECK:          [[STAGE:%.+]] = esi.stage %clk, %rst, %chanOutput {name = "buf_stage0"} : !esi.channel<i16, ValidReady(2)>
// CHECK-NEXT:     [[TRUE:%.+]] = hw.constant true
// CHECK-NEXT:     %rawOutput, %valid = esi.unwrap.vr [[STAGE]], [[TRUE]] : !esi.channel<i16, ValidReady(2)>
// CHECK-NEXT:     %false = hw.constant false
// CHECK-NEXT:     %buf_valid_dly0 = seq.compreg %valid, %clk reset %rst, %false : i1
// CHECK-NEXT:     %buf_valid_dly1 = seq.compreg %buf_valid_dly0, %clk reset %rst, %false : i1
// CHECK-NEXT:     [[OUT:%.+]] = esi.wrap.vo %rawOutput, %buf_valid_dly1 : !esi.channel<i16, ValidOnly>
// CHECK-NEXT:     hw.output [[OUT]] : !esi.channel<i16, ValidOnly>
hw.module @bufferDecreaseDelayVO(in %clk: !seq.clock, in %rst: i1, in %in: !esi.channel<i16, FIFO(2)>, out out: !esi.channel<i16, ValidOnly>) {
  %0 = esi.buffer %clk, %rst, %in {name = "buf"} : !esi.channel<i16, FIFO(2)> -> !esi.channel<i16, ValidOnly>
  hw.output %0 : !esi.channel<i16, ValidOnly>
}

// A smaller ValidReady or FIFO output delay holds the tokens in a FIFO, which
// accepts input data with a delay.

// CHECK-LABEL:  hw.module @bufferDecreaseDelayVR(
// CHECK:          [[STAGE:%.+]] = esi.stage %clk, %rst, %chanOutput : !esi.channel<i16, ValidReady(2)>
// CHECK:          %rawOutput, %valid = esi.unwrap.vr [[STAGE]], [[INPUT_EN:%.+]] : !esi.channel<i16, ValidReady(2)>
// CHECK:          %inputEn_dly0 = seq.compreg [[INPUT_EN]], %clk reset %rst, %false : i1
// CHECK-NEXT:     %inputEn_dly1 = seq.compreg %inputEn_dly0, %clk reset %rst, %false : i1
// CHECK-NEXT:     %out, %full, %empty_2, %almostFull = seq.fifo depth 4 rd_latency 1 almost_full 2 in %rawOutput rdEn {{%.+}} wrEn %inputEn_dly1 clk %clk rst %rst : i16
// CHECK:          [[NOT_FULL:%.+]] = comb.xor %almostFull, %true_0 {sv.namehint = "seqFifoNotFull"} : i1
// CHECK-NEXT:     [[INPUT_EN]] = comb.and {{%.+}}, [[NOT_FULL]] {sv.namehint = "inputEn"} : i1
// CHECK:          %chanOutput_4, %ready_5 = esi.wrap.vr %out, {{%.+}} : !esi.channel<i16, ValidReady(1)>
// CHECK:          hw.output %chanOutput_4 : !esi.channel<i16, ValidReady(1)>
hw.module @bufferDecreaseDelayVR(in %clk: !seq.clock, in %rst: i1, in %in: !esi.channel<i16, FIFO(2)>, out out: !esi.channel<i16, ValidReady(1)>) {
  %0 = esi.buffer %clk, %rst, %in : !esi.channel<i16, FIFO(2)> -> !esi.channel<i16, ValidReady(1)>
  hw.output %0 : !esi.channel<i16, ValidReady(1)>
}

// CHECK-LABEL:  hw.module @bufferDecreaseDelayFIFO(
// CHECK:          %inputEn_dly0 = seq.compreg
// CHECK-NEXT:     %out, %full, %empty, %almostFull = seq.fifo depth 3 almost_full 2 in %rawOutput rdEn %rden wrEn %inputEn_dly0 clk %clk rst %rst : i16
// CHECK:          %chanOutput, %rden = esi.wrap.fifo %out, %empty : !esi.channel<i16, FIFO>
// CHECK-NEXT:     hw.output %chanOutput : !esi.channel<i16, FIFO>
hw.module @bufferDecreaseDelayFIFO(in %clk: !seq.clock, in %rst: i1, in %in: !esi.channel<i16, ValidReady(1)>, out out: !esi.channel<i16, FIFO>) {
  %0 = esi.buffer %clk, %rst, %in : !esi.channel<i16, ValidReady(1)> -> !esi.channel<i16, FIFO>
  hw.output %0 : !esi.channel<i16, FIFO>
}
