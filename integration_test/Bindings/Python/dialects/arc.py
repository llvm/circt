# REQUIRES: bindings_python
# RUN: %PYTHON% %s | FileCheck %s

import sys

import circt
from circt.dialects import arc, hw
from circt.ir import Context, Location, FlatSymbolRefAttr, IntegerType

with Context() as ctx, Location.unknown():
  circt.register_dialects(ctx)

  i1 = IntegerType.get_signless(1)
  i8 = IntegerType.get_signless(8)
  i32 = IntegerType.get_signless(32)

  # Test StateType
  # CHECK: !arc.state<i32>
  state_type = arc.StateType.get(i32)
  print(state_type)

  # CHECK: bit_width: 32
  # CHECK: byte_width: 4
  print(f"bit_width: {state_type.bit_width}")
  print(f"byte_width: {state_type.byte_width}")

  # CHECK: type: i32
  print(f"type: {state_type.type}")

  # Test MemoryType
  # CHECK: !arc.memory<16 x i8, i8>
  memory_type = arc.MemoryType.get(16, i8, i8)
  print(memory_type)

  # CHECK: num_words: 16
  # CHECK: word_type: i8
  # CHECK: address_type: i8
  # CHECK: stride: 1
  print(f"num_words: {memory_type.num_words}")
  print(f"word_type: {memory_type.word_type}")
  print(f"address_type: {memory_type.address_type}")
  print(f"stride: {memory_type.stride}")

  # Test StorageType without size
  # CHECK: !arc.storage
  storage_type_no_size = arc.StorageType.get(ctx)
  print(storage_type_no_size)

  # CHECK: size: 0
  print(f"size: {storage_type_no_size.size}")

  # Test StorageType with size
  # CHECK: !arc.storage<256>
  storage_type_with_size = arc.StorageType.get(ctx, 256)
  print(storage_type_with_size)

  # CHECK: size: 256
  print(f"size: {storage_type_with_size.size}")

  # Test SimModelInstanceType
  # CHECK: !arc.sim.instance<@model_name>
  sim_model_attr = FlatSymbolRefAttr.get("model_name")
  sim_model_type = arc.SimModelInstanceType.get(sim_model_attr)
  print(sim_model_type)

  # CHECK: model: @model_name
  print(f"model: {sim_model_type.model}")
