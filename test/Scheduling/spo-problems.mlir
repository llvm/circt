// RUN: circt-opt %s -test-spo-problem

func @full_load(%a0 : i32, %a1 : i32, %a2 : i32, %a3 : i32, %a4 : i32, %a5 : i32) -> i32 attributes {
  operatortypes = [
    { name = "add", latency = 3, limit = 1}
  ] } {
  %0 = addi %a0, %a1 { opr = "add", problemStartTime = 0 } : i32
  %1 = addi %a1, %a1 { opr = "add", problemStartTime = 1 } : i32
  %2 = addi %a2, %a3 { opr = "add", problemStartTime = 2 } : i32
  %3 = addi %a3, %a4 { opr = "add", problemStartTime = 3 } : i32
  %4 = addi %a4, %a5 { opr = "add", problemStartTime = 4 } : i32
  return { problemStartTime = 7 } %4 : i32
}

func @partial_load(%a0 : i32, %a1 : i32, %a2 : i32, %a3 : i32, %a4 : i32, %a5 : i32) -> i32 attributes {
  operatortypes = [
    { name = "add", latency = 3, limit = 3}
  ] } {
  %0 = addi %a0, %a1 { opr = "add", problemStartTime = 0 } : i32
  %1 = addi %a1, %a1 { opr = "add", problemStartTime = 1 } : i32
  %2 = addi %a2, %a3 { opr = "add", problemStartTime = 0 } : i32
  %3 = addi %a3, %a4 { opr = "add", problemStartTime = 2 } : i32
  %4 = addi %a4, %a5 { opr = "add", problemStartTime = 1 } : i32
  return { problemStartTime = 10 } %4 : i32
}

func @multiple(%a0 : i32, %a1 : i32, %a2 : i32, %a3 : i32, %a4 : i32, %a5 : i32) -> i32 attributes {
  operatortypes = [
    { name = "slowAdd", latency = 3, limit = 2},
    { name = "fastAdd", latency = 1, limit = 1}
  ] } {
  %0 = addi %a0, %a1 { opr = "slowAdd", problemStartTime = 0 } : i32
  %1 = addi %a1, %a1 { opr = "slowAdd", problemStartTime = 1 } : i32
  %2 = addi %a2, %a3 { opr = "fastAdd", problemStartTime = 0 } : i32
  %3 = addi %a3, %a4 { opr = "slowAdd", problemStartTime = 1 } : i32
  %4 = addi %a4, %a5 { opr = "fastAdd", problemStartTime = 1 } : i32
  return { problemStartTime = 10 } %4 : i32
}
