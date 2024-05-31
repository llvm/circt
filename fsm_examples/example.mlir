fsm.machine @foo(%arg0: i1) -> i1 attributes {stateType = i1} {
  fsm.state @IDLE output  {
    %true = constant true
    fsm.output %true : i1
  } transitions  {
    fsm.transition @BUSY ...
  }

  fsm.state @BUSY output  {
    %false = constant false
    fsm.output %false : i1
  } transitions  {
    fsm.transition @BUSY ...
    fsm.transition @IDLE ...
  }
}
