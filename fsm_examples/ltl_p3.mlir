%error = unrealized_conversion_cast to !ltl.sequence
%state = unrealized_conversion_cast to !ltl.sequence
%e0 = ltl.implication %error, %state {state = "_1", signal q= "0", input = "1", error = "ERR"}: !ltl.sequence, !ltl.sequence