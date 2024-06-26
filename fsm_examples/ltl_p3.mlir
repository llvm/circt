%error = unrealized_conversion_cast to !ltl.sequence
%state = unrealized_conversion_cast to !ltl.sequence
%e0 = ltl.implication %error, %state {state = "ERR", signal= "0", input = "1"}: !ltl.sequence, !ltl.sequence