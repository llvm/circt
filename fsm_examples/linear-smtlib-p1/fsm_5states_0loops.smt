; 
(set-info :status unknown)
(declare-fun time-to-state (Int) Int)
(declare-fun var0_0 (Int) Int)
(assert
 (forall ((time Int) )(let (($x48 (<= time 0)))
 (let (($x56 (= (var0_0 (+ time 1)) (+ (var0_0 time) 1))))
 (let (($x10 (= (var0_0 0) 0)))
 (let (($x68 (and (and (and (= (time-to-state 0) 0) $x10) $x10) (and (= (time-to-state (+ time 1)) 1) $x56))))
 (let (($x70 (and (and $x68 (and (= (time-to-state (+ time 1)) 2) $x56)) (and (= (time-to-state (+ time 1)) 3) $x56))))
 (let (($x72 (and (and $x70 (and (= (time-to-state (+ time 1)) 4) $x56)) (and (= (time-to-state (+ time 1)) 5) $x56))))
 (and (and (and $x72 (and (distinct (time-to-state time) 4) true)) (>= time 0)) $x48))))))))
 )
(check-sat)
