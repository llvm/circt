; 
(set-info :status unknown)
(declare-fun var0_0 (Int) Int)
(declare-fun time-to-state (Int) Int)
(assert
 (let (($x10 (= (var0_0 0) 0)))
 (let (($x8 (= (time-to-state 0) 0)))
 (and $x8 $x10))))
(assert
 (forall ((time Int) )(let (($x47 (= (var0_0 (+ time 1)) (+ (var0_0 time) 1))))
 (let ((?x51 (time-to-state time)))
 (let (($x52 (= ?x51 0)))
 (=> $x52 (and (= (time-to-state (+ time 1)) 1) $x47))))))
 )
(assert
 (forall ((time Int) )(let (($x47 (= (var0_0 (+ time 1)) (+ (var0_0 time) 1))))
 (let ((?x51 (time-to-state time)))
 (let (($x70 (= ?x51 1)))
 (=> $x70 (and (= (time-to-state (+ time 1)) 2) $x47))))))
 )
(assert
 (forall ((time Int) )(let (($x47 (= (var0_0 (+ time 1)) (+ (var0_0 time) 1))))
 (let ((?x51 (time-to-state time)))
 (let (($x81 (= ?x51 2)))
 (=> $x81 (and (= (time-to-state (+ time 1)) 3) $x47))))))
 )
(assert
 (forall ((time Int) )(let (($x47 (= (var0_0 (+ time 1)) (+ (var0_0 time) 1))))
 (let ((?x51 (time-to-state time)))
 (let (($x92 (= ?x51 3)))
 (=> $x92 (and (= (time-to-state (+ time 1)) 4) $x47))))))
 )
(assert
 (forall ((time Int) )(let (($x47 (= (var0_0 (+ time 1)) (+ (var0_0 time) 1))))
 (let ((?x51 (time-to-state time)))
 (let (($x103 (= ?x51 4)))
 (=> $x103 (and (= (time-to-state (+ time 1)) 5) $x47))))))
 )
(assert
 (forall ((time Int) )(let ((?x51 (time-to-state time)))
 (let (($x107 (= ?x51 6)))
 (and $x107 (and (distinct (var0_0 time) 1) true)))))
 )
(check-sat)
