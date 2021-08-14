
hir.func @callee at %t(%x:i32)->(i32 delay 1){
    %res = hir.delay %x by 1  at %t : i32
    hir.return (%res):(i32)
}

hir.func @caller at %t(%x:i32) ->(i32 delay 1){
    %res = hir.call @callee(%x) at %t + 1 : !hir.func<(i32) -> (i32 delay 1)>
    hir.return (%res):(i32)
}
