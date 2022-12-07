# Assorted tests; TODO: systematic
xx = rand(ntuple(_ -> 3, 4)...);
@timev R = vmapreducethen(abs2, +, √, xx, dims=(2,4))
@timev R2 = .√mapreduce(abs2, +, xx, dims=(2,4))
R ≈ R2

XX = rand(ntuple(_ -> 5, 4)...);
@benchmark vmapreducethen(abs2, +, √, $XX, dims=(2,4))
@benchmark .√mapreduce(abs2, +, XX, dims=(2,4))

@benchmark vnorm3($x1, 2.0)
@benchmark vnorm($x1, 1.5)
@benchmark vnorm3($x1, 1.5)


@benchmark vnorm($x1, 2.0)
@benchmark norm($x1, 1.5)
@benchmark vnorm($x1, 1)
@benchmark vnorm2($x1, 1.5)

vmean(x1, dims=(2,4))
@benchmark vmean($lpds, dims=(1,2))
@benchmark mean($lpds, dims=(1,2))

@benchmark veuclidean($x1, $x2)
@benchmark norm($x1 .- $x2)
@benchmark euclidean($x1, $x2)
@test euclidean(x1, x2, dims=(2,4)) ≈ veuclidean(x1, x2, dims=(2,4))
@benchmark veuclidean($x1, $x2, dims=(2,4))
@benchmark euclidean($x1, $x2, dims=(2,4))

YY = rand(1:100, 3,3,3,3);
xxx = rand(3,3,3,3);

@benchmark vmapreducethen(abs2, +, abs2, $YY, dims=(2,4))
@benchmark vmapreducethen(abs2, +, abs2, $xxx, dims=(2,4))

abs2.(mapreduce(abs2, +, YY, dims=(2,4))) == vmapreducethen(abs2, +, abs2, YY, dims=(2,4))

@code_warntype _vmapreducethen!(abs2, +, abs2, zero, R, xxx, (static(2), static(4)))

x1 = rand(5,5,5,5);
x2 = rand(5,5,5,5);
x3 = rand(5,5,5,5);
@benchmark vmapreducethen(+, +, abs2, $x1, $x2, $x3)
@benchmark abs2(mapreduce(+, +, $x1, $x2, $x3))
@test abs2(mapreduce(+, +, x1, x2, x3)) ≈ vmapreducethen(+, +, abs2, x1, x2, x3)

@benchmark vmapreducethen(+, +, abs2, $x1, $x2, $x3, dims=(2,4))
@benchmark abs2.(mapreduce(+, +, $x1, $x2, $x3, dims=(2,4)))
