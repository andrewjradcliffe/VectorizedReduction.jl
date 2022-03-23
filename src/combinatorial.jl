#
# Date created: 2022-03-23
# Author: aradclif
#
#
############################################################################################
A = rand(5,5,5,5);
@timev vvmapreduce(identity, min, typemin, A, (1,2,3));
@timev vvmapreduce(identity, min, typemin, A, (1,3,2));
@timev vvmapreduce(identity, min, typemin, A, (2,1,3));
@timev vvmapreduce(identity, min, typemin, A, (2,3,1));
@timev vvmapreduce(identity, min, typemin, A, (3,1,2));
@timev vvmapreduce(identity, min, typemin, A, (3,2,1));
#
@timev vvmapreduce(identity, min, typemin, A, (1,2,4));
@timev vvmapreduce(identity, min, typemin, A, (1,4,2));
@timev vvmapreduce(identity, min, typemin, A, (2,1,4));
@timev vvmapreduce(identity, min, typemin, A, (2,4,1));
@timev vvmapreduce(identity, min, typemin, A, (4,1,2));
@timev vvmapreduce(identity, min, typemin, A, (4,2,1));
#
@timev vvmapreduce(identity, min, typemin, A, (1,4,3));
@timev vvmapreduce(identity, min, typemin, A, (1,3,4));
@timev vvmapreduce(identity, min, typemin, A, (4,1,3));
@timev vvmapreduce(identity, min, typemin, A, (4,3,1));
@timev vvmapreduce(identity, min, typemin, A, (3,1,4));
@timev vvmapreduce(identity, min, typemin, A, (3,4,1));
#
@timev vvmapreduce(identity, min, typemin, A, (4,2,3));
@timev vvmapreduce(identity, min, typemin, A, (4,3,2));
@timev vvmapreduce(identity, min, typemin, A, (2,4,3));
@timev vvmapreduce(identity, min, typemin, A, (2,3,4));
@timev vvmapreduce(identity, min, typemin, A, (3,4,2));
@timev vvmapreduce(identity, min, typemin, A, (3,2,4));
#
@timev minimum(A, dims=(1,2,3));
@timev minimum(A, dims=(1,3,2));
@timev minimum(A, dims=(2,1,3));
@timev minimum(A, dims=(2,3,1));
@timev minimum(A, dims=(3,1,2));
@timev minimum(A, dims=(3,2,1));

vvmapreduce(identity, min, typemin, A, (1,2,3)) == vvmapreduce(identity, min, typemin, A, (1,3,2))

@timev VectorizedStatistics._vsum(A, (1,2,3));
@timev VectorizedStatistics._vsum(A, (1,3,2));
@timev VectorizedStatistics._vsum(A, (2,1,3));
@timev VectorizedStatistics._vsum(A, (2,3,1));
@timev VectorizedStatistics._vsum(A, (3,1,2));
@timev VectorizedStatistics._vsum(A, (3,2,1));

@timev vvmapreduce3(identity, min, typemin, A, (1,2,3));
@timev vvmapreduce3(identity, min, typemin, A, (1,3,2));
@timev vvmapreduce3(identity, min, typemin, A, (2,1,3));
@timev vvmapreduce3(identity, min, typemin, A, (2,3,1));
@timev vvmapreduce3(identity, min, typemin, A, (3,1,2));
@timev vvmapreduce3(identity, min, typemin, A, (3,2,1));

@timev vvmapreduce(identity, min, typemin, A, (1,2));
@timev vvmapreduce(identity, min, typemin, A, (1,3));
@timev vvmapreduce(identity, min, typemin, A, (1,4));
@timev vvmapreduce(identity, min, typemin, A, (2,1));
@timev vvmapreduce(identity, min, typemin, A, (2,3));
@timev vvmapreduce(identity, min, typemin, A, (2,4));
@timev vvmapreduce(identity, min, typemin, A, (3,1));
@timev vvmapreduce(identity, min, typemin, A, (3,2));
@timev vvmapreduce(identity, min, typemin, A, (3,4));
@timev vvmapreduce(identity, min, typemin, A, (4,1));
@timev vvmapreduce(identity, min, typemin, A, (4,2));
@timev vvmapreduce(identity, min, typemin, A, (4,3));

