#
# Date created: 2022-03-23
# Author: aradclif
#
#
############################################################################################
A = rand(5,5,5,5);
@timev vvmapreduce(identity, min, typemax, A, (1,2,3));
@timev vvmapreduce(identity, min, typemax, A, (1,3,2));
@timev vvmapreduce(identity, min, typemax, A, (3,1,2));
