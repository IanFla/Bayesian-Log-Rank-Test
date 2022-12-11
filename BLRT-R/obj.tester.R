source("obj.sampler.R")


Tester <- R6::R6Class("Tester", list(
    # the Gibbs sampler of the MDP posterior
    Dat = NA, 
    
    initialize = function(L, R, size=1000, M=0.001, a=0.001, b=0.001) {
        # initialization
        self$Dat <- list(L=L, R=R, ind=which(L < R))
        self$Par <- list(size=size, M=M, a=a, b=b)
    }
))


if (1 == 1) {
    # codes for testing
    main <- function() {
        1
    }
    
    main()
}

