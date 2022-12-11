Sampler <- R6::R6Class("Sampler", list(
    # the Gibbs sampler of the MDP posterior
    Dat = NA, 
    Par = NA, 
    The = NA, 
    Tim = NA, 
    Sur = NA, 
    
    initialize = function(L, R, size=1000, M=0.001, a=0.001, b=0.001) {
        # initialization
        self$Dat <- list(L=L, R=R, ind=which(L < R))
        self$Par <- list(size=size, M=M, a=a, b=b)
    }, 
    
    G0_cdf = function(x) {
        # modifiable
        pweibull(x, shape=1)
    }, 
    
    G0_ppf = function(p) {
        # modifiable
        qweibull(p, shape=1)
    }, 
    
    G_cdf = function(x, the) {
        # the cdf of the base distribution
        1 - ((1 - self$G0_cdf(x))^the)
    }, 
    
    G_ppf = function(p, theta) {
        # the ppf of the base distribution
        self$G0_ppf(1 - ((1 - p)^(1 / the)))
    }, 
    
    tim2the = function(tim) {
        # sample theta given time
        tim <- unique(tim)
        a <- self$Par$a + length(tim)
        b <- self$Par$b - sum(log(1 - self$G0_cdf(tim)))
        
        rgamma(1, shape=a, rate=b)
    }, 
    
    the2tim = function(the, tim) {
        # sample time given theta
        for (i in sample(self$Dat$ind)) {
            flag <- (self$Dat$L[i] <= tim) & (tim <= self$Dat$R[i])
            flag[i] <- F
            pL <- self$G_cdf(self$Dat$L[i], the)
            pR <- self$G_cdf(self$Dat$R[i], the)
            prob <- pR - pL
            prob <- c(self$Par$M * prob, 1 * flag)
            prob <- prob / sum(prob)
            choice <- sample(length(prob), 1, prob=prob)
            if (choice == 1) {
                tim[i] <- self$G_ppf(pL + runif(1) * (pR - pL), the)
            } else {
                tim[i] <- tim[choice - 1]
            }
        }
        
        tim
    }, 
    
    imputation = function(thin=1, burn=100, the=1.0) {
        # impute the missing data
        pL <- self$G_cdf(self$Dat$L, the)
        pR <- self$G_cdf(self$Dat$R, the)
        tim <- self$G_ppf(pL + runif(length(pL)) * (pR - pL), the)
        for (b in 1:burn) {
            the <- self$tim2the(tim)
            tim <- self$the2tim(the, tim)
        }
        
        self$The <- rep(0.0, self$Par$size)
        self$Tim <- matrix(0.0, length(tim), self$Par$size)
        for (s in 1:self$Par$size) {
            for (t in 1:thin) {
                the <- self$tim2the(tim)
                tim <- self$the2tim(the, tim)
            }
            
            self$The[s] <- the
            self$Tim[, s] <- tim
        }
    }, 
    
    sampling = function() {
        # draw samples from the Dirichlet process
        
    }
))


if (1 == 0) {
    # codes for testing
    sampler <- Sampler$new()
}

