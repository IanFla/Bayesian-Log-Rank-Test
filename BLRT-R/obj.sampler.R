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
        pexp(x)
    }, 
    
    G0_ppf = function(p) {
        # modifiable
        qexp(p)
    }, 
    
    G_cdf = function(x, the) {
        # the cdf of the base distribution
        1 - ((1 - self$G0_cdf(x))^the)
    }, 
    
    G_ppf = function(p, the) {
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
    
    gibbs = function(thin=1, burn=100, the=1.0) {
        # impute the missing data by Gibbs
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
    
    dirichlet = function(grid) {
        # draw samples from the Dirichlet process
        add <- F
        if (grid[1] != 0.0) {
            grid <- c(0.0, grid)
            add <- T
        }
        self$Sur <- matrix(0.0, length(grid), self$Par$size)
        self$Sur[1, ] <- 1.0
        a <- (self$Par$M + nrow(self$Tim)) * rep(1, self$Par$size)
        for (j in 2:length(grid)) {
            c <- a
            a <- self$Par$M * (1 - self$G_cdf(grid[j], self$The)) + colSums(grid[j] < self$Tim)
            flag <- a > 0
            ratio <- rbeta(sum(flag), a[flag], c[flag] - a[flag])
            self$Sur[j, flag] <- ratio * self$Sur[j - 1, flag]
        }
        if (add) {
            self$Sur <- self$Sur[-1, ]
        }
    }
))


if (1 == 0) {
    # codes for testing
    library(ggplot2)
    library(survival)
    
    main <- function() {
        L <- rexp(100)
        C <- rexp(100)
        C[C > 1.0] <- 1.0
        D <- L <= C
        L <- D * L + (1 - D) * C
        R <- L
        R[!D] <- Inf
        fit <- survfit(Surv(L, D) ~ 1, conf.type="arcsin")
        df <- data.frame(time=c(0, fit$time, fit$time), 
                         surv=c(1, 1, fit$surv[-length(fit$surv)], fit$surv), 
                         upper=c(1, 1, fit$upper[-length(fit$upper)], fit$upper), 
                         lower=c(1, 1, fit$lower[-length(fit$lower)], fit$lower), 
                         grid=sort(c(0, fit$time - 0.0001, fit$time + 0.0001)))
        
        sample <- Sampler$new(L, R)
        print(system.time(sample$gibbs()))
        print(system.time(sample$dirichlet(df$grid)))
        df$mean <- rowMeans(sample$Sur)
        df$up <- apply(sample$Sur, 1, function(x) quantile(x, 0.975))
        df$low <- apply(sample$Sur, 1, function(x) quantile(x, 0.025))
        
        gr <- ggplot(df) + geom_line(aes(x=time, y=surv), color=1) + 
            geom_line(aes(x=time, y=upper), color=1, linetype=2) + 
            geom_line(aes(x=time, y=lower), color=1, linetype=2) + 
            geom_line(aes(x=grid, y=mean), color=2) + 
            geom_line(aes(x=grid, y=up), color=2, linetype=2) + 
            geom_line(aes(x=grid, y=low), color=2, linetype=2)
    }
    
    print(main())
}

