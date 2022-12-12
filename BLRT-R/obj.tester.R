source("obj.sampler.R")


Tester <- R6::R6Class("Tester", list(
    # the Bayesian log-rank test
    Dat = NA, 
    Res = NA, 
    
    initialize = function(L1, R1, L2, R2) {
        # initialization
        self$Dat <- list(L1=L1, R1=R1, L2=L2, R2=R2)
        self$Res <- list()
    }, 
    
    classic = function() {
        # perform classic log-rank test for right-censored data
        X <- c(self$Dat$L1, self$Dat$L2)
        D <- c(self$Dat$L1 == self$Dat$R1, self$Dat$L2 == self$Dat$R2)
        G <- c(rep(1, length(self$Dat$L1)), rep(2, length(self$Dat$L2)))
        fit <- survival::survdiff(survival::Surv(X, D) ~ G, rho=0)
        self$Res$pv.freq <- fit$pvalue
    }, 
    
    divide = function(num, den) {
        # num / den
        den[num == 0] <- 1
        
        num / den
    }, 
    
    bayesian = function(size=1000, M=0.001, a=0.001, b=0.001, Len=500) {
        # perform Bayesian log-rank test
        sample1 <- Sampler$new(self$Dat$L1, self$Dat$R1, size, M, a, b)
        sample2 <- Sampler$new(self$Dat$L2, self$Dat$R2, size, M, a, b)
        tau <- min(max(self$Dat$L1), max(self$Dat$L2))
        grid <- seq(0, tau, length=Len)
        sample1$gibbs()
        sample2$gibbs()
        sample1$dirichlet(grid)
        sample2$dirichlet(grid)
        
        R1 <- rowSums(outer(grid[-length(grid)], self$Dat$L1, "<="))
        R2 <- rowSums(outer(grid[-length(grid)], self$Dat$L2, "<="))
        RR <- R1 * R2 / (R1 + R2)
        H1 <- self$divide(sample1$Sur[-1, ] - sample1$Sur[-length(grid), ], 
                          sample1$Sur[-length(grid), ])
        H2 <- self$divide(sample2$Sur[-1, ] - sample2$Sur[-length(grid), ], 
                          sample2$Sur[-length(grid), ])
        Q1 <- colSums(RR * H1)
        Q2 <- colSums(RR * H2)
        self$Res$Q <- c(outer(Q1, Q2, "-"))
        self$Res$pv.baye <- 2 * min(mean(self$Res$Q > 0), mean(self$Res$Q < 0))
    }
))


if (1 == 1) {
    # codes for testing
    library(ggplot2)
    
    generate <- function(size=100) {
        L1 <- rexp(size)
        C1 <- rexp(size)
        C1[C1 > 1.0] <- 1.0
        D1 <- L1 <= C1
        L1 <- D1 * L1 + (1 - D1) * C1
        R1 <- L1
        R1[!D1] <- Inf
        
        L2 <- rexp(size)
        C2 <- rexp(size)
        C2[C2 > 1.0] <- 1.0
        D2 <- L2 <= C2
        L2 <- D2 * L2 + (1 - D2) * C2
        R2 <- L2
        R2[!D2] <- Inf
        
        list(L1=L1, R1=R1, L2=L2, R2=R2)
    }
    
    main <- function() {
        RES <- list(pv.freq=c(), pv.baye=c())
        for (i in 1:100) {
            print(i)
            Dat <- generate()
            test <- Tester$new(Dat$L1, Dat$R1, Dat$L2, Dat$R2)
            test$classic()
            test$bayesian()
            RES$pv.freq <- c(RES$pv.freq, test$Res$pv.freq)
            RES$pv.baye <- c(RES$pv.baye, test$Res$pv.baye)
        }
        
        df <- data.frame(alpha=seq(0, 1, length=100))
        df$pvalue <- rowMeans(outer(df$alpha, RES$pv.freq, ">="))
        df$pv2 <- rowMeans(outer(df$alpha, RES$pv.baye, ">="))
        gr <- ggplot(df) + geom_line(aes(x=alpha, y=pvalue), color=1) + 
            geom_line(aes(x=alpha, y=pv2), color=2)
        
        print(gr)
    }
    
    main()
}

