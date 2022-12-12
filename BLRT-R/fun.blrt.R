source("obj.tester.R")


blrt <- function(L1, R1, L2, R2, size=1000, M=0.001, a=0.001, b=0.001) {
    # Bayesian Log-Rank Test
    test <- Tester$new(L1, R1, L2, R2)
    test$bayesian(size, M, a, b, 3 * (length(L1) + length(L2)))
    
    list(pv=test$Res$pv.baye, Q=test$Res$Q)
}


if (1 == 0) {
    # codes for testing
    library(survival)
    library(ggplot2)
    library(latex2exp)
    
    main <- function() {
        data("cgd")
        G <- cgd0$treat
        cens <- cgd0$futime
        even <- cgd0$etime1
        even[is.na(even)] <- Inf
        D <- even <= cens
        L <- even
        L[!D] <- cens[!D]
        L <- L / max(L)
        R <- L
        R[!D] <- Inf
        L1 <- L[G == 0]
        R1 <- R[G == 0]
        L2 <- L[G == 1]
        R2 <- R[G == 1]
        
        set.seed(1997)
        res <- blrt(L1, R1, L2, R2)
        print(res$pv)
        gr <- ggplot() + geom_histogram(aes(x=res$Q), bins=100, fill=2) + 
            geom_vline(xintercept=0, linetype=2, color=4) + 
            xlab(TeX("$Q$")) + ylab("Relative Frequency")
        print(gr)
    }
    
    main()
}

