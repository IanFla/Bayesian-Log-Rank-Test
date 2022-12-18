cal.norm <- function() {
    pnorm(1) - pnorm(-1)
}

cal.t <- function(df) {
    s <- sqrt(df / (df - 2))
    
    pt(s, df) - pt(-s, df)
}

cal.gamma <- function(a) {
    m <- a
    s <- sqrt(a)
    
    pgamma(m + s, a, 1) - pgamma(m - s, a, 1)
}

cal.beta <- function(a, b) {
    m <- a / (a + b)
    s <- sqrt(a * b / (a + b + 1)) / (a + b)
    
    x <- seq(0, 1, length=1000)
    plot(x, dbeta(x, a, b))
    
    pbeta(m + s, a, b) - pbeta(m - s, a, b)
}
