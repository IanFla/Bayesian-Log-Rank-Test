data(diabetic, package="survival")
X1 <- diabetic$time[diabetic$trt == 0]
X2 <- diabetic$time[diabetic$trt == 1]
D1 <- diabetic$status[diabetic$trt == 0]
D2 <- diabetic$status[diabetic$trt == 1]
DRS0 <- data.frame(X1, D1)
DRS1 <- data.frame(X2, D2)
write.csv(DRS0, "../data/DRS0.csv")
write.csv(DRS1, "../data/DRS1.csv")

data(cgd, package="survival")
C <- cgd0$futime
T <- cgd0$etime1
T[is.na(T)] <- Inf
D <- as.numeric(T <= C)
X <- apply(cbind(T, C), 1, min)
G <- cgd0$treat
X1<-X[G == 0]
X2<-X[G == 1]
D1<-D[G == 0]
D2<-D[G == 1]
CGD0 <- data.frame(X1, D1)
CGD1 <- data.frame(X2, D2)
write.csv(CGD0, "../data/CGD0.csv")
write.csv(CGD1, "../data/CGD1.csv")
