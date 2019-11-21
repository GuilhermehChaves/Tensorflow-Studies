dados <- read.csv('C:/Users/DeepLearning/Desktop/tensorflow/git/003_nota_enem/nota-enem.csv', header = TRUE, sep = ',')
cor(dados[,2:4])
dados
modelo <- lm(data = dados, ï..nota ~ horas_dia + qtd_dias + media_escola)
step(modelo, direction = 'both', scale = 2.446^2)
modelo2 <- lm(formula = ï..nota ~ horas_dia + qtd_dias + media_escola, data = dados)
summary(modelo2)

par(mfrow = c(2,2))
plot(modelo, which = c(1:4), pch = 20)
plot(modelo2)

modelo2$coefficients
