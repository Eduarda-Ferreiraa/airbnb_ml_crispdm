install.packages("ggplot2")
install.packages("dplyr")

library(ggplot2)
library(dplyr)

data <- read.csv("/Users/eduardaferreira/Desktop/AB_NYC_2019.csv")  
##1
ggplot(data, aes(x = price)) +
  geom_histogram(binwidth = 10, fill = "skyblue", color = "black", boundary = 0) +
  xlim(0, 500) +
  labs(title = "Distribuição de Preços por Noite (até $500)",
       x = "Preço por Noite (USD)",
       y = "Frequência") +
  theme_minimal()
##2
ggplot(data, aes(x = room_type, fill = room_type)) +
  geom_bar() +
  scale_fill_viridis_d() +
  labs(title = "Distribuição por Tipo de Quarto",
       x = "Tipo de Quarto",
       y = "Contagem") +
  theme_minimal() +
  theme(legend.position = "none") 
##3
ggplot(data, aes(x = availability_365)) +
  geom_histogram(binwidth = 10, fill = "salmon", color = "black", boundary = 0) +
  labs(title = "Distribuição da Disponibilidade Anual (availability_365)",
       x = "Dias Disponíveis no Ano",
       y = "Frequência") +
  theme_minimal()

##4
ggplot(data, aes(x = neighbourhood_group, fill = neighbourhood_group)) +
  geom_bar() +
  scale_fill_brewer(palette = "Set2") +
  labs(title = "Distribuição por Distrito",
       x = "Distrito",
       y = "Frequência") +
  theme_minimal() +
  theme(legend.position = "none")  
