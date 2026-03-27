Predição de Mortalidade Neonatal com Machine Learning (PyCaret)
Descrição

Este repositório contém os códigos e a configuração experimental utilizados no estudo sobre predição de mortalidade neonatal por meio de técnicas de aprendizado de máquina.

O foco principal é a reprodutibilidade dos experimentos, utilizando a biblioteca PyCaret para padronização e comparação de modelos.

Objetivo

Reproduzir e avaliar modelos de machine learning aplicados à predição de mortalidade neonatal, com base em dados públicos de saúde.

Metodologia
Uso de PyCaret para automação de modelos (AutoML)
Problema de classificação binária (óbito neonatal vs. sobrevivência)
Pré-processamento realizado via setup()
Comparação entre múltiplos algoritmos
Dados

Os dados utilizados são provenientes de bases públicas de saúde.


*Reprodutibilidade*

Para executar o projeto:

pip install -r requirements.txt
jupyter notebook

Em seguida, execute o notebook principal disponível no repositório.

Configuração Experimental
Uso do setup() do PyCaret
Definição de divisão treino/teste via test_size
Uso de session_id para garantir reprodutibilidade

Resultados

Os modelos são avaliados com métricas padrão de classificação, como acurácia, AUC, recall, precisão e F1-score.

Disponibilidade do Código

O código utilizado no estudo está disponível neste repositório para fins de reprodutibilidade científica.

 Autor

Metusalen Rocha
