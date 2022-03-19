# Módulo 1: Topic modeling y LDA
El modelado de temas (topic modeling) es una técnica de machine learning que se usa para descubrir qué temas o tópicos ocurren en una colección de documentos.

Por ejemplo, si tuvieramos todas las noticias de ABC del 2020 en Australia, podríamos descubrir qué temas fueron los más hablados: el surgimiento del Covid, restricciones de viaje, las elecciones en EU, etc.

El modelo más popular para hacer topic modeling es el de **latent Dirichlet allocation (LDA)**, el cual es un modelo estadístico *generativo* y *no supervisado*. Para entender estos dos últimos conceptos, se implementa un **modelo generativo de lenguaje** y el modelo de **k-means clustering**, respectivamente.

(Cada modelo se implementa por separado).
![image](https://github.com/pedro9olivares/ML_and_AI_for_the_Working_Analyst/blob/fe2a60db6968dfb8e9869eed4df2c752f0e717c3/1_Topic_modeling_y_LDA/Esquema_para_LDA.png)

## Índice
* [Modelos generativos de lenguaje](#modelos-generativos-de-lenguaje)
* [K-means clustering](#k-means-clustering)
* [Latent Dirichlet allocation](#latent-dirichlet-allocation)

## Modelos generativos de lenguaje
Un modelo generativo de lenguaje es un modelo estadístico que se encarga de generar texto a partir de un vocabulario dado. En nuestro caso, el vocabulario estará dado por todas las palabras contenidas en el primer libro de Harry Potter. 

Se puede utilizar cualquier archivo .txt como vocabulario, si así se desea. Para hacerlo, solamente hace falta:
* subir su .txt a un repositorio de GitHub,
* obtener la liga raw y
* cambiar la siguiente línea por dicha liga raw:
![image](https://github.com/pedro9olivares/ML_and_AI_for_the_Working_Analyst/blob/main/1_Topic_modeling_y_LDA/imagenAux1.png)


## K-means clustering

## Latent Dirichlet allocation
