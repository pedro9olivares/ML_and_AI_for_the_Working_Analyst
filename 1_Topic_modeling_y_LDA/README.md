# Módulo 1: Topic modeling y LDA
El modelado de temas (topic modeling) es una técnica de machine learning que se usa para descubrir qué temas o tópicos ocurren en una colección de documentos.

Por ejemplo, si tuvieramos todas las noticias de ABC del 2020 en Australia, podríamos descubrir qué temas fueron los más hablados: el surgimiento del Covid, las restricciones de viaje, las elecciones en EU, etc.

El modelo más popular para hacer topic modeling es el de **latent Dirichlet allocation (LDA)**, el cual es un modelo estadístico *generativo* y *no supervisado*. Para entender estos dos últimos conceptos, se implementa un **modelo generativo de lenguaje** y el modelo de **k-means clustering**, respectivamente.

(Cada modelo se implementa por separado).
![image](https://github.com/pedro9olivares/ML_and_AI_for_the_Working_Analyst/blob/fe2a60db6968dfb8e9869eed4df2c752f0e717c3/1_Topic_modeling_y_LDA/Esquema_para_LDA.png)

## Índice
* [Modelos generativos de lenguaje](#modelos-generativos-de-lenguaje)
* [K-means clustering](#k-means-clustering)
* [Latent Dirichlet allocation](#latent-dirichlet-allocation)

## Modelos generativos de lenguaje
Un modelo generativo de lenguaje es un modelo estadístico que se encarga de generar texto a partir de un vocabulario dado. En nuestro caso, el vocabulario estará dado por todas las palabras contenidas en el primer libro de Harry Potter. 

Hay dos maneras de generar el texto, la primera es asumiendo una *distribución uniforme* entre todas las palabras del vocabulario y la segunda es asumiendo la *distribución real* de las palabras de nuestro vocabulario.

Se puede utilizar cualquier archivo .txt como vocabulario, si así se desea. Para hacerlo, solamente hace falta:
* subir su .txt a un repositorio de GitHub,
* obtener la liga raw y
* cambiar la siguiente línea por dicha liga raw:
```python
!wget https://raw.githubusercontent.com/sharanyavb/harry-potter/master/Books_Text/HP1.txt
```

### Generación de texto con distribución uniforme
Al asumir una distribución uniforme, solamente estamos eligiendo palabras al azar de nuestro vocabulario y concatenándolas. Predeciblemente, este método no generará textos con mucho sentido.

Después de un preprocesamiento de los datos (eliminar palabras repetidas, expandir contracciones), generamos texto a través de la siguiente instrucción, especificando cuántas palabras queremos generar:
```python
' '.join(random.sample(libro_limpio.split(),15)) 
```
Y obtenemos resultados como el siguiente:

<img width="510" alt="image" src="https://user-images.githubusercontent.com/61219691/159108532-a96fcf4b-fb05-4a7b-b9e5-41d6cb1ac872.png">

### Generación de texto con distribución real
Al tomar en cuenta la distribución real de las palabras, podremos generar un texto mucho más significativo y entendible. En nuestro ejemplo, podemos observar la siguiente distribución: 

<img width="610" alt="image" src="https://user-images.githubusercontent.com/61219691/159108128-288f91f1-fde0-4a68-8a99-b6a70b477169.png">

Con el uso de n-gramas, también podemos considerar aquellas secuencias de palabras con significado específico (por ejemplo, it was, on the, out of the, etc.).
Así, implementamos una función `selecciona_siguiente_token(secuencia)` que inicialmente escoge una palabra al azar del vocabulario para iniciar la secuencia de palabras que se generará. Después, recibirá iterativamente la secuencia de palabras hasta entonces generada para concatenar la palabra más probable que siga. Utilizando esta función obtenemos textos más significativos, como el siguiente:

![image](https://user-images.githubusercontent.com/61219691/159108951-dc7f47ea-a0bb-4215-9f2e-9d2d579a8073.png)

## K-means clustering
El algoritmo de k-means clustering tiene como objetivo particionar una base de datos o *dataset* en k grupos, donde cada registro del dataset pertenece al grupo cuyo valor medio es más cercano. Este algoritmo es un ejemplo de un método de aprendizaje no supervisado, ya que a priori no tenemos una clasificación o etiquetado de nuestros registros.

Matemáticamente, si <img src="https://render.githubusercontent.com/render/math?math=S = \{ x_i \} "> es nuestro dataset 

## Latent Dirichlet allocation
