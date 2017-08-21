Spanish Wikipedia Clustering
============================

GPU-based clustering over spanish wikipedia articles.

This project was made as part of an Artificial Intelligence undergrad Computer
Science course in 2015 by **Juan Carlos Pujol Mainegra** and **Damian Valdés
Santiago**.

Resumen
=======

En este trabajo se realiza una experimentación con los artículos de Wikipedia en
español, basada en métodos de agrupamiento (e.g. $k$-means, *DBSCAN*) y medidas
de evaluación de los mismos (e.g. *adjusted rand index*, *adjusted mutual
information*, *homogeneity*, *completeness*, *V-measure* y *silhouette
coefficient*). Estas medidas se basan en un conocimiento previo de las etiquetas
de los grupos, las cuales fueron extraídas de las fichas que contiene cada
artículo de Wikipedia. Se implementó como método de agrupamiento una versión de
$k$-means *n*-dimensional usando computación paralela en GPU, obteniéndose
resultados positivos. Como herramientas de software se utilizaron el lenguaje
`Python 2.7` y los módulos `nltk 3.0`, `sklearn 0.13`, `numpy 1.6`, `scipy 0.11`
y `py-opencl`.

Procesamiento de los artículos
==============================

Primeramente, se transformaron los JSON que contenían los artículos, en
elementos de una tabla de la base de datos `wiki.db` cuyo tamaño aproximado es
de 7 Gb, y donde cada artículo se guarda con un `id`, el `título`, el `usuario`
que lo publicó y el `texto` en el propio artículo. Este procedimiento se realizó
con el código `database_fill`.\ Luego, se realizó un estudio exploratorio sobre
los textos de los artículos en el formato específico de la Wikipedia para
determinar qué elementos pudieran eliminarse con el fin de refinar el texto para
después realizar el procesamiento de texto clásico. Se borran entonces las
tablas, plantillas, hipervínculos, referencias, fórmulas matemáticas, etc. Esto
se logró mediante expresiones regulares en Python y un parser XML.\ Este
filtrado del texto se realizó con el código en `plain_text` guardando el
resultado en la base de datos `miniwiki.db` cuya tabla `corpus_mini` contiene
`id`, `título`, `usuario` y el `texto` procesado. Además contiene una llave
foránea a la tabla `topics` donde se almacenan las etiquetas dadas por los
autores a cada artículo, lo que servirá para evaluar posteriormente la calidad
del agrupamiento. En el estudio se detectaron 518 tópicos.\ Durante este
proceder se hallaron artículos que no cumplen con el formato observado, por
ejemplo, no había un cierre equilibrado de corchetes (`[[]]`). Uno de estos
artículos es el referido a Azerbaiyán donde la tabla Clima no tiene el cierre
debido.

Extracción de características
=============================

Se utilizó el módulo `nltk` para la eliminación de las *stop words* y el
*Snowball stemming*. La experimentación demostró que estos dos procesamientos, a
pesar de su demora, no mejoran la eficacia del agrupamiento. Para la
construcción de los vectores de caracteristicas se utilizó un modelo vectorial
basado en frecuencia de términos y frecuencia inversa de términos (*tf-idf*).
Esta vectorización se realizó con la clase `TfidfVectorizer` del módulo
`sklearn`.

Algoritmos de agrupamiento
==========================

El agrupamiento (*clustering*) consiste en agrupar objetos similares entre ellos
y disimilares respecto a los objetos fuera del grupo. Existen muchos algoritmos
para clustering cuyo denominador común es el uso de distancias que permitan
determinar la similaridad o no de los datos. En el caso restrictivo donde cada
objeto sea descrito por los valores de solo dos atributos, se pueden representar
los datos en un plano como se muestra en la Figura \[fig:Data\].

![fig: Data](./doc/Data.svg)
fig: Data. Datos para agrupar [@Bramer2013].

Los puntos de la Figura \[fig:Data\] pueden agruparse visualmente en
cuatro grupos que se muestran en la Figura \[fig:Cluster1\].

![fig: Cluster1](./doc/Cluster1.svg)
fig: Cluster1. Datos agrupados [@Bramer2013]

Sin embargo, frecuentemente existen varias posibilidades para hacer el
agrupamiento. Por ejemplo, los puntos en la esquina inferior derecha de la
Figura \[fig:Data\] puede agruparse con dos clusters como se muestra en Figura
\[fig:Cluster2\].

![fig:Cluster2](./doc/Cluster2.svg)
fig:Cluster2. Datos agrupados (segunda versión) [@Bramer2013]

Estas distintas cantidades de clusters para un mismo conjunto de datos se debe a
la noción de similitud (disimilitud) entre dos datos que se use en el algoritmo.
La medida más usada es la distancia euclidiana.

$k$-means {#kmeans}
---------

El algoritmo $k$-means agrupa los datos intentando separar los datos en $n$
grupos de igual varianza, minimizando el criterio de la inercia entre grupos.
Este método requiere la especificación del número de grupos deseados. Se
comporta bien con un número creciente de datos pero los resultados del
agrupamiento depende de la inicialización. Como resultado, el cómputo es a
menudo costoso para diferentes inicializaciones de los centroides (puntos cuyas
coordenadas son el promedio de los puntos de cada grupo).

El algoritmo determina los $n$ centroides iniciales de manera aleatoria o
prefijada. Se computan las distancias de cada centroide al resto de los puntos
incorporando al grupo los $k$ datos más cercanos según la distancia euclidiana.
Una vez que un dato se incorpora a un grupo, se excluye del cómputo que
realizarán el resto de los centroides, pues este algoritmo debe particionar el
conjunto de datos. Luego de este procedimiento, se recalculan los centroides y
se repite el proceso hasta que se exceda un nuúmero de iteraciones, no cambien
los centroides de manera significativa entre cada paso u otro criterio de parada
que sea necesario. Al concluir el método se obtienen los datos particionados en
$n$ grupos.

Para la experimentación se usó la implementación de $k$-means presente en el
módulo `sklearn`.

$k$-means con OpenCL
--------------------

Esta variante consiste en el mismo algoritmo descrito en **kmeans** pero con
paralelización. Para esto se partió del ejemplo *k-means autoclustering* que
brinda AMD APP SDK versión 3.0 beta, y se modificó para trabajar con vectores de
$n$ dimensiones. Cada *kernel* (hilo de ejecución) computa la distancia de un
vector a todos los centroides, dejando al CPU la actualización de los mismos en
cada paso.

DBSCAN
------

El algoritmo DBSCAN agrupa los datos los puntos núcleo que tienen cierta
cantidad de vecinos en un cierto radio de acción. Después que se determinan los
núcleos, el grupo se expande agregando sus vecinos al grupo que contiene al
núcleo y se chequea recursivamente si alguno de ellos es un núcleo. Formalmente,
un punto es considerado núcleo si tiene más de un mínimo de puntos que tienen
una similaridad mayor que un umbral determinado. Este algoritmo es muy usado
para detectar *outliers* o ruido.

Comparación
-----------

Método    |Parámetros            | Escalabilidad  |Casos de uso                                                           | Distancia
----------|----------------------| ---------------|-----------------------------------------------------------------------|--------------------------------------
$k$-means |Número de grupos      | Muy grande     |Propóstio general, número par de grupos, pocos grupos, geometría plana | Distancia entre puntos
DBSCAN    |Tamaño de la vecindad | Muy grande     |Geometría no plana, tamaño impar de grupos                             | Distancia entre puntos más cercanos

#### 

Se les llama *grupos de geometría no plana* a los grupos cuya forma no
es circular, cuadrada, etc.

Evaluación del agrupamiento
===========================

Adjusted Rand Index
-------------------

Mide cuán similares son dos asignaciones de cluster: la real y la dada por un
algoritmo. El número que se le de al cluster no tiene importancia. Mientras más
cercano es su valor a 1 mejor es el agrupamiento, es peor si el valor es
negativo o cercano a 1. El rango de valores es $[-1,1]$. Esta medida no le da
importancia a la estructura del agrupamiento.

Si $C$ son las asignaciones de clase reales de los datos y $K$ las del
agrupamiento, se definen $a$ y $b$ como el número de pares de elementos que
están en la misma clase en $C$ y en el mismo grupo en $K$; y el número de pares
de elementos que están en diferentes clases en $C$ y en diferentes grupos en
$K$, respectivamente.

El Rand Index sin ajustar es: $$RI = \frac{a + b}{C_2^{n_{samples}}}$$ donde
$C_2^{n_{samples}}$ es el número total de posibles pares en el conjunto de
datos.

El Rand Index ajustado es:
$$ARI = \frac{RI - Expected_{RI}}{max(RI) - Expected_{RI}}$$

Adjusted Mutual Information
---------------------------

Similar al Rand Index. El rango de valores es $[0,1]$. El valor 0 indica
independencia pura y 1 igualdad de agrupamiento.

Homogeneity y Completeness
--------------------------

Rosenberg and Hirschberg (2007) definen caracteristicas deseables en cualquier
agrupamiento:

-   Homogenity: cada cluster contiene solo miembros de su clase.

-   Completeness: todos los miembros de la clase son asignados al mismo
    cluster.

Ambas medidas están en el rango $[0,1]$ y mientras el valor esté más cercano a
1, mejor será el agrupamiento.

Las definiciones formales son:
$$h = 1 - \frac{H(C|K)}{H(C)} \qquad c = 1 - \frac{H(K|C)}{H(K)}$$ donde
$H(C|K)$ es la entropía condicional de las clases dadas las asignaciones
de los grupos:
$$H(C|K) = - \sum_{c=1}^{|C|} \sum_{k=1}^{|K|} \frac{n_{c,k}}{n} \cdot log \left( \frac{n_{c,k}}{n_k} \right)$$
y $H(C)$ es la entropía de las clases:
$$H(C) = - \sum_{c=1}^{|C|} \frac{n_c}{n} \cdot log \left( \frac{n_c}{n} \right)$$
donde $n$ es el número de datos, $n_c$ y $n_k$ es el número de datos en
la clase $c$ y el grupo $k$ respectivamente, y $n_{c,k}$ es el número de
datos de la clase $c$ asignados al grupo $k$.

V-Measure
---------

Esta medida puede usarse para evaluar la similitud entre dos asignaciones
independientes en el mismo conjunto de datos. Si es mala (cercana a 0), se usan
*homogenity* y *completeness* como medida de calidad. No asume estructura del
cluster.

La definición formal es: $$v = 2 \cdot \frac{h \cdot c}{h + c}$$

Silhoutte Coefficient
---------------------

No se necesita la asignación correcta y la dada por el algoritmo de
agrupamiento. Mientras mayor sea el coeficiente, más definidos estarán los
grupos. El rango de valores es $[-1,1]$, el valor $-1$ indica un mal
agrupamiento, 1 indica un agrupamiento denso y bien separado, y 0 indica grupos
superpuestos. Funciona bien en metodos basados en densidad (e.g. DBSCAN).

La definición formal es: $$s = \frac{b - a}{\max(a, b)}$$ donde $a$ es la
distancia promedio entre cada dato y todos los puntos en su mismo grupo, y $b$
es la distancia promedio entre cada dato y todos los puntos del grupo más
cercano al grupo del dato analizado.

Experimentación
===============

La experimentación se realiza con el archivo `vectorize.py` donde pueden hacer
dos tipos de pruebas: agrupando los artículos solo en dos grupos o agrupándolos
en un número creciente de grupos.

Dos grupos con 1000 artículos
-----------------------------

Se toman los primeros 1000 artículos que están en la base de datos. El método
$k$-means se ejecuta con $k=?$ y elección aleatoria de centroides. DBSCAN se
ejecuta con $eps = $0.83, $min_samples = 1$. Mean Shift se ejecuta con la
estimación por defecro para el ancho de banda.\ Con los tópicos *Ficha de
ciclista* (con 1219 artículos) y *Ficha de científico* (con 2438 artículos). Al
ejecutar la experimentación se obtuvieron los siguientes resultados: Términos
obtenidos luego de la vectorización:

    [   15px            2000            2001            2002            2003           
        2004            2005            2006            2007            2008           
        2009            2010            2011            2012            2013           
        2014            20px            2º              alemania        amateur        
        and             archivo         argentina       año             años           
        botánica        botánicos       campeonato      carrera         categoría      
        ciclismo        ciclista        ciclistas       ciencias        colombia       
        contrarreloj    cup             después         dos             ed             
        enlaces         equipo          equipos         españa          estudios       
        et              etapa           etapas          externos        francia        
        física          ganó            general         giro            gold           
        gran            grandes         historia        http            ik             
        in              instituto       investigación   isbn            italia         
        juegos          medal           miembro         mundial         mundo          
        nacido          nacional        of              olímpicos       palmarés       
        parte           png             pp              premio          primera        
        profesional     profesor        publicaciones   referencias     resultados     
        ruta            ser             siglo           svg             team           
        template        teoría          the             tour            trabajo        
        unidos          universidad     vuelta          vueltas         with            ]           


Método        |     ARI      |     AMI      |     H      |     C      |     V      |      SC      | Grupos | Tiempo (s)
--------------|--------------|--------------|------------|------------|------------|--------------|--------|------------
$k$-means     |  0.9979999   |  0.9942933   | 0.9942953  | 0.9942961  |  0.99429   |  0.3580206   |   2    | 0.08099985
KMeansOpenCL  |  0.9979999   |  0.9942933   | 0.9942953  | 0.9942961  | 0.9942957  |  0.3580206   |   2    | 0.01300001
DBSCAN        | 0.0002829324 | 0.0001105662 | 0.03061625 | 0.08104242 | 0.04444286 | $-$0.4573770 |   57   |   0.147

Número creciente de grupos
--------------------------

Se tomaron combinaciones de grupos de los siguientes tópicos:

Cantidad de documentos |             Tópicos
-----------------------|---------------------------------
76001                  |         Ficha de taxón
64854                  |  Ficha de entidad subnacional
12488                  |       Ficha de localidad
11207                  |       Ficha de futbolista
10439                  |        Ficha de artista
9068                   |         Ficha de actor
7987                   |         Ficha de álbum
7632                   |        Ficha de película
7443                   |        Ficha de persona
6140                   |       Ficha de deportista
5788                   |       Ficha de autoridad
5547                   |        Ficha de estación
4404                   |        Ficha de sencillo
3568                   |  Ficha de serie de televisión
3170                   |      Ficha de organización
3126                   |    Ficha de equipo de fútbol
2954                   |        Ficha de escritor
2731                   |        Ficha de revista
2635                   |          Ficha de isla
2438                   |       Ficha de científico
2086                   |       Ficha de videojuego
2085                   |     Ficha de cuerpo de agua
2051                   |   Ficha de vía de transporte
1874                   |          Ficha de país
1788                   |         Ficha de noble
1697                   |         Ficha de libro
1640                   |     Ficha de obra de teatro
1605                   |       Ficha de conflicto
1520                   |         Ficha de templo
1390                   |        Ficha de edificio
1388                   |        Ficha de montaña
1339                   | Ficha de episodio de televisión
1219                   |        Ficha de ciclista
1201                   |    Ficha de espacio natural
1192                   |        Ficha de militar
1174                   |    Ficha de torneo de fútbol
1147                   |   Ficha de compuesto químico
1133                   |   Ficha de transporte público
1083                   |        Ficha de partido
1081                   |   Ficha de recinto deportivo
1043                   |      Ficha de universidad
1011                   |     Ficha de cuerpo celeste

#### 

Se obtuvieron los siguientes resultados con Ficha de torneo de fútbol y
Ficha de escritor:

Método        |    ARI     |     AMI     |     H      |     C      |     V      |      SC       | Grupos |   Tiempo
--------------|------------|-------------|------------|------------|------------|---------------|--------|-------------
$k$-means     | 0.9920120  |  0.9811704  | 0.9811772  | 0.9811886  | 0.9811829  |   0.3779664   |   2    | 0.08299994
KMeansOpenCL  | 0.9920120  |  0.9811704  | 0.9811772  | 0.9811886  | 0.9811829  |   0.3779664   |   2    | 0.009999990
DBSCAN        | 0.00738551 | 0.009947429 | 0.08011485 | 0.08281686 | 0.08144345 | $-$0.45190686 |   16   |  0.1289999

#### 

Con Ficha de estación, Ficha de serie de televisión, Ficha de videojuego y Ficha
de templo se obtuvo que:

Método        |    ARI     |     AMI     |     H      |     C     |     V      |      SC      | Grupos |  Tiempo
--------------|------------|-------------|------------|-----------|------------|--------------|--------|-----------
$k$-means     | 0.9622013  |  0.9480221  | 0.9480643  | 0.9483980 |  0.948231  |   0.405231   |   4    | 0.312000
KMeansOpenCL  | 0.6205381  |  0.7150742  | 0.7153060  | 0.8220825 | 0.76498631 |  0.19023479  |   4    | 0.031000
DBSCAN        | 0.00161035 | 0.002103556 | 0.04129197 | 0.1498903 | 0.06474729 | $-$0.4775756 |   65   | 0.5250000

Discusión
=========

En general, el algoritmo $k$-means es el que mejor comportamiento tiene mientras
que DBSCAN etiqueta los datos en muchos grupos obteniéndose malos resultados. La
causa de esto puede ser la existencia de términos comunes, lo que hace que la
vectorización no logre discriminar correctamente.

A raíz de estos resultados se realizó una extracción de características tomando
los términos exclusivos de las clases conocidas, i.e. eliminando los términos
comunes de las clases. De esta forma se obtuvieron resultados prometedores.
