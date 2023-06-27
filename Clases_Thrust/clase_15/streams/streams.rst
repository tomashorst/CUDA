==========
ICNPG 2013
==========


Clase 15 - Streams
==================


Ancho de banda
--------------

En el directorio ``bandwidth`` hay un benchmark que realiza una medición del ancho de banda entre host
y device copiando un buffer del device al host y otro buffer del host al device.

1. El código no usa *pinned memory* para los buffers del host, lo que hace que las copias sean más lentas. Cambiar el código para que pida *pinned memory* y medir.
2. Las copias se hacen de manera bloqueante y secuencial, pero el bus PCIe es bidireccional. Teniendo los buffers en *pinned memory*, podemos hacer las dos copias de manera simultánea usando *streams*. Cambiar el código para que use *streams* y medir.
3. (Extra) El código mide el tiempo total pero no sabemos si alguno de los dos sentidos de copia es más rápido. Podemos medir estos tiempos insertando *eventos* en cada *stream* antes y después de la copia y midiendo el tiempo transcurrido entre ellos. Realizar este último cambio y medir.


Multiplicación de muchas matrices
---------------------------------

En el directorio ``batchmm`` hay un programa sintético que realiza una serie de multiplicaciones de matrices con el segundo operador fijo, sin realizar superposición entre comunicación y cómputo.

1. Usar streams para ocultar la latencia de las copias. Ver el resultado en el profiler.
2. Reemplazar slow_mm por cublasSgemm(), sin dejar de usar streams.
