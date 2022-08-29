# Contra la mil cabezas de la Hidra: ¿Cómo llevar registro de experimentos con ayuda de MLFlow y Hydra?

El código de este repositorio es para la charla del 31 de Agosto de 2022
para la comunidad [Data-AR](https://twitter.com/dataArCommunity).

El notebook de la presentación está en el notebook 
[mlflow-hydra](./mlflow-hydra.ipynb).

## Instalación

Para poder correr el notebook deberán tener Python >= 3.8 (en teoría podrían
hacerlo con versiones arriba de Python 3.6, pero deberán hacer downgrade de
algunos de los paquetes, en particular MLFlow >= 1.24 es sólo compatible con
Python > 3.7). Les recomiendo utilizar [`pyenv`](https://github.com/pyenv/pyenv).

Primero crean el virtual environment e instalan los requerimientos:

    $ python -m venv venv
    $ source ./venv/bin/activate
    (venv) $ pip install --upgrade pip && pip install -r requirements.txt

También es recomendable que instalen [yq](https://github.com/mikefarah/yq) si
quieren correr las celdas que sean del estilo:

    !yq -C . < ./hydra_basic/conf/config.yaml

De todas formas no es necesario, lo único que hacen es mostrar el contenido del
archivo yaml con un pretty print y syntax highlightning. De todas maneras, las
celdas ya fueron pre-computadas y deberían mostrarse correctamente.

De la misma forma, algunas celdas utilizan el comando de GNU/Linux
[`tree`](https://linux.die.net/man/1/tree) que se instala de distintas formas de
acuerdo a su distro (y en muchos casos viene pre-instalado). Nuevamente no es
necesario y las celdas que lo utilizan ya tienen la salida pre-computada.
