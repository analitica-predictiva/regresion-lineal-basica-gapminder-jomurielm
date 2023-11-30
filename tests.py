"""
Calificación del laboratorio
-----------------------------------------------------------------------------------------
"""

import sys

import preguntas


def test_01():
    """
    ---< Input/Output test case >----------------------------------------------------
    Pregunta 01
    pip3 install scikit-learn pandas numpy
    python3 tests.py 01
    (139,)
    (139,)
    (139, 1)
    (139, 1)
    """
    preguntas.pregunta_01()


def test_02():
    """
    ---< Input/Output test case >----------------------------------------------------
    Pregunta 02
    pip3 install scikit-learn pandas numpy
    python3 tests.py 02
    (139, 10)
    -0.7869
    69.6029
    <class 'pandas.core.series.Series'>
    0.629
    """
    preguntas.pregunta_02()


def test_03():
    """
    ---< Input/Output test case >----------------------------------------------------
    Pregunta 03
    pip3 install scikit-learn pandas numpy
    python3 tests.py 03
    0.6192
    """
    preguntas.pregunta_03()


def test_04():
    """
    ---< Input/Output test case >----------------------------------------------------
    Pregunta 04
    pip3 install scikit-learn pandas numpy
    python3 tests.py 04
    R^2: 0.6880
    Root Mean Squared Error: 4.7154
    """
    preguntas.pregunta_04()


test = {
    "01": test_01,
    "02": test_02,
    "03": test_03,
    "04": test_04,
}[sys.argv[1]]

test()
