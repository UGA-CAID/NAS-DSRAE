from collections import namedtuple

Genotype = namedtuple('Genotype', 'recurrent concat')

PRIMITIVES = [
    'none',
    'tanh',
    'relu',
    'sigmoid',
    'identity'    # no change, equal
]
STEPS = 8
CONCAT = 8

#ENAS = Genotype(
#    recurrent = [
#        ('tanh', 0),
#        ('tanh', 1),
#        ('relu', 1),
#        ('tanh', 3),
#        ('tanh', 3),
#        ('relu', 3),
#        ('relu', 4),
#        ('relu', 7),
#        ('relu', 8),
#        ('relu', 8),
#        ('relu', 8),
#    ],
#    concat = [2, 5, 6, 9, 10, 11]
#)
ENAS = Genotype(recurrent=[('relu', 0), ('sigmoid', 0), ('relu', 0), ('identity', 0), ('identity', 4), ('relu', 5), ('tanh', 5), ('identity', 5)],concat=range(1,9))

RNN = Genotype(recurrent=[('tanh', 0)], concat=range(1, 2))
DARTS_V2 = Genotype(recurrent=[('sigmoid', 0), ('relu', 1), ('relu', 1), ('identity', 1), ('tanh', 2), ('sigmoid', 5), ('tanh', 3), ('relu', 5)], concat=range(1, 9))
DARTS_V3 = Genotype(recurrent=[('?', 0), ('?', 1), ('?', 2), ('?', 1),('?',0)], concat=range(1, 4)) #RANDOM SELECT
DARTS_V4 = Genotype(recurrent=[('relu', 0), ('tanh', 1), ('tanh', 1), ('identity', 3), ('identity', 4), ('identity', 0), ('relu', 3), ('identity', 7)], concat=range(1, 9)) #embedding
DARTS = DARTS_V4

