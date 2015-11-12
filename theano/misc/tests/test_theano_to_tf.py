import unittest

import numpy as np
from tensorflow import *
import tensorflow as tf
import theano.misc.theano_to_tf as TF

import theano
import theano.tensor as tt


class TheanoUtilsTest(unittest.TestCase):
    # Functional tests for individual ops

    def testTrueDiv(self):
        # Functional test for graph that uses TrueDiv
        x = tt.vector('x')
        s = x / 2.
        f1 = theano.function([x], s, mode=theano.compile.FAST_COMPILE)
        f2 = TF.function([x], s, mode=theano.compile.FAST_COMPILE)

        x0 = np.ones(3).astype(np.float32)
        np.testing.assert_array_equal(f1(x0), f2(x0))

        # sanity check, make sure last node is division
        last_op = f1.maker.fgraph.outputs[0].owner.op.scalar_op
        self.assertEquals(type(last_op), theano.scalar.basic.TrueDiv)

    def testNeg(self):
        mode = theano.compile.FAST_COMPILE
        x = tt.vector('x')
        s = -x
        f1 = theano.function([x], s, mode=mode)
        f2 = TF.function([x], s, mode=mode)
        x0 = np.ones((3)).astype(np.float32)
        np.testing.assert_array_equal(f1(x0), f2(x0))

    def testExp(self):
        mode = theano.compile.FAST_COMPILE
        x = tt.vector('x')
        s = tt.exp(x)
        f1 = theano.function([x], s, mode=mode)
        f2 = TF.function([x], s, mode=mode)
        x0 = np.zeros((3)).astype(np.float32)
        np.testing.assert_array_equal(f1(x0), f2(x0))

    def testAdd(self):
        mode = theano.compile.FAST_COMPILE
        x = tt.vector('x')
        y = tt.vector('y')
        s = x + y
        f1 = theano.function([x, y], s, mode=mode)
        f2 = TF.function([x, y], s, mode=mode)

        x0 = np.ones((3)).astype(np.float32)
        y0 = np.ones((3)).astype(np.float32)
        np.testing.assert_array_equal(f1(x0, y0), f2(x0, x0))

    def testDot(self):
        mode = theano.compile.FAST_COMPILE
        x = tt.dmatrix('x')
        y = tt.dmatrix('y')
        f1 = theano.function([x, y], tt.dot(x, y), mode=mode)
        f2 = TF.function([x, y], tt.dot(x, y), mode=mode)
        x0 = np.array([[1, 2], [3, 4]], dtype=np.float32)
        y0 = np.array([[10, 20], [30, 40]], dtype=np.float32)
        np.testing.assert_allclose(f1(x0, y0), f2(x0, y0))
