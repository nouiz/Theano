from itertools import izip

import numpy
from theano import Op, Apply, scalar
from theano.tensor.elemwise import Elemwise

try:
    import pygpu
    from pygpu.tools import ScalarArg, ArrayArg
    from pygpu.elemwise import ElemwiseKernel
except ImportError:
    pass

from theano.sandbox.gpuarray.basic_ops import as_gpuarray_variable
from theano.sandbox.gpuarray.type import GpuArrayType

from theano.gof.utils import MethodNotDefined

def _is_scalar(v):
    False

def make_argument(v, name):
    if _is_scalar(v):
        return ScalarArg(numpy.dtype(v.type.dtype), name)
    else:
        return ArrayArg(numpy.dtype(v.type.dtype), name)

def ensure_allocated(storage, shape, dtype):
    odat = storage[0]
    if odat is not None:
        if odat.shape != shape:
            # It is unsafe to try to resize odat,
            # we have to allocate output storage.
            odat = None
    if odat is None:
        odat = pygpu.empty(shape, dtype=dtype)
    storage[0] = odat
    return odat

def as_C_string_const(s):
    return '\n'.join('"%s\\n"' % (l.replace('"', '\\"'))
                     for l in s.split('\n'))

class GpuElemwise(Elemwise):
    nin = property(lambda self: self.scalar_op.nin)
    nout = property(lambda self: self.scalar_op.nout)

    def __init__(self, scalar_op, name=None, nfunc_spec=None):
        # We do not support inplace since it is a lie anyway
        # (the scalar_op code will never modify anything inplace)
        Elemwise.__init__(self, scalar_op, inplace_pattern=None, name=name,
                          nfunc_spec=nfunc_spec)

    def __str__(self):
        if self.name is not None:
            return self.name
        return "GpuElemwise{%s}<gpuarray>" % (self.scalar_op,)

    def make_node(self, *inputs):
        res = Elemwise.make_node(self, *inputs)
        outputs = [GpuArrayType(broadcastable=o.type.broadcastable,
                                dtype=o.type.dtype)() for o in res.outputs]
        inputs = [as_gpuarray_variable(i) for i in inputs]
        res = Apply(self, inputs, outputs)
        # Try to generate the kernel to catch SupportCodeErrors
        k = self.generate_kernel(res, 'test')
        return res

    def generate_kernel(self, node, nodename):
        inps = [make_argument(i, 'i%d' % (n,)) for n, i in
                enumerate(node.inputs)]
        scal_ins = [scalar.Scalar(i.dtype) for i in node.inputs]

        outs = [make_argument(o, 'o%d' % (n,)) for n, o in
                enumerate(node.outputs)]
        scal_out = [scalar.Scalar(o.dtype) for o in node.outputs]

        fake_node = Apply(self.scalar_op, [i() for i in scal_ins],
                          [o() for o in scal_out])

        try:
            code = self.scalar_op.c_support_code_apply(fake_node, nodename)
            if code:
                raise SupportCodeError(code)
        except MethodNotDefined:
            pass

        support_code = ""
        try:
            support_code = self.scalar_op.c_support_code()
        except MethodNotDefined:
            pass

        if (support_code != "#define THEANO_MACRO_MOD(x,y) (x % y)" and
            support_code != ""):
            # The macro is fine, the C++ struct is not.
            raise SupportCodeError(support_code)

        kop = self.scalar_op.c_code(fake_node, nodename+'_scalar',
                                    [i.name+'[i]' for i in inps],
                                    [o.name+'[i]' for o in outs],
                                    dict(fail='return;'))

        # Translate types for scalar composite ops (except complex).
        support_code += """
#define npy_float64 ga_double
#define npy_float32 ga_float
#define npy_uint8 ga_ubyte
#define npy_int8 ga_byte
#define npy_uint16 ga_ushort
#define npy_int16 ga_short
#define npy_uint32 ga_uint
#define npy_int32 ga_int
#define npy_uint64 ga_ulong
#define npy_int64 ga_long
"""
        return ElemwiseKernel(None, inps+outs, kop, preamble=support_code)

    def c_support_code_apply(self, node, nodename):
        # This is useless by itself, but will serve an eventual c_code
        # implementation
        k = self.generate_kernel(node, nodename)

        nd = node.inputs[0].type.ndim
        res = []
        for i in range(1, nd):
            var = "static const char %s_%s[] = " % (nodename, str(i))
            res.append(var + as_C_string_const(k.render_basic(i)) + ';')
            res.append("static const gpukernel *%s_%s_k = NULL;" % (nodename,
                                                                    str(i)))
        var = "static const char %s_c[] = " % (nodename,)
        res.append(var + as_C_string_const(k.contig_src) + ';')
        res.append("static const gpukernel *%s_c_k = NULL;" % (nodename,))
        return '\n'.join(res)

    def c_code(self, *args):
        # do not pick up the Elemwise version
        raise MethodNotDefined('c_code')

    def perform(self, node, inputs, output_storage):
        # Try to reuse the kernel from a previous call to hopefully
        # avoid recompiling
        if not hasattr(node, '_cache_elemwise_k'):
            node._cache_elemwise_k = self.generate_kernel(node, "kcode")

        out_shape = []
        for values in izip(*[input.shape for input in inputs]):
            if any(v == 0 for v in values):
                # All non-broadcasted dimensions should be zero
                assert max(values) <= 1
                out_shape.append(0)
            else:
                out_shape.append(max(values))
        out_shape = tuple(out_shape)

        outs = [ensure_allocated(storage, out_shape, output.type.dtype)
                for output, storage in izip(node.outputs, output_storage)]

        # the dict call is there to avoid a syntax error in python < 2.6
        node._cache_elemwise_k(*(inputs+outs), **dict(broadcast=True))


class SupportCodeError(Exception):
    """
    We do not support certain things (such as the C++ complex struct)
    """
