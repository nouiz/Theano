# Functions for pretty-printing Theano nodes for debugging
import collections

from tensorflow import GraphDef, NodeDef

import theano


def _ReadableOpName(theano_op):
    """Make simplified name, theano.tensor.elemwise.Sum becomes -> Sum."""

    assert isinstance(theano_op, theano.gof.Op)
    opname = theano_op.__class__.__name__
    if opname == "Elemwise":
        opname += ("{" + theano_op.scalar_op.__class__.__name__ + "}")
        return opname


def _ReadableNodeName(theano_node):
    """Give a readable name for Theano node."""

    # Case 1: Tensor constant, use provided signature/hash method
    if type(theano_node).__name__ == "TensorConstant":
        return "const:" + theano_node.signature().theano_hash()
    # Case 2: named TensorVariable, return it's name
    if type(theano_node).__name__ == "TensorVariable":
        # Have to use __name__ instead of isinstance
        # because module theano.tensor.var is shadowed by function
        # theano.tensor.var
        if theano_node.name:
            return theano_node.name
        else:
            # Case 3: unnamed TensorVariable, return it"s id
            return "theano_id:" + str(id(theano_node))
    elif isinstance(theano_node, theano.gof.graph.Apply):
        # Case 4: unnamed Apply node, create canonical name
        # by concatenating it"s operation type and input names.
        # This means graph can"t have two nodes with same type
        # and set of inputs
        # TODO(yaroslavvb): fix the above
        # TODO(yaroslavvb): make sure ordering of inputs is deterministic
        name = _ReadableOpName(theano_node.op) + ":"
        name += ",".join([_ReadableNodeName(i) for i in theano_node.inputs])
        return name
    else:
        assert None, "unsupported node type: " + str(type(theano_node))


def _MakeNode(name, op, inputs=None, signature=None):
    """Helper function to create TensorFlow node."""

    result = NodeDef()
    result.name = name
    result.op = op
    if inputs is not None:
        result.input.extend(inputs)
    if signature is not None:
        result.signature = signature
    return result


def _InitializeTensorShape(tensor_shape, dims_list):
    """Initializes brain.TensorShapeProto with vals from dims_list."""
    for dimsize in dims_list:
        dim = tensor_shape.dim.add()
        dim.size = dimsize


def _InitializeTensorVals(tensor, vals):
    tensor.float_val.extend([float(v) for v in vals])


def _InitializeBoolOpArgs(op_args, bool_vals):
    """Copies bool_vals into brain.OpArgs."""
    for b in bool_vals:
        arg = op_args.args.add()
        arg.b = b


class TensorFlowTheanoFunction(object):
    """Object acts like Theano function, delegates calls to TensorFlow
    client.

    """

    def __init__(self, inputs, outputs, graph_def):
        """Initializes backing TensorFlow client.

        Args:
          inputs: list of strings of input variables
          outputs: list of strings of output variables
          graph_def: graph_pb2.GraphDef object
        """
        self._inputs = inputs
        self._outputs = [output + ":0" for output in outputs]
        self._graph_def = graph_def
        logging.vlog(1, "Making TensorFlow client with graphdef " +
                     str(graph_def))
        logging.vlog(1, "inputs are " + str(self._inputs))
        logging.vlog(1, "outputs are " + str(self._outputs))

        self._session = session.Session("local")
        self._session.Create(graph_def)

    def __call__(self, *args, **kwargs):
        feeds = {}
        for (argpos, arg) in enumerate(args):
            # TODO(yaroslavvb): convert [[]] to numpy
            feeds[self._inputs[argpos]] = arg
            logging.info("Calling run with inputs=%s, outputs=%s",
                         str(feeds.keys()),
                         str(self._outputs))

        fetches = self._session.Run(self._outputs, feeds)
        results = [fetches[output] for output in self._outputs]

        # in the special case of 1 output, extract element from the list
        # to mimic Theano behavior
        if len(results) == 1:
            return fetches[self._outputs[0]]
        return [fetches[output] for output in self._outputs]

    def __del__(self):
        logging.info("Destroying session.")
        self._session.Close()


def ConvertToTensorFlowTheanoFunction(theano_function):
    """Takes theano function object, produces TensorFlowTheanoFunction object.

    Converts Theano graph to TensorFlow graph, instantiates TensorFlow backend
    and wraps it in Theano function-like object

    Args:
      theano_function: Theano function object

    Returns:
      Object that acts like Theano function,
      delegates calls to TensorFlow client.
    """

    assert type(theano_function) == theano.compile.function_module.Function

    # object that holds function graph + inputs + outputs
    function_graph = theano_function.maker.fgraph
    assert type(function_graph) == theano.gof.fg.FunctionGraph

    # save ids because function_graphs gets destroyed later
    input_ids = [id(tensor_var) for tensor_var in function_graph.inputs]
    output_ids = [id(tensor_var) for tensor_var in function_graph.outputs]

    # Convert Theano graph to GraphDef, building up a map of Theano node
    # id's (Python id()) to human readable strings ("0", "1", ...)
    # stored in converter.node_name_map. function_graph is destroyed in the
    # process
    converter = TheanoConverter()
    graph_def = converter.ConvertToGraphDef(function_graph)
    inputs = [converter.node_name_map[i] for i in input_ids]
    outputs = [converter.node_name_map[i] for i in output_ids]

    return TensorFlowTheanoFunction(inputs, outputs, graph_def)


def function(inputs, outputs, mode):
    return TheanoConverter.function(inputs, outputs, mode)


class TheanoConverter(object):
    """Object handling conversion to TensorFlowTheanoFunction.

    Theano graph consists of two kinds of nodes: value nodes (TensorVariable,
    TensorConstant) and apply nodes (theano.gof.graph.Apply). Apply nodes
    represent an operation that takes value nodes as inputs and produces some
    value nodes are outputs. TensorVariable nodes have names that are assigned
    by the user. Apply nodes are crated automatically and are nameless and
    unhashable.

    TensorFlow graph (brain.GraphDef) consists of "Operations". An operation is
    like a named Theano "Apply" node. It's configured with a set of inputs,
    described as a "operation:enpoint", a name, operation description. It's
    output is stored under 1 or more endpoints.

    Theano "value" nodes can be thought of "output endpoints". You can
    "Fetch" an output endpoint or "Feed" a value into it. Because
    endpoints are always associated with some operation, we need some
    dummy operation for theano input value nodes (TensorVariable). Use
    ParamsOp for it.

    TensorConstant could be either implemented as a dummy operation
    where we feed constant value during runtime like for
    TensorVariable, or it can be treated as an Apply node. Do the
    latter since we have a ConstantOperation.

    The conversion happens by iterating over Theano-graph, and calling
    a customizable converter function to produce NodeDef object for
    each Apply node.  Leaf TensorVariable nodes get turned into
    ParamsOp, and TensorConstant nodes get turned into
    ConstantOp. Because Apply nodes don't have names we use names "0",
    "1", ... assigned in order of first time the node was encountered
    during Breadth First Search of the graph starting with inputs.

    """

    def __init__(self):
        """Initializes the converter."""

        self.node_name_map = {}

    @staticmethod
    def function(inputs, outputs=None, mode=None):
        """An analogue of function in theano/compile/function.py."""
        theano_function = theano.function(inputs, outputs, mode)
        tensorflow_function = ConvertToTensorFlowTheanoFunction(
            theano_function)
        return tensorflow_function

    def ConvertToGraphDef(self, function_graph):
        """Convert Theano function object to GraphDef, destroying
        function_graph.

        """

        nodes_queue = collections.deque()
        nodes_queue.extend(function_graph.outputs)
        done_nodes_list = []
        done_nodes_set = set()

        # Breadth-first search starting with outputs
        # nodes may occur more than once
        while nodes_queue:
            current_node = nodes_queue.popleft()
            if current_node not in done_nodes_set:
                done_nodes_list.append(current_node)
                done_nodes_set.add(current_node)
                if current_node.owner:
                    # Iterate over children right-to-left because
                    # later list is reversed and these become
                    # left-to-right
                    nodes_queue.extend(reversed(current_node.owner.inputs))

        # change ordering to more natural "inputs first"
        done_nodes_list.reverse()

        # Build node_name_map. Assing each variable node a number, in the order
        # of inputs-first, left to right BFS.
        node_count = 0
        for node in done_nodes_list:
            if id(node) in self.node_name_map:
                continue
            self.node_name_map[id(node)] = str(node_count)
            node_count += 1

        # Call conversion util for each node to generate list of NodeDef
        tensor_flow_nodes = []  # stored list of brain.NodeDef
        for node in done_nodes_list:
            new_tensor_flow_nodes = []
            if not node.owner:
                # adding leaf node, either TensorConstant or TensorVariable
                new_tensor_flow_nodes = self.LeafNodeConvert(node)
            else:
                # TODO(yaroslavvb): switch to using fully qualified class name
                # adding TensorVariable connected to Apply node
                # (theano.gof.graph.Apply is stored in node.owner)
                # switch conversion based on Theano class name
                theano_op_name = node.owner.op.__class__.__name__
                if theano_op_name == "Sum":
                    new_tensor_flow_nodes = self.SumConvert(node)
                elif theano_op_name == "Elemwise":
                    new_tensor_flow_nodes = self.ElemwiseConvert(node)
                    # theano.tensor.blas.Dot22
                    # theano.tensor.basic.Dot
                elif theano_op_name == "Dot22" or theano_op_name == "Dot":
                    new_tensor_flow_nodes = self.DotConvert(node)
                elif theano_op_name == "DimShuffle":
                    new_tensor_flow_nodes = self.DimShuffleConvert(node)
                else:
                    assert False, ("Unsupported Apply operation: " +
                                   theano_op_name +
                                   " - " + str(node.owner.op.__class__))

            logging.vlog(1, "Converted " + _ReadableNodeName(node) + " to " +
                         ",".join([str(op) for op in new_tensor_flow_nodes]))

            # perhaps do the above automatically like this
            # new_tensor_flow_nodes =
            # (self.theano_map[theano_op_name])(node.owner)

            tensor_flow_nodes.extend(new_tensor_flow_nodes)

        # combined brain.Operation objects into brain.GraphDef
        graph_def = GraphDef()
        graph_def.node.extend(tensor_flow_nodes)
        return graph_def

    def ElemwiseConvert(self, node):
        """Take Theano node attached to Elemwise result, return list
        of NodeDef.

        """

        assert node.owner, "Was expecting node with owner, got leaf node"
        apply_node = node.owner
        assert type(apply_node) == theano.gof.graph.Apply
        assert type(apply_node.op) == theano.tensor.elemwise.Elemwise
        node_name = self.node_name_map[id(node)]
        input_names = [self.node_name_map[id(tensor_var)]
                       for tensor_var in apply_node.inputs]

        if type(apply_node.op.scalar_op) == theano.scalar.basic.Add:
            return [_MakeNode(node_name, "add", input_names,
                              "float,float->float")]
        elif type(apply_node.op.scalar_op) == theano.scalar.basic.Exp:
            return [_MakeNode(node_name, "exp", input_names)]
        elif type(apply_node.op.scalar_op) == theano.scalar.basic.Mul:
            return [_MakeNode(node_name, "mul", input_names)]
        elif type(apply_node.op.scalar_op) == theano.scalar.basic.Neg:
            return [_MakeNode(node_name, "neg", input_names)]
        elif type(apply_node.op.scalar_op) == theano.scalar.basic.Sqr:
            # TODO(yaroslavvb): use Square op instead of mul
            return [_MakeNode(node_name, "mul",
                              [input_names[0], input_names[0]])]
        elif type(apply_node.op.scalar_op) == theano.scalar.basic.Sub:
            return [_MakeNode(node_name, "sub", input_names)]
        elif type(apply_node.op.scalar_op) == theano.scalar.basic.TrueDiv:
            return [_MakeNode(node_name, "div", input_names)]
        else:
            assert False, ("Unknown ElemwiseConvert type " +
                           type(apply_node.op.scalar_op))

    def LeafNodeConvert(self, node):
        """Takes TensorVariable node with no parents, returns
        Parameter node.

        """

        assert node.owner is None, "Must be a leaf node"
        if type(node).__name__ == "TensorVariable":
            assert node.name, "Leaf variable node must have a name"

        node_name = self.node_name_map[id(node)]

        if type(node).__name__ == "TensorVariable":
            return [_MakeNode(node_name, "params", None, "->float,float_ref")]
        elif type(node).__name__ == "TensorConstant":
            tf_node = _MakeNode(node_name, "const", None, "->float")
            tf_tensor = tf_node.Extensions[
                constant_op_pb2.ConstantOpDef.ext].tensor
            _InitializeTensorVals(tf_tensor, [node.tag.unique_value])
            _InitializeTensorShape(tf_tensor.tensor_shape, node.data.shape)
            return [tf_node]

    def DotConvert(self, node):
        """Take Theano Variable node with Dot operation, return list
        of NodeDef.

        """

        apply_node = node.owner
        assert type(apply_node) == theano.gof.graph.Apply
        node_name = self.node_name_map[id(node)]
        input_names = [self.node_name_map[id(tensor_var)]
                       for tensor_var in apply_node.inputs]
        tf_node = _MakeNode(node_name, "matmul", input_names)
#    TODO:
        op_args = tf_node.Extensions[graph_pb2.OpArgs.ext]
        _InitializeBoolOpArgs(op_args, [False, False])

        return [tf_node]
