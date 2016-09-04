import numpy as np
from numpy.testing import assert_allclose

from keraflow.layers import Input
from keraflow.models import Model
from keraflow.utils.generic_utils import unlist_if_one


def layer_test(layer, inp_vals, exp_output=None, random_exp={}, multi_input=False, debug=False, input_args={}, test_serialization=True, train_mode=True):
    if multi_input:
        input_vals = []
        for val in inp_vals:
            input_vals.append(np.asarray(val))
    else:
        input_vals = [np.asarray(inp_vals)]

    if exp_output is not None:
        exp_output = np.asarray(exp_output)

    if 'shape' in input_args:
        if 'batch_size' in input_args:
            input_shapes = [(input_args['batch_size'],)+input_args['shape']]
        else:
            input_shapes = [(None,)+input_args['shape']]
        del input_args['shape']
    else:
        input_shapes = [val.shape for val in input_vals]
    input_layers = [Input(shape[1:], **input_args) for shape in input_shapes]

    model = Model(input_layers, layer(unlist_if_one(input_layers)))
    model.compile('sgd', 'mse')

    output = model.predict(input_vals, train_mode=train_mode)[0]  # result of the first (ant the only) output channel
    output_shape = layer.output_shape(unlist_if_one(input_shapes))

    # check output(), output_shape() implementation
    cls_name = layer.__class__.__name__

    if debug:
        print(cls_name)
        if exp_output is not None:
            print("Expected Output:\n{}".format(exp_output))
        print("Output:\n{}".format(output))
        if exp_output is not None:
            print("Expected Output Shape:\n{}".format(exp_output.shape))
        print("Output shape:\n{}".format(output_shape))
        if debug==2:
            import sys
            sys.exit(-1)

    if exp_output is not None:
        assert_allclose(output, exp_output, err_msg='===={}.output() incorrect!====\n'.format(cls_name))
        if None in output_shape:
            assert output_shape[0] is None
            assert_allclose(output_shape[1:], exp_output.shape[1:], err_msg='===={}.output_shape() incorrect!===='.format(cls_name))
        else:
            assert_allclose(output_shape, exp_output.shape, err_msg='===={}.output_shape() incorrect!===='.format(cls_name))
    else:
        # No exp_output, test if the shape of the output is the same as that provided by output_shape function.
        if None in output_shape:
            assert output_shape[0] is None
            assert_allclose(output_shape[1:], output.shape[1:], err_msg='===={}.output_shape() incorrect!===='.format(cls_name))
        else:
            assert_allclose(output_shape, output.shape, err_msg='===={}.output_shape() incorrect!===='.format(cls_name))

    lim = 1e-2
    if 'std' in random_exp:
        assert abs(output.std() - random_exp['std']) < lim
    if 'mean' in random_exp:
        assert abs(output.mean() - random_exp['mean']) < lim
    if 'max' in random_exp:
        assert abs(output.max() - random_exp['max']) < lim
    if 'min' in random_exp:
        assert abs(output.min() - random_exp['min']) < lim

    if test_serialization:
        # check if layer is ok for serialization
        arch_fname = '/tmp/arch_{}.json'.format(cls_name)
        weight_fname = '/tmp/weight_{}.hkl'.format(cls_name)
        if len(model.trainable_params)==0:
            weight_fname = None

        model.save_to_file(arch_fname, weight_fname, overwrite=True, indent=2)
        try:
            model2 = Model.load_from_file(arch_fname, weight_fname)
        except:
            assert False, '====Reconstruction of the model fails. "{}" serialization problem!===='.format(cls_name)

        model2.compile('sgd', 'mse')
        model2_output = model2.predict(input_vals, train_mode=train_mode)[0]

        if len(random_exp)==0:
            assert_allclose(output, model2_output, err_msg='====Reconstructed model predicts different. "{}" serialization problem!====\n'.format(cls_name))
