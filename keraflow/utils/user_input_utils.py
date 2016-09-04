import numpy as np

from .exceptions import KeraFlowError as KError


class UserInput(object):
    '''Utility class for flexible user input assiging optimizers, regularizers, numpy input...
    To bring both flexibly and convince to argument passing in Keraflow, for some arguments of some functions, users can pass a single value, a dictionary, or a list.  We normalize the input to an ordered list of arrays (same order as `legal_names`). For more information, please refer to Argument Passing section of @ref Tutorials.md"
    '''
    def __init__(self, legal_names):
        '''
        @param legal_names: list of str. Ordered target names of user's input, e.g. ['W', 'b'] for setting layer weight or ['output1', 'output2'] for setting outputports' attributes (loss, metric etc.).
        '''
        self.legal_names = legal_names

    def __call__(self, user_input, array=False, allow_missing=False, no_single_value=True, error_context=''):
        '''
        @param user_input: dict/list/single object(s)
        @param array: boolean. Whether each target accepts an numpy array. If only single target, a list is also acceptable. The list well be automatically transformed to a numpy array.
        @param allow_missing: False or 'skip', 'fill_none'.
            If False, len(user_inputs) should be equal to len(legal_names). Return list of values
            If 'skip', return a list of (name, value) tuples, target that is not specified is not returned.
            If 'fill_none', return a list of (name, value) tuples, target that is not specified is filled with None.
        @param no_single_value: boolean. Whether user_input could be a single value
        @param error_context: str. Debug messages when user_ipnut is not acceptable.

        @return list of values (`allow_missing`=False) or list of (name, value) tuples (otherwise).
        '''
        if not array and no_single_value:
            if not isinstance(user_input, dict) and not isinstance(user_input, list):
                raise KError("You could only pass a dict or a list for "+error_context)

        if array:
            res = self._to_arrays_list(user_input, allow_missing, error_context)
        else:
            res = self._to_list(user_input, allow_missing, error_context)

        if allow_missing=='skip':
            rres = []
            for i, r in enumerate(res):
                if r is not None:
                    rres.append((self.legal_names[i], r))
            return rres
        elif allow_missing=='fill_none':
            return zip(self.legal_names, res)
        else:
            assert not allow_missing, "allow_missing takes only False, 'skip', 'fill_none'!!"
            return res

    def _to_list(self, user_input, allow_missing=False, error_context=''):
        if isinstance(user_input, dict):
            for name in user_input:
                if name not in self.legal_names:
                    raise KError("Unknown name '{}' for {}. Legal names: {}!!".format(name, error_context, self.legal_names))

            res = []
            for name in self.legal_names:
                if name in user_input:
                    res.append(user_input[name])
                elif allow_missing:
                    res.append(None)
                else:
                    raise KError("Missing name '{}' for {}!!.\n Provided names: {}".format(name, error_context, user_input.keys()))
            return res
        elif isinstance(user_input, list):
            if not allow_missing and len(user_input) != len(self.legal_names):
                raise KError('Unmatched length for {}. Expected: {}. Given: {}'.format(error_context, len(self.legal_names), len(user_input)))
            if len(user_input)>len(self.legal_names):
                raise KError('More inputs than expected for {}. Expected: {}. Given: {}'.format(error_context, len(self.legal_names), len(user_input)))
            return user_input+[None]*(len(self.legal_names)-len(user_input))
        else:
            if not allow_missing and len(self.legal_names)!=1:
                raise KError('Unmatched length for {}. Expected: {}. Given: {}'.format(error_context, len(self.legal_names), 1))
            else:
                return [user_input]*len(self.legal_names)

    def _to_arrays_list(self, user_input, allow_missing=False, error_context=''):
        if hasattr(user_input, 'shape'):
            if not allow_missing and len(self.legal_names)!=1:
                raise KError('Unmatched length for {}. Expected: {}. Given: {}'.format(error_context, len(self.legal_names), 1))
            res = [user_input]
        elif type(user_input) is dict:
            res = self._to_list(user_input, allow_missing, error_context)
        elif type(user_input) is list:
            if not hasattr(user_input[0], 'shape'):
                user_input = [np.asarray(user_input)]
            res = self._to_list(user_input, allow_missing, error_context)
        else:
            raise KError('Input user_input should be  be a Numpy array, or list/dict of Numpy arrays!!')

        for i in range(len(res)):
            if res[i] is not None:
                res[i] = np.asarray(res[i])

        return res
