import json

import numpy as np


def is_diagonal(matrix):
    return np.count_nonzero(matrix - np.diag(np.diagonal(matrix))) == 0


def is_identity(matrix):
    return (is_diagonal(matrix) and
            np.all(np.diag(matrix) == 1))


def is_zeros(array):
    return np.all(array == 0)


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            # process several special cases: zeros, diagonal, identity
            values = None
            if is_zeros(obj):
                constructor = 'zeros'
            elif is_identity(obj):
                constructor = 'identity'
            elif is_diagonal(obj):
                constructor = 'diag'
                values = np.diagonal(obj).tolist()
            else:
                constructor = 'array'
                values = obj.tolist()

            return {
                'dtype': f'{obj.dtype}',
                'constructor': constructor,
                'values': values,
                'shape': obj.shape
            }

        return json.JSONEncoder.default(self, obj)


def decode_object(decoded_dict):
    if 'dtype' in decoded_dict:
        constructor_name = decoded_dict['constructor']
        if constructor_name == 'array':
            return np.array(decoded_dict['values'], dtype=decoded_dict['dtype'])
        elif constructor_name == 'diag':
            return np.diag(decoded_dict['values']).astype(decoded_dict['dtype'])
        elif constructor_name == 'identity':
            return np.identity(decoded_dict['shape'][0], dtype=decoded_dict['dtype'])
        else:
            constructor = getattr(np, constructor_name)
            return constructor(decoded_dict['shape'], dtype=decoded_dict['dtype'])

    return decoded_dict
