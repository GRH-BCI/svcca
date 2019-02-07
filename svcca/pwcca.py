# Copyright 2018 Google Inc.
#
# Licensed under the Apache License, Version 2.0 (the 'License');
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an 'AS IS' BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# MODIFICATION NOTICE AS PER §4 B) OF THE APACHE LICENSE, VERSION 2.0
# THIS FILE WAS MODIFIED


'''
The function for computing projection weightings.

See:
https://arxiv.org/abs/1806.05759
for full details.

'''

import svcca.linalg as linalg
import svcca.cca_core as cca_core

def compute_pwcca(acts1, acts2, epsilon=0., **kwargs):
    ''' Computes projection weighting for weighting CCA coefficients

    Args:
         acts1: 2d numpy array, shaped (neurons, num_datapoints)
     acts2: 2d numpy array, shaped (neurons, num_datapoints)

    Returns:
     Original cca coefficient mean and weighted mean

    '''
    sresults = cca_core.get_cca_similarity(acts1, acts2, epsilon=epsilon,
                                           compute_dirns=False, compute_coefs=True, **kwargs)
    if linalg.sum(sresults['x_idxs']) <= linalg.sum(sresults['y_idxs']):
        dirns = linalg.dot(sresults['coef_x'],
                           (acts1[sresults['x_idxs']] - \
                            sresults['neuron_means1'])) + sresults['neuron_means1']
        coefs = sresults['cca_coef1']
        acts  = acts1
        idxs  = sresults['x_idxs']
    else:
        dirns   = linalg.dot(sresults['coef_y'],
                           (acts1[sresults['y_idxs']] - \
                            sresults['neuron_means2'])) + sresults['neuron_means2']
        coefs   = sresults['cca_coef2']
        acts    = acts2
        idxs    = sresults['y_idxs']
        P, _    = linalg.qr(linalg.transpose(dirns))
        weights = linalg.sum(
            linalg.abs(linalg.dot(linalg.transpose(P), linalg.transpose(acts[idxs]))),
            axis=1
        )
        weights = weights/linalg.sum(weights)

    return linalg.sum(weights*coefs), weights, coefs
