import numpy as np
from numpy.random import mtrand
from scipy.spatial.transform import Rotation
import json
from tqdm import tqdm

class GeneratorBase:

    def __init__(self, dimensions, labeled=None):
        self.dimensions = dimensions
        self.labeled = labeled

    def sample(self, rd:np.random.RandomState, defined_values:list):
        raise NotImplementedError("Subclasses should implement this method.")
    

    def _interp(self, cdf, rand_val, idx):
        p1 = cdf[idx - 1] if idx > 0 else 0.0
        p2 = cdf[idx] if idx < len(cdf) - 1 else 1.0
        r = (rand_val - p1) / (p2 - p1)
        pls = 2.0 * np.sqrt(r * 0.5) if rand_val - p1 <= (p2 - p1)*0.5 else 2.0 * (1.0 - np.sqrt((1.0 - r) * 0.5))
        if np.isnan(pls):
            raise ValueError(f"Invalid interpolation parameters pls {pls}.")
        return idx + pls
    

    def serialized(self):
        raise NotImplementedError("Subclasses should implement this method.")
    

    def _sample_plane(self, rd:np.random.RandomState, eps=1e-8):
        
        _max_retry_count = 5
        while True:
            # 由于idx1几乎不可能涉及pdf全0的行，这里基本上不会循环

            rand_val = rd.rand()
            
            idx1 = np.searchsorted(self.cdf[:, -1], rand_val)
            idx1 = np.clip(idx1, 0, self.cdf.shape[1] - 1)

            _pdf = self.pdf[idx1, :]
            _pdf_sum = np.sum(_pdf)

            if _pdf_sum >= eps:
                # print(f"idx: {idx1}, sum: {_pdf_sum}\n _PDF: {_pdf}")
                break

            _max_retry_count -= 1
            if _max_retry_count <= 0:
                # reject this sample
                return None, None

        rand_val2 = rd.rand()
        
        conditional_pdf = _pdf / _pdf_sum
        conditional_cdf = np.cumsum(conditional_pdf)

        idx2 = np.searchsorted(conditional_cdf, rand_val2) 
        
        idx1 = self._interp(self.cdf[:, -1], rand_val, idx1)
        idx2 = self._interp(conditional_cdf, rand_val2, idx2)
    
        # return idx1, idx2
        return (idx1 / self.pdf.shape[0]) - 0.5 , (idx2 / self.pdf.shape[1]) - 0.5


class OneDPDF(GeneratorBase):

    def __init__(self, pdf, dimensions, labeled=None):
        
        super().__init__(dimensions, labeled)
        
        self.pdf = np.array(pdf) / np.sum(pdf) 
        self.cdf = np.cumsum(self.pdf)


    def sample(self, rd:np.random.RandomState, defined_values=None):
        
        rand_val = rd.rand()
        
        idx = np.searchsorted(self.cdf, rand_val, side='left')
        idx = self._interp(self.cdf, rand_val, idx)
        
        return tuple([(idx / len(self.pdf)) - 0.5])


    def serialized(self):
        return {
            'pdf': self.pdf.tolist(),
            'dimensions': self.dimensions,
            'labeled': self.labeled,
        }



class TwoDPDF(GeneratorBase):

    def __init__(self, pdf, dimensions, labeled=None):

        super().__init__(dimensions, labeled)
        
        self.pdf = np.array(pdf) / np.sum(pdf) 
        self.cdf = np.cumsum(self.pdf).reshape(self.pdf.shape)


    def _sample_1(self, rd:np.random.RandomState, defined_axes, defined_value, eps=1e-8):

        org_idxed = int(defined_value * self.pdf.shape[defined_axes])
        idxed = np.clip(org_idxed,   0, self.pdf.shape[defined_axes] - 1)
        
        _pdf = self.pdf[idxed, :] if defined_axes == 0 else self.pdf[:, idxed]

        norm_pdf = np.sum(_pdf)

        if norm_pdf < eps:
            # print("Sampling rejected.")
            ret = [0,0]
            ret[defined_axes] = defined_value
            ret[1 - defined_axes] = None
            return tuple(ret)
        
        rand_val = rd.rand()
            
        conditional_pdf = _pdf / norm_pdf
        conditional_cdf = np.cumsum(conditional_pdf)
        
        idx = np.searchsorted(conditional_cdf, rand_val)
        idx = self._interp(conditional_cdf, rand_val, idx)

        ret = [0,0]
        ret[defined_axes] = defined_value
        ret[1 - defined_axes] = (idx / self.pdf.shape[1 - defined_axes]) - 0.5

        return tuple(ret)


    def sample(self, rd:np.random.RandomState, defined_values):

        if defined_values[0] is None and defined_values[1] is None:
            # all undefined
            return self._sample_plane(rd)
        
        elif defined_values[0] is None: 
            # axes 1 defined
            return self._sample_1(1, rd, defined_values[1])
        
        elif defined_values[1] is None: 
            # axes 2 defined
            return self._sample_1(0, rd, defined_values[0])
        
        else: 
            # all defined
            return defined_values[0], defined_values[1]
        

    def serialized(self):
        return {
            'pdf': self.pdf.tolist(),
            'dimensions': self.dimensions,
            'labeled': self.labeled,
        }


class SampleManifold:

    @staticmethod
    def get_func(func_type:str):
        if func_type == 'swiss_roll':
            return SampleManifold._sample_swiss_roll
        elif func_type == 's_curve':
            return SampleManifold._sample_s_curve
        elif func_type == 'cylinder':
            return SampleManifold._sample_cylinder
        elif func_type == 'none':
            return None
        else:
            raise ValueError(f"Invalid sample function type '{func_type}'.")

    @staticmethod
    def _sample_swiss_roll(sample_t, sample_y):
        
        u = 3
        t = np.pi * (0.0 + u * (sample_t + 0.5))
        max_t = np.pi * (1.0 + u * 1)
        y = sample_y
        x = t * np.cos(t) / max_t
        z = t * np.sin(t) / max_t
        
        return x, y, z


    @staticmethod
    def _sample_cylinder(sample_t, sample_y):

        t = 2 * np.pi * sample_t
        y = sample_y
        x = np.cos(t) * 0.5
        z = np.sin(t) * 0.5

        return x,y,z


    @staticmethod
    def _sample_s_curve(sample_t, sample_y):

        t = 3.0 * np.pi * sample_t
        y = sample_y
        x = np.sin(t) * 0.5
        z = np.sign(t) * (np.cos(t) - 1) * 0.5
        
        return x,y,z


class ThreeDPDF(GeneratorBase):

    def __init__(self, pdf, dimensions, manifold:str, labeled=None, rotations:list=[0,0,0], rot_axis='xyz'):
        
        super().__init__(dimensions, labeled)
        
        self.pdf = np.array(pdf) / np.sum(pdf) 
        self.cdf = np.cumsum(self.pdf).reshape(self.pdf.shape)

        self.rotations = rotations
        self.rot_axis = rot_axis

        self.rotation = Rotation.from_euler(rot_axis, rotations, degrees=True)

        self.manifold = manifold
        self._manidold_func = SampleManifold.get_func(manifold)


    def _sample(self, rd:np.random.RandomState):
        idx1, idx2 = self._sample_plane(rd)

        point = self._manidold_func(idx1, idx2)

        point = self.rotation.apply(list(point))

        return tuple(point)


    def sample(self, rd:np.random.RandomState, defined_values):
        if all(val is None for val in defined_values):
            return self._sample(rd)
        else:
            return tuple(defined_values)


    def serialized(self):
        return {
            'pdf': self.pdf.tolist(),
            'dimensions': self.dimensions,
            'manifold': self.manifold,
            'labeled': self.labeled,
            'rotations': self.rotations,
            'rot_axis': self.rot_axis
        }


class PDP(ThreeDPDF):
    def __init__(self, pdf, dimensions, labeled=None, rotations:list=[0,0,0], rot_axis='xyz'):
        super().__init__(pdf, dimensions, 'none', labeled, rotations, rot_axis)


    def _sample(self, rd:np.random.RandomState):

        idx1, idx2 = self._sample_plane(rd)
        
        point = self.rotation.apply([idx1, idx2, 0])

        return tuple(point)

    
    def serialized(self):
        return {
            'pdf': self.pdf.tolist(),
            'dimensions': self.dimensions,
            'labeled': self.labeled,
            'rotations': self.rotations,
            'rot_axis': self.rot_axis
        }


class SynGenerator:

    def __init__(self, samples, dims, generators:list, weights:list):
        
        self.samples = samples
        self.dims = dims
        self.generators = generators
        self.weights = weights

        self._check_generators()


    def _check_generators(self):
        related_dim_set = set()
        oneD_dim_set = set()
        for gen in self.generators:
            if isinstance(gen, OneDPDF):
                oneD_dim_set.add(gen.dimensions[0])
            for dim in gen.dimensions:
                related_dim_set.add(dim)
        
        total_dim = set(range(self.dims))

        without_oneD = list(total_dim - oneD_dim_set)
        if len(without_oneD) > 0:
            print(f"Warning, dim {list(without_oneD)} is not related to any 1D generator, add uniform 1d generator")
            for dim in without_oneD:
                self.generators.append(OneDPDF(np.ones((11)), [dim], labeled=None))
                self.weights.append(1)

        undefined_dims = list(related_dim_set - total_dim)
        if len(undefined_dims) > 0:
            raise ValueError(f"Dim {undefined_dims} is undefined")


    def generate(self, seed=None):
        
        rd = mtrand.RandomState(seed)

        dataset = np.zeros((self.samples, self.dims))
        labels = np.zeros(self.samples)

        p_weights = np.array(self.weights) / np.sum(self.weights)

        for j in tqdm(range(self.samples)):
            sample = [None] * self.dims
            label = 0  
            for i in range(self.dims):
                while sample[i] is None:  
                    
                    generator = rd.choice(self.generators, p=p_weights)

                    if i in generator.dimensions:
                        
                        dims = generator.dimensions
                        values = [sample[dim] for dim in dims]
                        
                        sampled_values = generator.sample(rd, values)

                        has_none = False
                        
                        for dim, val in zip(dims, sampled_values):
                            
                            sample[dim] = val
                            
                            has_none = True if val is None else has_none

                        if has_none is False and generator.labeled is not None:
                            
                            label = generator.labeled

            dataset[j, :] = sample 
            labels[j] = label 
        
        return dataset, labels
    

class SynGeneratorParser:

    @staticmethod
    def load(filename):
        
        with open(filename, 'r', encoding='utf-8') as f:

            s = f.read()
            config = json.loads(s)

            generators = []
            weights = []
            for gen in config['generators']:
                t = globals()[gen['type']](**gen['params'])
                generators.append(t)
                weights.append(gen['weight'])
            
            return SynGenerator(config['samples'], config['dims'], generators, weights)
    

    @staticmethod
    def save(filename, obj:SynGenerator):
                
        config = {
            'samples': obj.samples,
            'dims': obj.dims,
            'generators': [
                {'type':type(gen).__name__, 'weight':weight, 'params':gen.serialized()} 
                    for gen,weight in zip(obj.generators, obj.weights)]
        }

        with open(filename, 'w', encoding='utf-8') as f:
            f.write(json.dumps(config, indent=4, ensure_ascii=False))
            f.flush()

