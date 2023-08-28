from dataclasses import dataclass
import struct
from typing import Dict, Tuple

import numpy as np


@dataclass
class Gaussian:
    position: np.ndarray[3]
    scale: np.ndarray[3]
    rotQuat: np.ndarray[4]
    opacity: np.ndarray[1]
    shCoeffs: np.ndarray[3,]
    camera_space_pos: np.ndarray[3] = None

def sigmoid(x):
    return 1 / (1 + np.exp(-x))


class Gaussians:
    """Load a gaussian ply file"""
    def __init__(self, array_buffer: bytes):
        # Decode the header
        vertex_count, property_types, vertex_data = self.decode_header(array_buffer)
        self.num_gaussians = vertex_count
        
        # Figure out the SH degree
        n_rest_coeffs = 0
        for prop in property_types.keys():
            if prop.startswith('f_rest_'):
                n_rest_coeffs += 1
        n_coeffs_per_color = n_rest_coeffs // 3
        self.spherical_harmonics_degree = int(np.sqrt(n_coeffs_per_color + 1)) - 1

        print(self.spherical_harmonics_degree)
        
        # Figure out SH order
        sh_feature_order = []
        for rgb in range(3):
            sh_feature_order.append(f'f_dc_{rgb}')
        for i in range(n_coeffs_per_color):
            for rgb in range(3):
                sh_feature_order.append(f'f_rest_{rgb * n_coeffs_per_color + i}')
                
        # Read and pack gaussians
        self.gaussians = []
        offset = 0
        for i in range(vertex_count):
            offset, raw_vertex = self.read_raw_vertex(offset, vertex_data, property_types)
            self.gaussians.append(
                Gaussian(
                    position=np.array([raw_vertex['x'], raw_vertex['y'], raw_vertex['z']]),
                    scale=np.exp(np.array([raw_vertex['scale_0'], raw_vertex['scale_1'], raw_vertex['scale_2']])),
                    rotQuat=np.array([raw_vertex['rot_0'], raw_vertex['rot_1'], raw_vertex['rot_2'], raw_vertex['rot_3']]),
                    opacity=sigmoid(np.array([raw_vertex['opacity']])),
                    shCoeffs=np.array([raw_vertex[feature] for feature in sh_feature_order], dtype=np.float32).reshape(-1, 3)
                )
            )
            
    def decode_header(self, array_buffer: bytes) -> Tuple[int, Dict[str, str], memoryview]:
        """Decode ply header, return vertex count, property types and vertex data"""
        header_text = array_buffer[:array_buffer.find(b'end_header') + len(b'end_header') + 1].decode('ascii')
        lines = header_text.split('\n')
        
        vertex_count = 0
        property_types = {}
        for line in lines:
            if line.startswith('element vertex'):
                vertex_count = int(line.split()[2])
            elif line.startswith('property'):
                prop = line.split()
                property_types[prop[2]] = prop[1]
                
        vertex_data = memoryview(array_buffer)[len(header_text):]
        return vertex_count, property_types, vertex_data

    def read_raw_vertex(self, offset: int, vertex_data: memoryview, property_types: Dict[str, str]) -> Tuple[int, Dict[str, float]]:
        """Read a single vertex into a dict"""
        vertex = {}
        for prop, prop_type in property_types.items():
            if prop_type == 'float':
                vertex[prop] = struct.unpack_from('f', vertex_data, offset)[0] 
                offset += 4
            elif prop_type == 'uchar':
                vertex[prop] = struct.unpack_from('B', vertex_data, offset)[0] / 255.0
                offset += 1
        return offset, vertex

    @property
    def n_sh_coeffs(self) -> int:
        if self.spherical_harmonics_degree == 0:
            return 1
        elif self.spherical_harmonics_degree == 1: 
            return 4
        elif self.spherical_harmonics_degree == 2:
            return 9
        elif self.spherical_harmonics_degree == 3:
            return 16
        else:
            raise ValueError(f"Unsupported SH degree: {self.spherical_harmonics_degree}")