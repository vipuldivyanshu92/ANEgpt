"""
ane_bridge.py — Python ctypes wrapper for ANE private API bridge (libane_bridge.dylib)

Provides a Pythonic interface to compile and run MIL programs on Apple Neural Engine
via the reverse-engineered _ANEClient/_ANECompiler private APIs.
"""

import ctypes
import ctypes.util
import os
import struct
import numpy as np
from pathlib import Path
from typing import Optional, List, Tuple

# Locate the shared library
_LIB_SEARCH_PATHS = [
    Path(__file__).parent.parent.parent / "ane-training" / "bridge" / "libane_bridge.dylib",
    Path.home() / "workspace" / "control-room" / "ane-training" / "bridge" / "libane_bridge.dylib",
    Path("/usr/local/lib/libane_bridge.dylib"),
]

_lib = None

def _load_lib():
    global _lib
    if _lib is not None:
        return _lib
    for p in _LIB_SEARCH_PATHS:
        if p.exists():
            _lib = ctypes.CDLL(str(p))
            break
    if _lib is None:
        raise RuntimeError(
            "Cannot find libane_bridge.dylib. Build it with:\n"
            "  cd ane-training/bridge && make"
        )

    _lib.ane_bridge_init.restype = ctypes.c_int
    _lib.ane_bridge_init.argtypes = []

    _lib.ane_bridge_compile.restype = ctypes.c_void_p
    _lib.ane_bridge_compile.argtypes = [
        ctypes.c_char_p, ctypes.c_size_t,
        ctypes.c_void_p, ctypes.c_size_t,
        ctypes.c_int, ctypes.POINTER(ctypes.c_size_t),
        ctypes.c_int, ctypes.POINTER(ctypes.c_size_t),
    ]

    _lib.ane_bridge_compile_multi_weights.restype = ctypes.c_void_p
    _lib.ane_bridge_compile_multi_weights.argtypes = [
        ctypes.c_char_p, ctypes.c_size_t,
        ctypes.POINTER(ctypes.c_char_p),
        ctypes.POINTER(ctypes.c_void_p),
        ctypes.POINTER(ctypes.c_size_t),
        ctypes.c_int,
        ctypes.c_int, ctypes.POINTER(ctypes.c_size_t),
        ctypes.c_int, ctypes.POINTER(ctypes.c_size_t),
    ]

    _lib.ane_bridge_eval.restype = ctypes.c_bool
    _lib.ane_bridge_eval.argtypes = [ctypes.c_void_p]

    _lib.ane_bridge_write_input.restype = None
    _lib.ane_bridge_write_input.argtypes = [
        ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p, ctypes.c_size_t
    ]

    _lib.ane_bridge_read_output.restype = None
    _lib.ane_bridge_read_output.argtypes = [
        ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p, ctypes.c_size_t
    ]

    _lib.ane_bridge_free.restype = None
    _lib.ane_bridge_free.argtypes = [ctypes.c_void_p]

    _lib.ane_bridge_get_compile_count.restype = ctypes.c_int
    _lib.ane_bridge_get_compile_count.argtypes = []

    _lib.ane_bridge_reset_compile_count.restype = None
    _lib.ane_bridge_reset_compile_count.argtypes = []

    return _lib


class ANEKernel:
    """Wrapper around a compiled ANE kernel handle."""

    def __init__(self, handle, n_inputs, n_outputs, input_sizes, output_sizes):
        self._handle = handle
        self.n_inputs = n_inputs
        self.n_outputs = n_outputs
        self.input_sizes = input_sizes
        self.output_sizes = output_sizes

    def eval(self):
        """Run the kernel on ANE."""
        lib = _load_lib()
        return lib.ane_bridge_eval(self._handle)

    def write_input(self, idx, data):
        """Write data to input tensor."""
        lib = _load_lib()
        raw = np.ascontiguousarray(data)
        lib.ane_bridge_write_input(
            self._handle, idx,
            raw.ctypes.data_as(ctypes.c_void_p),
            raw.nbytes
        )

    def read_output(self, idx, shape, dtype=np.float32):
        """Read data from output tensor."""
        lib = _load_lib()
        out = np.empty(shape, dtype=dtype)
        lib.ane_bridge_read_output(
            self._handle, idx,
            out.ctypes.data_as(ctypes.c_void_p),
            out.nbytes
        )
        return out

    def free(self):
        """Release kernel resources."""
        if self._handle:
            lib = _load_lib()
            lib.ane_bridge_free(self._handle)
            self._handle = None

    def __del__(self):
        self.free()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.free()


class ANEBridge:
    """High-level Python interface to Apple Neural Engine."""

    def __init__(self):
        self._lib = _load_lib()
        self._initialized = False

    def init(self):
        """Initialize the ANE runtime."""
        if self._initialized:
            return
        ret = self._lib.ane_bridge_init()
        if ret != 0:
            raise RuntimeError("Failed to initialize ANE runtime")
        self._initialized = True

    def compile(self, mil_text, weight_data=None, input_sizes=None, output_sizes=None):
        """Compile a MIL program with a single weight file."""
        self.init()

        mil_bytes = mil_text.encode('utf-8')
        n_in = len(input_sizes)
        n_out = len(output_sizes)
        in_arr = (ctypes.c_size_t * n_in)(*input_sizes)
        out_arr = (ctypes.c_size_t * n_out)(*output_sizes)

        if weight_data:
            wdata = (ctypes.c_ubyte * len(weight_data)).from_buffer_copy(weight_data)
            handle = self._lib.ane_bridge_compile(
                mil_bytes, len(mil_bytes),
                ctypes.cast(wdata, ctypes.c_void_p), len(weight_data),
                n_in, in_arr, n_out, out_arr)
        else:
            handle = self._lib.ane_bridge_compile(
                mil_bytes, len(mil_bytes),
                None, 0,
                n_in, in_arr, n_out, out_arr)

        if not handle:
            raise RuntimeError("ANE compile failed")
        return ANEKernel(handle, n_in, n_out, input_sizes, output_sizes)

    def compile_multi_weights(self, mil_text, weights, input_sizes, output_sizes):
        """Compile with multiple named weight files."""
        self.init()

        mil_bytes = mil_text.encode('utf-8')
        n_weights = len(weights)
        names_list = list(weights.keys())
        datas_list = list(weights.values())

        names_arr = (ctypes.c_char_p * n_weights)(
            *[n.encode('utf-8') for n in names_list])

        data_buffers = []
        data_ptrs = (ctypes.c_void_p * n_weights)()
        data_lens = (ctypes.c_size_t * n_weights)()

        for i, d in enumerate(datas_list):
            buf = (ctypes.c_ubyte * len(d)).from_buffer_copy(d)
            data_buffers.append(buf)
            data_ptrs[i] = ctypes.cast(buf, ctypes.c_void_p)
            data_lens[i] = len(d)

        n_in = len(input_sizes)
        n_out = len(output_sizes)
        in_arr = (ctypes.c_size_t * n_in)(*input_sizes)
        out_arr = (ctypes.c_size_t * n_out)(*output_sizes)

        handle = self._lib.ane_bridge_compile_multi_weights(
            mil_bytes, len(mil_bytes),
            names_arr, data_ptrs, data_lens, n_weights,
            n_in, in_arr, n_out, out_arr)

        if not handle:
            raise RuntimeError("ANE compile (multi-weight) failed")
        return ANEKernel(handle, n_in, n_out, input_sizes, output_sizes)

    def build_weight_blob(self, weights, transpose=False):
        """
        Build an ANE weight blob from float32 numpy array.
        Format: 128-byte header + fp16 weight data.
        """
        assert weights.dtype == np.float32
        assert weights.ndim == 2

        rows, cols = weights.shape

        if transpose:
            w_t = np.empty((cols, rows), dtype=np.float16)
            for i in range(rows):
                for j in range(cols):
                    w_t[j, i] = np.float16(weights[i, j])
            fp16_data = w_t.tobytes()
        else:
            fp16_data = weights.astype(np.float16).tobytes()

        wsize = len(fp16_data)
        header = bytearray(128)
        header[0] = 0x01
        header[4] = 0x02
        header[64] = 0xEF
        header[65] = 0xBE
        header[66] = 0xAD
        header[67] = 0xDE
        header[68] = 0x01
        struct.pack_into('<I', header, 72, wsize)
        struct.pack_into('<I', header, 80, 128)

        return bytes(header) + fp16_data

    @property
    def compile_count(self):
        return self._lib.ane_bridge_get_compile_count()

    def reset_compile_count(self):
        self._lib.ane_bridge_reset_compile_count()


# --- MIL Text Generators ---

# MIL header template (uses %% for literal braces, %s for substitution)
_MIL_HEADER = (
    'program(1.3)\n'
    '[buildInfo = dict<string, string>({{"coremlc-component-MIL", "3510.2.1"}, '
    '{"coremlc-version", "3505.4.1"}, {"coremltools-component-milinternal", ""}, '
    '{"coremltools-version", "9.0"}})]\n'
)


def generate_conv_mil(in_ch, out_ch, spatial):
    """
    Generate MIL for conv kernel: y = W @ x  (as 1x1 convolution)
    Input: [1, in_ch, 1, spatial] fp32
    Output: [1, out_ch, 1, spatial] fp32
    Weight: [out_ch, in_ch, 1, 1] fp16 from BLOBFILE at offset 64
    """
    body = (
        '{\n'
        '    func main<ios18>(tensor<fp32, [1, %d, 1, %d]> x) {\n'
        '        string d1 = const()[name = string("d1"), val = string("fp16")];\n'
        '        tensor<fp16, [1, %d, 1, %d]> x16 = cast(dtype = d1, x = x)[name = string("cx")];\n'
        '        tensor<fp16, [%d, %d, 1, 1]> W = const()[name = string("W"), '
        'val = tensor<fp16, [%d, %d, 1, 1]>(BLOBFILE(path = string("@model_path/weights/weight.bin"), offset = uint64(64)))];\n'
        '        string pt = const()[name = string("pt"), val = string("valid")];\n'
        '        tensor<int32, [2]> st = const()[name = string("st"), val = tensor<int32, [2]>([1, 1])];\n'
        '        tensor<int32, [4]> pd = const()[name = string("pd"), val = tensor<int32, [4]>([0, 0, 0, 0])];\n'
        '        tensor<int32, [2]> dl = const()[name = string("dl"), val = tensor<int32, [2]>([1, 1])];\n'
        '        int32 gr = const()[name = string("gr"), val = int32(1)];\n'
        '        tensor<fp16, [1, %d, 1, %d]> y16 = conv(dilations = dl, groups = gr, pad = pd, '
        'pad_type = pt, strides = st, weight = W, x = x16)[name = string("cv")];\n'
        '        string d2 = const()[name = string("d2"), val = string("fp32")];\n'
        '        tensor<fp32, [1, %d, 1, %d]> y = cast(dtype = d2, x = y16)[name = string("co")];\n'
        '    } -> (y);\n}\n'
    ) % (in_ch, spatial, in_ch, spatial, out_ch, in_ch, out_ch, in_ch,
         out_ch, spatial, out_ch, spatial)

    return _MIL_HEADER + body

def generate_dyn_matmul_mil(in_ch, out_ch, spatial):
    """
    Generate MIL for dynamic matmul: y = W @ x
    Input x: [1, in_ch, 1, spatial + out_ch] fp32 (activations + transposed weights)
    Output: [1, out_ch, 1, spatial] fp32
    """
    sp = spatial + out_ch
    body = (
        '{\n'
        '    func main<ios18>(tensor<fp32, [1, %d, 1, %d]> x) {\n'
        '        string d1 = const()[name = string("d1"), val = string("fp16")];\n'
        '        tensor<fp16, [1, %d, 1, %d]> xh = cast(dtype = d1, x = x)[name = string("cin")];\n'
        
        '        tensor<int32, [4]> ba = const()[name=string("ba"), val=tensor<int32, [4]>([0,0,0,0])];\n'
        '        tensor<int32, [4]> sa = const()[name=string("sa"), val=tensor<int32, [4]>([1,%d,1,%d])];\n'
        '        tensor<fp16, [1,%d,1,%d]> act = slice_by_size(x=xh,begin=ba,size=sa)[name=string("act")];\n'
        
        '        tensor<int32, [4]> bw = const()[name=string("bw"), val=tensor<int32, [4]>([0,0,0,%d])];\n'
        '        tensor<int32, [4]> sw = const()[name=string("sw"), val=tensor<int32, [4]>([1,%d,1,%d])];\n'
        '        tensor<fp16, [1,%d,1,%d]> wt = slice_by_size(x=xh,begin=bw,size=sw)[name=string("wt")];\n'
        
        '        tensor<int32, [4]> ra = const()[name=string("ra"), val=tensor<int32, [4]>([1,1,%d,%d])];\n'
        '        tensor<fp16, [1,1,%d,%d]> a2 = reshape(shape=ra,x=act)[name=string("a2")];\n'
        
        '        tensor<int32, [4]> pm = const()[name=string("pm"), val=tensor<int32, [4]>([0,1,3,2])];\n'
        '        tensor<fp16, [1,1,%d,%d]> a3 = transpose(perm=pm,x=a2)[name=string("a3")];\n'
        
        '        tensor<int32, [4]> rw = const()[name=string("rw"), val=tensor<int32, [4]>([1,1,%d,%d])];\n'
        '        tensor<fp16, [1,1,%d,%d]> W = reshape(shape=rw,x=wt)[name=string("W")];\n'
        
        '        bool bF = const()[name=string("bF"), val=bool(false)];\n'
        '        tensor<fp16, [1,1,%d,%d]> yh = matmul(transpose_x=bF,transpose_y=bF,x=a3,y=W)[name=string("yh")];\n'
        
        '        tensor<fp16, [1,1,%d,%d]> yt = transpose(perm=pm,x=yh)[name=string("yt")];\n'
        '        tensor<int32, [4]> ro = const()[name=string("ro"), val=tensor<int32, [4]>([1,%d,1,%d])];\n'
        '        tensor<fp16, [1,%d,1,%d]> yr = reshape(shape=ro,x=yt)[name=string("yr")];\n'
        
        '        string d2 = const()[name = string("d2"), val = string("fp32")];\n'
        '        tensor<fp32, [1, %d, 1, %d]> y = cast(dtype = d2, x = yr)[name = string("co")];\n'
        '    } -> (y);\n}\n'
    ) % (in_ch, sp, 
         in_ch, sp, 
         in_ch, spatial, in_ch, spatial,
         spatial, in_ch, out_ch, in_ch, out_ch,
         in_ch, spatial, in_ch, spatial,
         spatial, in_ch,
         in_ch, out_ch, in_ch, out_ch,
         spatial, out_ch,
         out_ch, spatial,
         out_ch, spatial, out_ch, spatial,
         out_ch, spatial)

    return _MIL_HEADER + body
