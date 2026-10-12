"""NVRTC binary and cache contracts, independent of installed GPU hardware."""
import ctypes as ct
import hashlib

import pytest

from dgfl.crypto import gpu


class Function:
    def __init__(self, callback):
        self.callback = callback

    def __call__(self, *arguments):
        return self.callback(*arguments)


def write(pointer, kind, value):
    ct.cast(pointer, ct.POINTER(kind)).contents.value = value


class Compiler:
    binary = b'\x7fELF\x02\x01\x01\0compiled\0cuda-binary'

    def __init__(self):
        self.options = []
        self.destroyed = 0
        self.result_size = len(self.binary)
        self.compile_status = 0
        self.functions = {}

    def __getattr__(self, name):
        callback = getattr(type(self), '_' + name, None)
        if callback is None:
            raise AttributeError(name)
        return self.functions.setdefault(name, Function(lambda *args: callback(self, *args)))

    def _nvrtcVersion(self, major, minor):
        write(major, ct.c_int, 12)
        write(minor, ct.c_int, 9)
        return 0

    def _nvrtcCreateProgram(self, program, *_arguments):
        write(program, ct.c_void_p, 123)
        return 0

    def _nvrtcCompileProgram(self, _program, count, options):
        self.options.append([options[index] for index in range(count)])
        return self.compile_status

    def _nvrtcGetCUBINSize(self, _program, length):
        write(length, ct.c_size_t, self.result_size)
        return 0

    def _nvrtcGetCUBIN(self, _program, output):
        ct.memmove(output, self.binary, len(self.binary))
        return 0

    def _nvrtcGetPTXSize(self, _program, length):
        write(length, ct.c_size_t, len(b'PTX output\0'))
        return 0

    def _nvrtcGetPTX(self, _program, output):
        ct.memmove(output, b'PTX output\0', len(b'PTX output\0'))
        return 0

    def _nvrtcGetProgramLogSize(self, _program, length):
        write(length, ct.c_size_t, len(b'compiler failed\0'))
        return 0

    def _nvrtcGetProgramLog(self, _program, output):
        ct.memmove(output, b'compiler failed\0', len(b'compiler failed\0'))
        return 0

    def _nvrtcDestroyProgram(self, program):
        self.destroyed += 1
        write(program, ct.c_void_p, None)
        return 0


@pytest.fixture
def compiler(tmp_path, monkeypatch):
    source = tmp_path / 'src' / 'dgfl' / 'crypto' / 'gpu.py'
    headers = source.with_name('cuda')
    headers.mkdir(parents=True)
    for name in ('field.cuh', 'g1.cuh', 'gt.cuh'):
        (headers / name).write_bytes(b'// ' + name.encode())
    monkeypatch.setattr(gpu, '__file__', str(source))
    compiler = Compiler()
    monkeypatch.setattr(gpu, '_library', lambda _path: compiler)
    device = gpu.CudaRuntime.__new__(gpu.CudaRuntime)
    device.nvrtc_path = 'test-nvrtc'
    device.info = {}
    return device, compiler, tmp_path / 'runtime' / 'gpu-cache'


def test_compile_targets_device_machine_code_and_keeps_binary_nuls_in_cache(compiler):
    device, nvrtc, cache = compiler
    assert device.compile(8, 9) == Compiler.binary
    assert b'--gpu-architecture=sm_89' in nvrtc.options[0]
    assert not any(b'compute_' in option for option in nvrtc.options[0])
    paths = list(cache.glob('*.cubin'))
    assert len(paths) == 1
    assert paths[0].read_bytes() == Compiler.binary
    assert device.info['cubin_sha256'] == hashlib.sha256(Compiler.binary).hexdigest()
    nvrtc.compile_status = 6  # A valid binary cache must work without recompiling.
    assert device.compile(8, 9) == Compiler.binary
    assert nvrtc.destroyed == 1


def test_legacy_ptx_cache_cannot_be_used_as_a_native_binary(compiler):
    device, _nvrtc, cache = compiler
    contents = [b'// field.cuh', b'// g1.cuh', b'// gt.cuh']
    old_options = [b'--std=c++14', b'--gpu-architecture=compute_89',
                   b'--device-as-default-execution-space', b'--Ofast-compile=mid']
    digest = hashlib.sha256(b'\0'.join(contents + old_options)).hexdigest()
    cache.mkdir(parents=True)
    legacy = cache / f'{digest}-sm89-nvrtc12.9.ptx'
    legacy.write_bytes(b'legacy PTX\0')
    assert device.compile(8, 9) == Compiler.binary
    assert legacy.read_bytes() == b'legacy PTX\0'


def test_cubin_cache_is_specific_to_actual_gpu_architecture(compiler):
    device, nvrtc, cache = compiler
    assert device.compile(8, 9) == Compiler.binary
    nvrtc.binary = b'\x7fELF\0different-sm-binary'
    nvrtc.result_size = len(nvrtc.binary)
    assert device.compile(8, 6) == b'\x7fELF\0different-sm-binary'
    assert b'--gpu-architecture=sm_86' in nvrtc.options[-1]
    assert len(list(cache.glob('*.cubin'))) == 2


@pytest.mark.parametrize('size', [0, 64 * 1024 * 1024])
def test_invalid_binary_size_fails_before_allocating_or_caching(compiler, size):
    device, nvrtc, cache = compiler
    nvrtc.result_size = size
    with pytest.raises(gpu.GPUUnavailable, match='CUBIN'):
        device.compile(8, 9)
    assert nvrtc.destroyed == 1
    assert not list(cache.glob('*.cubin'))


def test_compile_failure_is_explicit_and_releases_the_program(compiler):
    device, nvrtc, cache = compiler
    nvrtc.compile_status = 6
    with pytest.raises(gpu.GPUUnavailable, match='compiler failed'):
        device.compile(8, 9)
    assert nvrtc.destroyed == 1
    assert not list(cache.glob('*.cubin'))
