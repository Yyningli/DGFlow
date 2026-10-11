"""Lazy same-origin TensorBoard ASGI application, with no listening socket."""
import threading
import time
from importlib import resources

from starlette.concurrency import run_in_threadpool
from starlette.middleware.wsgi import WSGIMiddleware
from starlette.responses import JSONResponse, Response

from dgfl.experiments.tensorboard import URL, is_key_scalar
from dgfl.services.tensorboard_metrics import KeyMetricDataProvider

_LOCALE_ROOT = URL + 'dgflow-i18n/'
_LOCALE_ASSETS = {'zh-CN.js': 'text/javascript; charset=utf-8',
                  'zh-CN.css': 'text/css; charset=utf-8'}
_HTML_LIMIT = 2 * 1024 * 1024
_LOCALE_MARKER = b'data-dgflow-locale="zh-CN"'
_LOCALE_HTML = (
    '<link rel="stylesheet" href="' + _LOCALE_ROOT + 'zh-CN.css" data-dgflow-locale="zh-CN">'
    '<script defer src="' + _LOCALE_ROOT + 'zh-CN.js" data-dgflow-locale="zh-CN"></script>'
).encode('ascii')


def _locale_bytes(name):
    # name comes exclusively from _LOCALE_ASSETS; requests cannot read arbitrary
    # package/workspace files. importlib.resources also works in installed wheels.
    return resources.files('dgfl.services').joinpath('assets', name).read_bytes()


async def _localized_html(application, scope, receive, send):
    """Adapt the homepage only; never rewrite native JS, JSON, or scalar data."""
    if scope.get('method') != 'GET' or scope.get('path') not in (URL, URL + 'index.html'):
        await application(scope, receive, send)
        return
    start = None
    chunks = []
    size = 0
    buffering = False

    async def localized_send(message):
        nonlocal start, size, buffering
        if message['type'] == 'http.response.start':
            headers = {key.lower(): value.lower() for key, value in message.get('headers', [])}
            buffering = (message['status'] == 200 and
                         headers.get(b'content-type', b'').split(b';')[0] == b'text/html' and
                         headers.get(b'content-encoding', b'identity') == b'identity')
            if buffering:
                start = message
            else:
                await send(message)
            return
        if message['type'] != 'http.response.body' or not buffering:
            await send(message)
            return
        body = message.get('body', b'')
        chunks.append(body)
        size += len(body)
        if size > _HTML_LIMIT:
            # Bounded memory even if an unexpected upstream page is streaming.
            await send(start)
            for index, chunk in enumerate(chunks):
                await send({'type': 'http.response.body', 'body': chunk,
                            'more_body': index < len(chunks) - 1 or message.get('more_body', False)})
            chunks.clear()
            buffering = False
        elif not message.get('more_body', False):
            original = b''.join(chunks)
            chunks.clear()
            body = original
            if _LOCALE_MARKER not in body:
                index = body.lower().find(b'</head>')
                if index < 0:
                    index = body.lower().find(b'</body>')
                if index >= 0:
                    body = body[:index] + _LOCALE_HTML + body[index:]
            if body != original:
                start = dict(start, headers=[(key, value) for key, value in start.get('headers', [])
                                            if key.lower() not in (b'content-length', b'etag', b'content-md5')]
                             + [(b'content-length', str(len(body)).encode('ascii'))])
            await send(start)
            await send({'type': 'http.response.body', 'body': body, 'more_body': False})
            buffering = False

    await application(scope, receive, localized_send)


class EmbeddedTensorBoard:
    def __init__(self, logs, *, reload_interval=2.):
        self.logs = logs
        self._lock = threading.RLock()
        self._application = None
        self._ingester = None
        self._last_reload = 0.
        self._reload_interval = max(0., float(reload_interval))
        self._error = None

    def status(self):
        result = self.logs.status()
        result['initialized'] = self._application is not None
        if self._error:
            result['last_error'] = self._error
        return result

    def _prepare(self):
        with self._lock:
            try:
                self.logs.validate_tree()
                if self._application is None:
                    from tensorboard import program
                    from tensorboard.backend import application
                    from tensorboard.backend.event_processing.data_ingester import LocalDataIngester
                    from tensorboard.plugins.core.core_plugin import CorePluginLoader
                    from tensorboard.plugins.hparams.hparams_plugin import HParamsPlugin
                    from tensorboard.plugins.metrics.metrics_plugin import MetricsPlugin
                    from tensorboard.plugins.scalar.scalars_plugin import ScalarsPlugin
                    # MetricsPlugin is the official Angular "timeseries"
                    # frontend; ScalarsPlugin alone selects the old Polymer
                    # dashboard. Its run selector also reads the HParams APIs.
                    # Keep scalar APIs for history/export compatibility and
                    # avoid arbitrary installed plugin entrypoints.
                    tensorboard = program.TensorBoard(plugins=[
                        CorePluginLoader(), MetricsPlugin, ScalarsPlugin, HParamsPlugin])
                    tensorboard.configure(argv=['tensorboard'], logdir=str(self.logs.logdir),
                                          path_prefix=URL.rstrip('/'), load_fast='false',
                                          reload_interval=0, reload_task='blocking',
                                          max_reload_threads=1, purge_orphaned_data=False)
                    self._ingester = LocalDataIngester(tensorboard.flags)
                    self._ingester.start()
                    wsgi = application.TensorBoardWSGIApp(
                        tensorboard.flags, tensorboard.plugin_loaders,
                        KeyMetricDataProvider(self._ingester.data_provider, is_key_scalar),
                        tensorboard.assets_zip_provider, self._ingester.deprecated_multiplexer)
                    self._application = WSGIMiddleware(wsgi)
                    self._last_reload = time.monotonic()
                elif time.monotonic() - self._last_reload >= self._reload_interval:
                    # Blocking, one-shot reloads execute in a request worker. TB's
                    # permanent daemon reloader has no public stop API, so never
                    # start it in the controller process.
                    self._ingester.start()
                    self._last_reload = time.monotonic()
                self._error = None
                return self._application
            except Exception as exc:
                self._error = ('TensorBoard 页面暂不可用：' + type(exc).__name__ + ': ' + str(exc))
                self._error = self._error.replace(str(self.logs.runtime), '[runtime]')[:400]
                return None

    async def __call__(self, scope, receive, send):
        if scope['type'] != 'http':
            return
        path = scope.get('path', '')
        if path.startswith(_LOCALE_ROOT):
            name = path[len(_LOCALE_ROOT):]
            if scope.get('method') not in ('GET', 'HEAD'):
                response = Response(status_code=405, headers={'Allow': 'GET, HEAD'})
            elif name == 'labels.json':
                from dgfl.experiments.tensorboard import scalar_label_catalog
                response = JSONResponse(scalar_label_catalog(), headers={'Cache-Control': 'no-cache'})
            elif name in _LOCALE_ASSETS:
                response = Response(_locale_bytes(name), media_type=_LOCALE_ASSETS[name],
                                    headers={'Cache-Control': 'no-cache',
                                             'X-Content-Type-Options': 'nosniff'})
            else:
                response = Response(status_code=404)
            if scope.get('method') == 'HEAD':
                response.body = b''
            await response(scope, receive, send)
            return
        application = await run_in_threadpool(self._prepare)
        if application is None:
            result = {**self.status(), 'available': False, 'reason': self._error}
            await JSONResponse(result, status_code=503)(scope, receive, send)
            return
        # Starlette removes a Mount's root_path when converting PATH_INFO.
        # TensorBoard's path_prefix needs the full /tensorboard/... path so that
        # its HTML base href and native API/static asset routes agree.
        nested = dict(scope, root_path='')
        await _localized_html(application, nested, receive, send)

    def close(self):
        with self._lock:
            self._application = None
            self._ingester = None
            self._last_reload = 0.
