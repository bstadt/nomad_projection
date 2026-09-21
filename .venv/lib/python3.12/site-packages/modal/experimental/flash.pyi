import modal.client
import modal_proto.api_pb2
import subprocess
import typing
import typing_extensions

class _FlashManager:
    def __init__(
        self,
        client: modal.client._Client,
        port: int,
        process: typing.Optional[subprocess.Popen] = None,
        health_check_url: typing.Optional[str] = None,
        startup_timeout: int = 30,
        exit_grace_period: int = 0,
        h2_enabled: bool = False,
        is_server: bool = False,
    ):
        """Initialize self.  See help(type(self)) for accurate signature."""
        ...

    async def is_port_connection_healthy(
        self, process: typing.Optional[subprocess.Popen], timeout: float = 0.5
    ) -> tuple[bool, typing.Optional[Exception]]: ...
    async def _start(self): ...
    async def _start_server_tunnel(self) -> None: ...
    async def _start_flash_registration(self, host: str, port: int) -> None: ...
    async def _deregister(self): ...
    async def _drain_container(self):
        """Background task that checks if we've encountered too many failures and drains the container if so."""
        ...

    async def _wait_for_port_success(self, host: str, port: int) -> bool: ...
    async def _run_heartbeat(self, host: str, port: int): ...
    def get_container_url(self): ...
    async def stop(self): ...
    async def close(self): ...

class FlashManager:
    def __init__(
        self,
        client: modal.client.Client,
        port: int,
        process: typing.Optional[subprocess.Popen] = None,
        health_check_url: typing.Optional[str] = None,
        startup_timeout: int = 30,
        exit_grace_period: int = 0,
        h2_enabled: bool = False,
        is_server: bool = False,
    ): ...

    class __is_port_connection_healthy_spec(typing_extensions.Protocol):
        def __call__(
            self, /, process: typing.Optional[subprocess.Popen], timeout: float = 0.5
        ) -> tuple[bool, typing.Optional[Exception]]: ...
        async def aio(
            self, /, process: typing.Optional[subprocess.Popen], timeout: float = 0.5
        ) -> tuple[bool, typing.Optional[Exception]]: ...

    is_port_connection_healthy: __is_port_connection_healthy_spec

    class ___start_spec(typing_extensions.Protocol):
        def __call__(self, /): ...
        async def aio(self, /): ...

    _start: ___start_spec

    class ___start_server_tunnel_spec(typing_extensions.Protocol):
        def __call__(self, /) -> None: ...
        async def aio(self, /) -> None: ...

    _start_server_tunnel: ___start_server_tunnel_spec

    class ___start_flash_registration_spec(typing_extensions.Protocol):
        def __call__(self, /, host: str, port: int) -> None: ...
        async def aio(self, /, host: str, port: int) -> None: ...

    _start_flash_registration: ___start_flash_registration_spec

    class ___deregister_spec(typing_extensions.Protocol):
        def __call__(self, /): ...
        async def aio(self, /): ...

    _deregister: ___deregister_spec

    class ___drain_container_spec(typing_extensions.Protocol):
        def __call__(self, /):
            """Background task that checks if we've encountered too many failures and drains the container if so."""
            ...

        async def aio(self, /):
            """Background task that checks if we've encountered too many failures and drains the container if so."""
            ...

    _drain_container: ___drain_container_spec

    class ___wait_for_port_success_spec(typing_extensions.Protocol):
        def __call__(self, /, host: str, port: int) -> bool: ...
        async def aio(self, /, host: str, port: int) -> bool: ...

    _wait_for_port_success: ___wait_for_port_success_spec

    class ___run_heartbeat_spec(typing_extensions.Protocol):
        def __call__(self, /, host: str, port: int): ...
        async def aio(self, /, host: str, port: int): ...

    _run_heartbeat: ___run_heartbeat_spec

    def get_container_url(self): ...

    class __stop_spec(typing_extensions.Protocol):
        def __call__(self, /): ...
        async def aio(self, /): ...

    stop: __stop_spec

    class __close_spec(typing_extensions.Protocol):
        def __call__(self, /): ...
        async def aio(self, /): ...

    close: __close_spec

class __flash_forward_spec(typing_extensions.Protocol):
    def __call__(
        self,
        /,
        port: int,
        process: typing.Optional[subprocess.Popen] = None,
        health_check_url: typing.Optional[str] = None,
        startup_timeout: int = 30,
        exit_grace_period: int = 0,
        h2_enabled: bool = False,
        is_server: bool = False,
    ) -> FlashManager:
        """Forward a port to the Modal Flash service, exposing that port as a stable endpoint.
        This is a highly experimental method that can break or be removed at any time without warning.
        Do not use this method unless explicitly instructed to do so by Modal support.
        """
        ...

    async def aio(
        self,
        /,
        port: int,
        process: typing.Optional[subprocess.Popen] = None,
        health_check_url: typing.Optional[str] = None,
        startup_timeout: int = 30,
        exit_grace_period: int = 0,
        h2_enabled: bool = False,
        is_server: bool = False,
    ) -> FlashManager:
        """Forward a port to the Modal Flash service, exposing that port as a stable endpoint.
        This is a highly experimental method that can break or be removed at any time without warning.
        Do not use this method unless explicitly instructed to do so by Modal support.
        """
        ...

flash_forward: __flash_forward_spec

class __flash_get_containers_spec(typing_extensions.Protocol):
    def __call__(self, /, app_name: str, cls_name: str) -> list[typing.Any]:
        """Return a list of flash containers for a deployed Flash service.

        Each entry exposes `task_id`, `host`, and `port` attributes.

        This is a highly experimental method that can break or be removed at any time without warning.
        Do not use this method unless explicitly instructed to do so by Modal support.
        """
        ...

    async def aio(self, /, app_name: str, cls_name: str) -> list[typing.Any]:
        """Return a list of flash containers for a deployed Flash service.

        Each entry exposes `task_id`, `host`, and `port` attributes.

        This is a highly experimental method that can break or be removed at any time without warning.
        Do not use this method unless explicitly instructed to do so by Modal support.
        """
        ...

flash_get_containers: __flash_get_containers_spec

def _http_server(
    port: typing.Optional[int] = None,
    *,
    proxy_regions: list[str] = [],
    startup_timeout: int = 30,
    exit_grace_period: typing.Optional[int] = None,
    h2_enabled: bool = False,
):
    """Decorator for Flash-enabled HTTP servers on Modal classes.

    Args:
        port: The local port to forward to the HTTP server.
        proxy_regions: The regions to proxy the HTTP server to.
        startup_timeout: The maximum time to wait for the HTTP server to start.
        exit_grace_period: The time to wait for the HTTP server to exit gracefully.
    """
    ...

def http_server(
    port: typing.Optional[int] = None,
    *,
    proxy_regions: list[str] = [],
    startup_timeout: int = 30,
    exit_grace_period: typing.Optional[int] = None,
    h2_enabled: bool = False,
):
    """Decorator for Flash-enabled HTTP servers on Modal classes.

    Args:
        port: The local port to forward to the HTTP server.
        proxy_regions: The regions to proxy the HTTP server to.
        startup_timeout: The maximum time to wait for the HTTP server to start.
        exit_grace_period: The time to wait for the HTTP server to exit gracefully.
    """
    ...

class _FlashContainerEntry:
    """A class that manages the lifecycle of Flash manager for Flash containers.

    It is intentional that stop() runs before exit handlers and close().
    This ensures the container is deregistered first, preventing new requests from being routed to it
    while exit handlers execute and the exit grace period elapses, before finally closing the tunnel.
    """

    flash_manager: typing.Optional[FlashManager]

    def __init__(self, http_config: modal_proto.api_pb2.HTTPConfig, is_server: bool = False):
        """Initialize self.  See help(type(self)) for accurate signature."""
        ...

    def enter(self): ...
    def stop(self): ...
    def close(self): ...
