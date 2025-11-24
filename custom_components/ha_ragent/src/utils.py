import socket
import logging

_logger = logging.getLogger(__name__)

def remove_thinking_block(text: str):
    pass

def get_value(value: object, default: object) -> object:
    """Returns the value when not null, otherwise the default parameter."""
    return value if value else default

def is_valid_host(host: str) -> bool:
    """Checks if the provided hostname is valid."""
    try:
        socket.gethostbyname(host)
        return True
    except socket.gaierror:
        return False
    
def format_url(hostname: str, port: str, ssl: bool, path: str) -> str:
    return f"{'https' if ssl else 'http'}://{hostname}{ ':' + str(port) if port else ''}{path}"

def try_parse_int(value: str, default: int = 0) -> int:
    try:
        return int(value)
    except ValueError:
        return default