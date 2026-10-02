import time
import socket
import logging
from urllib.parse import urlparse
from tensorboard import program

log_path = "logs"

PINNED_CARDS_QUERY = "?pinnedCards=%5B%7B%22plugin%22%3A%22scalars%22%2C%22tag%22%3A%22loss%2Fg%2Ftotal%22%7D%2C%7B%22plugin%22%3A%22scalars%22%2C%22tag%22%3A%22loss%2Fd%2Ftotal%22%7D%2C%7B%22plugin%22%3A%22scalars%22%2C%22tag%22%3A%22loss%2Fg%2Fkl%22%7D%2C%7B%22plugin%22%3A%22scalars%22%2C%22tag%22%3A%22loss%2Fg%2Fmel%22%7D%5D"


def local_network_address():
    # Connecting a UDP socket sends nothing; it only makes the OS pick the
    # interface that routes outward, whose address is the one on the LAN.
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
            s.connect(("192.0.2.1", 80))
            return s.getsockname()[0]
    except OSError:
        return None


def launch_tensorboard_pipeline(lan: bool = False, port: int | None = None):
    logging.getLogger("root").setLevel(logging.WARNING)
    logging.getLogger("tensorboard").setLevel(logging.WARNING)

    # Without --port, TensorBoard takes 6006 or the next free port after it.
    argv = [None, "--logdir", log_path]
    if port is not None:
        argv += ["--port", str(port)]
    if lan:
        argv.append("--bind_all")

    tb = program.TensorBoard()
    tb.configure(argv=argv)
    url = tb.launch()

    if lan:
        address = local_network_address()
        if address is not None:
            url = f"http://{address}:{urlparse(url).port}/"
        print(
            "TensorBoard is open to the local network (no authentication - "
            "anyone on this network can view it)."
        )

    print(f"Access the tensorboard using the following link:\n{url}{PINNED_CARDS_QUERY}")

    while True:
        time.sleep(600)
