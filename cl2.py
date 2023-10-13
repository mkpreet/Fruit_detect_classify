import socket
import io
from PIL import ImageTk, Image

BUF_LEN = 1024
FORMAT = "utf-8"
SERVER = '192.168.42.79'
PORT = 1244
with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
    stream = None
    ADDRESS = (SERVER, PORT)
    s.connect(ADDRESS)
    s.send('PIC'.encode(FORMAT))
    stream = io.BytesIO()
    chunk = s.recv(BUF_LEN)
    while chunk:
        print(f"Received {len(chunk)} bytes.")
        stream.write(chunk)
        chunk = s.recv(BUF_LEN)
    stream.seek(0)
    image = Image.open(stream).convert('RGB')
    image.show()
    image.save("image_received.png")



