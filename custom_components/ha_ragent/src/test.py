from db_backends.mongodb_backend import MongoDbBackend
from embeddings.ollama_embedding import OllamaEmbedding
from models.device import SmartHomeDevice
import time

db_backend = MongoDbBackend('localhost', 'admin', 'mongodb')
embedding = OllamaEmbedding('http://localhost:11434/', db_backend)

devices = [
    SmartHomeDevice(
        id="light_1",
        name="Living Room Light",
        type="light",
        location="living room",
        description="Smart LED bulb compatible with voice control.",
        capabilities=["on", "off", "dim", "color"]
    ),
    SmartHomeDevice(
        id="thermo_1",
        name="Hallway Thermostat",
        type="thermostat",
        location="hallway",
        description="Wi-Fi thermostat with scheduling.",
        capabilities=["set_temperature", "schedule", "off"]
    ),
    SmartHomeDevice(
        id="plug_1",
        name="Bedroom Smart Plug",
        type="plug",
        location="bedroom",
        capabilities=["on", "off"]
    )
]


embedding.embed_devices(devices)
time.sleep(2)
emb = embedding.embed_text("Turn on the light in the living room")
devices = db_backend.load_device_embeddings(emb)
print(devices)