import os
MODEL_DIR = os.getenv("MODEL_DIR", "../models")
MODEL_FILE = os.getenv("MODEL_FILE", "model_2.bin")
VOCAB_FILE = os.getenv("VOCAB_FILE", "bert-base-uncased-vocab.txt")
BATCH_SIZE = int(os.getenv("BATCH_SIZE", "16"))
USE_GPU = os.getenv("USE_GPU", "YES") == "YES"
LISTEN_PORT = int(os.getenv("LISTEN_PORT", "5000"))

MODEL_PATH = os.path.join(MODEL_DIR, MODEL_FILE)
VOCAB_PATH = os.path.join(MODEL_DIR, VOCAB_FILE)

print(f"SDG WEBSERVICE STARTING")
print(f"Model dir: {MODEL_DIR}")
print(f"Using model {MODEL_PATH}, vocab {VOCAB_PATH}")
print(f"For inference, use GPU: {USE_GPU}, batch size {BATCH_SIZE}")
print(f"Going to listen op port {LISTEN_PORT}")