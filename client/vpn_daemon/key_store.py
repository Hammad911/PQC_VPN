# client/vpn_daemon/key_store.py
"""
Secure in-memory key handling with wiping on cleanup.
Never writes private keys to disk.
"""
import ctypes
import keyring

class SecureKeyStore:
    def __init__(self):
        self._keys = {}   # in-memory only

    def store(self, name: str, key_bytes: bytes):
        self._keys[name] = bytearray(key_bytes)

    def get(self, name: str) -> bytes:
        return bytes(self._keys.get(name, b""))

    def wipe(self, name: str):
        """Securely overwrite key in memory before deleting."""
        if name in self._keys:
            buf = self._keys[name]
            ctypes.memset(
                (ctypes.c_char * len(buf)).from_buffer(buf),
                0, len(buf)
            )
            del self._keys[name]

    def wipe_all(self):
        for name in list(self._keys.keys()):
            self.wipe(name)