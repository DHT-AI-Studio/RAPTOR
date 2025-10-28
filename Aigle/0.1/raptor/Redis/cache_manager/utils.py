import hashlib
import pickle
from typing import Any

def hash_query(data: Any) -> str:
    """
    Return a SHA256 hash of the given data.

    The data is first pickled into a bytes object, then hashed.

    :param data: The data to hash
    :return: A SHA256 hash of the data, as a hexadecimal string
    """
    binary = pickle.dumps(data)
    return hashlib.sha256(binary).hexdigest()