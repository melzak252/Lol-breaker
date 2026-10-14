import sys
import betting_app.core.db
from betting_app.core.db import connect

print(f"betting_app.core.db file: {betting_app.core.db.__file__}")
conn = connect()
print(f"Connection class: {type(conn)}")
print(f"Connection class module: {type(conn).__module__}")
print(f"Connection class file: {sys.modules[type(conn).__module__].__file__}")
print(f"Has executemany: {hasattr(conn, 'executemany')}")
