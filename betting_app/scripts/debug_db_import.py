import sys
import betting_app.core.db
from betting_app.core.db import _ConnectionWrapper

print(f"betting_app.core.db file: {betting_app.core.db.__file__}")
print(f"_ConnectionWrapper: {_ConnectionWrapper}")
print(f"dir(_ConnectionWrapper): {dir(_ConnectionWrapper)}")
print(f"sys.modules['betting_app.core.db']: {sys.modules.get('betting_app.core.db')}")
