from . import core
from . import datasets
from . import dev
from . import gen_utils
from . import net_trainer
from . import nets
from . import stat_utils
from . import test_livenet
from . import utils
from . import visual_utils


# this piece of code intended to assist work with ipython notebooks
# it reloads all libs imported above
# so if some libs where updated, change will be effective for notebook
# use with caution - if one change class definition dynamically,
# it may lead to undefined behaviour when interacts with older already created class' instances
# but in most cases author found this reload works and very helpful
import re
import importlib
with open(__file__) as f:
    content = f.read()
imports = re.findall(r"^from\s+\.\s+import\s+\w+\s+", content, flags=re.MULTILINE)
for entry in imports:
    entry = ' '.join(entry.split())  # normalise spaces
    package_name = entry[14:]  # omit 'from . import ' at beginning
    exec(f"importlib.reload({package_name})")
