try: from .D2_common import NestedUNet2D
except ImportError: from D2_common import NestedUNet2D
class NestedUNet2D(NestedUNet2D): pass
