"""Protected conversion entry: the acp_supermimic startup hook queues this module.

The historical entry name is required by the host's conversion memory guard.
This repository adapts JWP BVH and upstream Tiangong LAFAN NPZ references.
"""
import sys
if "--lafan-segment-from" in sys.argv:
    from tools.lafan_segments import main
elif "--lafan-source" in sys.argv:
    from tools.lafan_pipeline import main
elif "--repair-from" in sys.argv:
    from tools.jwp_repair import main
else:
    from tools.jwp_pipeline import main

if __name__ == "__main__":
    main()
