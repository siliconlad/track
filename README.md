# Track

An experiment tracking library built on top of [pybag](https://github.com/siliconlad/pybag).

## Minimal Logger Examples

### 1. Log messages

```python
from track import Logger

logger = Logger("demo", output_dir="run.mcap").open()
logger.info("training started")
logger.warning("learning rate is high")
logger.close()
```

### 2. Log an image (NumPy array)

```python
import numpy as np
from track import Logger

image = np.zeros((64, 64, 3), dtype=np.uint8)
image[:, :, 1] = 255  # green

logger = Logger("demo", output_dir="run.mcap").open()
logger.log_image("camera/rgb", image, format="png")
logger.close()
```

### 3. Log a point cloud (structured array)

```python
import numpy as np
from track import Logger

dtype = np.dtype([("x", "f4"), ("y", "f4"), ("z", "f4")])
points = np.zeros(3, dtype=dtype)
points["x"] = [0.0, 1.0, 0.0]
points["y"] = [0.0, 0.0, 1.0]
points["z"] = [0.0, 0.0, 0.0]

logger = Logger("demo", output_dir="run.mcap").open()
logger.log_pointcloud("lidar", points)
logger.close()
```

### 4. Add metadata and attachments

```python
from track import Logger

logger = Logger("demo", output_dir="run.mcap").open()
logger.add_metadata("experiment", {"name": "baseline", "epoch": "1"})
logger.add_attachment(
    "config.json",
    b'{"batch_size": 32}',
    media_type="application/json",
)
logger.close()
```
