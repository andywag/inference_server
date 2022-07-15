
import requests
import time

tic = time.time()
r = requests.post("http://120.92.42.245:12501/v1/ruDALLE/generate",json={"text":"dark matter in the universe"})
print("Found", time.time() - tic)
print("CC", r)
result = r.json()
print("A", result.keys())
b64_string = result['b64img']

import io, base64, cv2
import numpy as np

#img = cv2.imread(io.BytesIO(base64.b64decode(b64_string)))
#cv2_img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

byte = base64.b64decode(b64_string)
jpg_as_np = np.frombuffer(byte, dtype=np.uint8)
img = cv2.imdecode(jpg_as_np, flags=1)
print("Found", time.time() - tic)

#print(img)
#print(result)