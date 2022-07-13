
import requests
r = requests.post('http://120.92.14.211:12501/v1/ruDALLE/generate',{'text':'I am the master'})
r.json()
result = r.json()
print("A", result.keys())
b64_string = result['b64img']

import io, base64, cv2
import numpy as np

#img = cv2.imread(io.BytesIO(base64.b64decode(b64_string)))
#cv2_img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

bytess = base64.b64decode(b64_string)
jpg_as_np = np.frombuffer(bytess, dtype=np.uint8)
print(jpg_as_np.shape)
img = cv2.imdecode(jpg_as_np, flags=1)
print(img)