import time

import comfy.utils
import numpy as np
from PIL import Image

# You can use this node to save full size images through the websocket, the
# images will be sent in exactly the same format as the image previews: as
# binary images on the websocket with a 8 byte header indicating the type
# of binary message (first 4 bytes) and the image format (next 4 bytes).

# Note that no metadata will be put in the images saved with this node.


class SaveImageWebsocket:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "images": ("IMAGE",),
            }
        }

    RETURN_TYPES = ()
    FUNCTION = "save_images"

    OUTPUT_NODE = True

    CATEGORY = "api/image"

    def save_images(self, images):
        pbar = comfy.utils.ProgressBar(images.shape[0])
        for step, image in enumerate(images):
            i = 255.0 * image.cpu().numpy()
            img = Image.fromarray(np.clip(i, 0, 255).astype(np.uint8))
            pbar.update_absolute(step, images.shape[0], ("PNG", img, None))

        return {}

    @classmethod
    def IS_CHANGED(s, images):  # noqa: ARG003 - ComfyUI passes the node's inputs
        return time.time()


NODE_CLASS_MAPPINGS = {
    "SaveImageWebsocket": SaveImageWebsocket,
}
