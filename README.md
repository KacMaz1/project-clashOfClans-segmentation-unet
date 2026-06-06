# Training process
full process in the U-net technical raport clash of clans.docx

# Srodek Segmenter
<img width="1448" height="728" alt="image" src="https://github.com/user-attachments/assets/770110a4-bfb4-4b49-b9ea-66f3073e6157" />

Open-source model for segmenting the clash of clans base area named `srodek` on screenshots.

Test result: IoU `0.9832`, Dice `0.9915`.

small dataset (only 10 test images)

## What This Includes

The checkpoint file is only the neural network model. By itself, the model returns a raw segmentation prediction for each pixel.

The full `srodek_segmenter.py` script adds the remaining steps needed for practical use:

- image resize,
- model prediction,
- thresholding,
- postprocessing,
- mask overlay generation.

So the recommended way to use this repository is through `srodek_segmenter.py`, not by loading the checkpoint alone.

## Output

Internally, the neural network produces one value per pixel. After sigmoid, this becomes a probability map that says how likely each pixel is to belong to `srodek`.

The segmenter converts that probability map into:

- `mask`: a binary image where `1` means `srodek` and `0` means background,
- `overlay`: the input image with the predicted mask drawn on top.

The segmenter always resizes the input image to the model resolution before prediction: 1024 x 576

## Using The Checkpoint Directly

The checkpoint also stores the recommended threshold:

```python
checkpoint = torch.load("models/srodek_resnet34_unet_final.pth", map_location="cpu")
threshold = checkpoint["best_threshold"]  # 0.465
```

If you load the model manually, apply sigmoid first and then use that threshold:

```python
logits = model(image_tensor)
probability = torch.sigmoid(logits)
mask = probability > threshold
```

The checkpoint contains the model weights and this threshold value, but it does not run the full postprocessing by itself. The ready-to-use postprocessing is implemented in `srodek_segmenter.py`.

## Postprocessing

After thresholding, `srodek_segmenter.py` cleans the binary mask with the same postprocessing used for the final result:

- morphological closing,
- filling the largest external contour.

This step helps make the output mask more solid and removes small gaps or fragmented areas. If you use only the checkpoint manually, you will get the raw thresholded mask unless you also implement this postprocessing step.
more details in word doc.

## Expected Input

The model was trained on the FULL SCREEN standard base view shown after entering a battle. It should work best on screenshots taken from that same view.

Small framing differences should usually be fine, because the training data included light zoom and position variation. Strong zoom-out, strong zoom-in, unusual camera framing, different game screens, or heavy UI changes may reduce the quality of the mask.

## Installation

```powershell
pip install -r requirements.txt
```

## GUI

```powershell
python srodek_segmenter.py --gui
```

Choose an image in the file picker. The window will show the resized input image and the output image with the predicted mask overlay.

## Command Line

```powershell
python srodek_segmenter.py --input input.jpg --output output.png
```

The output file is saved as an image with the predicted mask overlay.

## Python

```python
from srodek_segmenter import SrodekSegmenter

segmenter = SrodekSegmenter()
mask, overlay, resized_image = segmenter.predict_file("input.jpg")
segmenter.save_overlay("input.jpg", "output.png")
```

## Files

The model checkpoint is included in:

```text
models/srodek_resnet34_unet_final.pth
```

Full project details are described in a separate Word document.
