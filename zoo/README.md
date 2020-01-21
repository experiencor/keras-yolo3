## Notes

#### On License Plate Model

To do training with [config_license_plates.json](config_license_plates.json), you need 2 models for the keras implementation to work:
1. The pretrained license plates model called `pretrained_lp.h5` which was taken from https://github.com/anuj200199/licenseplatedetection.
1. The backend `backend.h5` model absolutely required for this project (look into any of the models' links in the main README).

If you wish to just predict on a new input, then you'll need `backend.h5` and `license_plate.h5` models.