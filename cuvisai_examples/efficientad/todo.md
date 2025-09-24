# data.
- put data under HF
- instead of ok-nok etc put data under meaning ful subfolders, e.g. defects, no defects. simplify the dataloader
- L320 of train.py in effcientad is trying to get quantiles only on good samples. what are charfacteristics of these samples?
- The autoencoder is not used in the combined anomaly_map. if not used could we skip it in the model declaration too?
- teacher is supposed to get augmented image in the pass for training AE but it is getting the same image.