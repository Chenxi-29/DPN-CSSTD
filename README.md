# Spatial-Spectral Contrastive Self-Supervised Learning with Dual Path Networks for Hyperspectral Target Detection (DPN-CSSTD)

## Files
- `main.py`
- `model.py`
- `dataset.py`
- `DPNCSSTD_pretrain.py`
- `DPNCSSTD_finetune.py`
- `DPNCSSTD_detection.py`
---

## Requirement
- Python `3.9.13`
- PyTorch `1.13.0`
---

## Run
### The specific process is as follows:

1. **Pretraining**  
   - **Remove the `DPNCSSTD_pretrain(args)` comments in `main.py`.**
   - **Run `main.py` to retrain the pretraining model.**  
   - A well-pretrained model on the Sandiego(400*400) dataset is provided; you can directly fine-tune and test it.


2. **Fine-tuning and Detection**  
   - **Run `main.py` to fine-tune the target detection model and complete the final detection.** 
   - If you are using a new dataset, adjust `test_image` for you need.
   - The new dataset should contain `data`, `TargetPrioriSpectra`, and `map` values.  
     If they do not match, you can modify them in the `loaddata` function in `dataset.py`.

**To Run the Main File**  
Execute the following command to start the main process:
```bash
python main.py
```

## Cite
### If you find this code useful for your research, please consider citing, thank you!
```bibtex
@ARTICLE{10504845,
  author={Chen, Xi and Zhang, Yuxiang and Dong, Yanni and Du, Bo},
  journal={IEEE Transactions on Geoscience and Remote Sensing}, 
  title={Spatial-Spectral Contrastive Self-Supervised Learning With Dual Path Networks for Hyperspectral Target Detection}, 
  year={2024},
  volume={62},
  number={},
  pages={1-12},
  keywords={Task analysis;Feature extraction;Detectors;Adaptation models;Training;Convolution;Self-supervised learning;Contrastive learning;convolution neural network (CNN);hyperspectral target detection (HTD);self-supervised learning (SSL);simple linear iterative clustering (SLIC) segmentation},
  doi={10.1109/TGRS.2024.3390946}}
```
