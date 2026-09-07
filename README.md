# Brain MRI image screening demo

This educational project trains a binary image classifier for MRI images organized into `tumor` and `no_tumor` folders, then serves it in a small web UI. **It is not a medical device and cannot diagnose or exclude a brain tumor.** Do not use it for clinical decisions.

## 1. Requirements

Install Python 3.10 or 3.11. In PowerShell, from this folder:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

If PowerShell blocks activation, run this once for the current terminal, then activate again:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
```

## 2. Prepare data

Create this exact layout. `no_tumor` alphabetically maps to label 0, and `tumor` maps to label 1.

```text
data/
  train/
    no_tumor/
      image1.jpg
    tumor/
      image2.jpg
  validation/
    no_tumor/
      image3.jpg
    tumor/
      image4.jpg
```

Use de-identified research/teaching images only, and split data by patient—not merely by image—to prevent misleading evaluation.

## 3. Train

```powershell
python train.py --data-dir data --epochs 10
```

The first run downloads MobileNetV2 ImageNet weights. The model is written to `model/brain_tumor_classifier.keras`.

## 4. Run the web app

```powershell
streamlit run app.py
```

Open the localhost URL Streamlit prints (normally `http://localhost:8501`), then upload a JPG or PNG MRI image.

## Notes

- Start with a held-out patient-level test set before interpreting performance.
- Accuracy alone is insufficient; report sensitivity, specificity, AUC, confidence intervals, and subgroup performance.
- Real clinical deployment requires appropriate validation, regulatory review, privacy/security controls, and clinician oversight.
