# Three-dimensional-tumor-reconstruction-algorithm

This repository contains a Python pipeline for reconstructing 3D tumor volumes from 2D ultrasound segmentation masks in COCO format.  
The workflow includes mask extraction, slice ordering, 3D volume assembly, interpolation, surface reconstruction and tumor volume estimation.

# Files structure
- folder `images/` - Series of ultrasound image slices organized per mouse, day and imaging plane (ax-mean axial, sg-means sagittal)
- file `Three-dimensional_tumor_reconstruction.ipynb` - Jupyter notebook for 3D tumor volume reconstruction using binary masks and Marching Cubes algorithm
- file `reconstruction.py` - Python script version of the reconstruction workflow
- file `requirements.txt` and `environment.yml` - Dependency lists for pip and conda respectively
- file `parameters.xlsx` - Excel file containing parameters extracted from VevoLab RAW metadata

# Requirements
All dependencies are listed in `requirements.txt` and `environment.yml`.

Install with pip:
```bash
pip install -r requirements.txt
```
or use conda:
```bash
conda env create -f environment.yml
conda activate tumor-reconstruction
```
You can run either Jupyter Notebook or Python script.

# Running the Reconstruction
To run the `reconstruction.py` script on a specific imaging series, update the input path in the script. Edit the line:
```python
with open('images/name_of_the_series/_annotations.coco.json', 'r') as file:
    data = json.load(file)
```
Replace `name_of_the_series` with the appropriate folder name (e.g., `1339_04.03_ax`) corresponding to chosen dataset.

The script will load the annotations file, generate binary masks than perform linear interpolation and the 3D volume reconstruction using the Marching Cubes algorithm and PyVista visualization.

To perform linear interpolation I modified the script from: [https://github.com/bnsreenu/python_for_microscopists/blob/master/Tips_and_Tricks_50_interpolate_images_in_a_stack.ipynb]

### Physical imaging parameters
Most imaging series share the same parameters:
- `STEP_SIZE_MM = 0.1016` 
- `BMODE_DEPTH_MM = 13.7360715866089`
- `BMODE_WIDTH_MM = 14.0524999238551`

All image stacks have a resolution of **height = 1204** and **width = 928** pixels.  
Because of this, the morphological dilation step uses **16 iterations**, which corresponds to the spatial calibration of the imaging system.  
If the image dimensions or scale change, the required number of dilation iterations should be adjusted proportionally.

The relevant code segment:

```python
for _ in range(16):
    interpolated_data_edges = binary_dilation(
        interpolated_data_edges,
        structure=np.ones((3, 3, 3))
    )
```
However, there are exceptions.
The exact values were extracted from VevoLab RAW XML metadata.
A summary of parameter values is provided in the Excel file `parameters.xlsx`.

### Data structure
Five mouse models were used: 1338, 1339, 1340, 1341, 1342. Each mouse was imaged multiple times in two planes: axial and sagittal.

Each session has its own folder containing:
- B-mode image slices
- COCO annotation file (_annotations.coco.json)
