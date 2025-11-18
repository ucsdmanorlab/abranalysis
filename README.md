# Auditory Brainstem Response Analyzer (ABRA)
ABRA is a web application created using Streamlit which allows users to batch-upload and analyze ABR data. Read the preprint [here](https://www.biorxiv.org/content/10.1101/2024.06.20.599815v2)!

It can either be run on the [web](https://abra.ucsd.edu) OR locally (instructions below).

If using the web interface, [click here](#usage) or scroll down to Usage instructions below.


## Local installation

Quick install instructions are below. For more detail see the full [ABRA Instructions](https://github.com/ucsdmanorlab/abranalysis/blob/main/ABRA_Instructions.pdf).

(Recommended) Install in a separate conda environment:
1. If conda is not installed, first install [miniconda](https://www.anaconda.com/docs/getting-started/miniconda/install)

2. Open “Anaconda Prompt (miniconda3)” to run miniconda after install

3. Create and activate a conda environment for the abra installation:
```
conda create -y -n abra python=3.12 git
conda activate abra
```
4. Clone this github repo and install:
```
git clone https://github.com/ucsdmanorlab/abranalysis.git
```
Type yes if prompted, then:
```
cd abranalysis
pip install -r requirements.txt
```
5. Launch the streamlit app:
```
streamlit run ABRA.py
```
**On future uses,** you only need to open the Anaconda Prompt, then run the following to activate the ABRA environment, move to the ABRA installation directory, and launch the streamlit app. 
```
conda activate abra
cd abranalysis
streamlit run ABRA.py
```
As new updates are released in the future, you can always pull the latest version by running `git remote update` from inside your abranalysis directory. 

## API-only Installation

If you only want to use ABRA's analysis functions programmatically without the web interface:

```
conda create -y -n abra python=3.12 
conda activate abra
pip install -r requirements-api.txt
```

See notebooks for usage. 

## Usage

1. Upload a file (Tucker Davis .arf, EPFL .tsv or .asc, or standardized .csv). If you're uploading a .csv, there must be a `Level(dB)` column and a `Freq(Hz)` column, and the vector of data points ends each corresponding row.
![Screen Recording of drag-and-drop file upload to ABRA](https://github.com/user-attachments/assets/c093c6cc-5372-4f21-abe4-909eb67d3322)

Click the thumbnail below to see an example of the .csv format:
<p align="center">
<img width="200" alt="image" src="https://github.com/abhierra2/ucsdpracticum/assets/138847449/55317741-5585-47c8-9e45-90df52de8957">
</p>

2. Choose the "Plotting and Analysis" tab, and select the data you'd like to plot or output. 
![Screen recording of switching to the plotting tab, choosing a specific frequency and dB SPL, and then choosing various plotting and table output functions](https://github.com/user-attachments/assets/114ffb6d-333d-46ff-9ca9-45791d59f637)

### Optional usage:

* Hover over the top right of a data table to download to csv
![download_csv](https://github.com/user-attachments/assets/62367808-e46d-43d7-84ff-e0b5e0639c13)

* Edit peaks for individual waveforms: click "Edit peaks table", make edits, and click "Done editing"
  ![Untitled (1)](https://github.com/user-attachments/assets/df4ece33-2fc2-49b9-abfb-9d7bf4ab9c47)

* Edit thresholds for waveform stacks
  ![edit_threshold gif](https://github.com/user-attachments/assets/7e7dc14a-744b-44db-94e4-9d806f4dccc2)

* Clear manual edits using the Manual Edits Management toolbar
  ![clear_manuals](https://github.com/user-attachments/assets/9bcfe194-7ca4-498b-8c54-cd929a46c9c7)


