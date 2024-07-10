# Auditory Brainstem Response Analyzer (ABRA)
ABRA is a web application created using Streamlit which allows users to batch-upload and analyze ABR data.
It can either be run on the [web](https://ucsdabranalysis.streamlit.app) OR locally.
<br></br>
## Instructions for running the app locally:
1. Download and Install Visual C++ Build Tools: Go to the Microsoft C++ Build Tools page.
Download and run the installer.
During installation, make sure to select the "Desktop development with C++" workload.
After installation, you might need to restart your command prompt or terminal for the changes to take effect.
2. Clone this repository to your computer and open a new anaconda command prompt: <br></br>
`(base) C:\Users\username\abranalysis-main>conda create -n abra python=3.9` <br></br>
`(base) C:\Users\username\abranalysis-main> conda activate abra` <br></br>
`(abra) C:\Users\username\abranalysis-main>pip install -r requirements.txt` <br></br>
`(abra) C:\Users\username\abranalysis-main>streamlit run wave_plot_app.py` <br></br>
## Using the app:
First upload your file. If you're loading an Tucker Davis .arf file please select whether you got the file from BioSigRP or BioSigRZ. If you're uploading a .csv file make sure that the title for the decibel column is `Level(dB)`, the title for the frequency column is `Freq(Hz)`, and the vector of data points (in microvolts) ends each corresponding row.
<br></br>
Here is an example of what it should look like:
<p align="center">
<img width="900" alt="image" src="https://github.com/abhierra2/ucsdpracticum/assets/138847449/55317741-5585-47c8-9e45-90df52de8957">
</p>
There is flexibility with this format. For example, if you are importing a .csv file extracted from BioSig, as long as the title for the decibel column is `Level(dB)` and the title for the frequency column is `Freq(Hz)`, it should work. Usually, the waves are recorded over 10 ms with 244 data points. Make sure to truncate the waves to 10 ms for best analysis, and the code interpolates each wave over 244 points if it is not 244 points.
<br></br>
Select "Level". If uploading multiple files, make sure to check the files you want to analyze. The rest of the app is self explanatory, but an important thing to note is that when you check "Plot Time Warped Curves", it only works for plotting waves at one frequency.
<br>
<p align="center">
<img width="350" alt="image showing how to upload files" src="https://github.com/abhierra2/ucsdpracticum/assets/138847449/f56df5a4-4712-4a12-bfe9-5f7b0fcc2ed6">
</p>
<br>
Here is an example of when you click "Plot a Single Wave (Freq, dB)". This also shows the metrics table and our estimated threshold for you to investigate.
<p align="center">
<img width="850" alt="image" src="https://github.com/abhierra2/ucsdpracticum/assets/138847449/c9b5ebd5-a8c8-40de-87aa-36b4af22b311">
</p>
