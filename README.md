# Auditory Brainstem Response Analyzer (ABRA)
ABRA is a web application created using Streamlit which allows users to batch-upload and analyze ABR data.
It can either be run on the [web](ucsdabranalysis.streamlit.app)
OR
locally by going to working directory and running `streamlit run wave_plot_app.py` in your terminal.
<br>
First upload your file. If you're loading an Tucker Davis .arf file please select whether you got the file from BioSigRP or BioSigRZ. If you're uploading a .csv file make sure that the title for the decibel column is `Level(dB)`, the title for the frequency column is `Freq(Hz)`, and the vector of data points ends each corresponding row.
<br></br>
Here is an example of what it should look like:
<p align="center">
<img width="200" alt="image" src="https://github.com/abhierra2/ucsdpracticum/assets/138847449/55317741-5585-47c8-9e45-90df52de8957">
</p>
Usually, the waves are usually recorded over 10 ms with 244 data points, but we can truncate the waves if they are longer than 10 ms and interpolate over 244 points if it is not 244 points. This works best for our thresholding model.
<br>
Select `Level`. If uploading multiple files, make sure to check the files you want to analyze. The rest of the app is self explanatory, but an important thing to note is that when you check `Plot Time Warped Curves`, it only works for plotting waves at one frequency.
<br>
<p align="center">
<img width="150" alt="image showing how to upload files" src="https://github.com/abhierra2/ucsdpracticum/assets/138847449/f56df5a4-4712-4a12-bfe9-5f7b0fcc2ed6">
</p>
<br>
Here is an example of when you click `Plot a Single Wave (Freq, dB)`. This also shows the metrics table and our estimated threshold for you to investigate.
<p align="center">
<img width="350" alt="image" src="https://github.com/abhierra2/ucsdpracticum/assets/138847449/c9b5ebd5-a8c8-40de-87aa-36b4af22b311">
</p>
