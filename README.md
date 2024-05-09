# Auditory Brainstem Response Analyzer (ABRA)
ABRA is a web application created using Streamlit which allows users to batch-upload and analyze ABR data.
It can either be run on the [web](ucsdabranalysis.streamlit.app)
OR
locally by going to working directory and running `streamlit run wave_plot_app.py` in your terminal.
<b>
First upload your file. If you're loading an Tucker Davis .arf file please select whether you got the file from BioSigRP or BioSigRZ. If you're uploading a .csv file make sure that the title for the decibel column is `Level(dB)`, the title for the frequency column is `Freq(Hz)`, and the vector of data points ends each corresponding row. Select `Level`. If uploading multiple files, make sure to check the files you want to analyze. The rest of the app is self explanatory, but an important thing to note is that when you check 'Plot Time Warped Curves', it only works for plotting waves at one frequency.
<img width="299" alt="image" src="https://github.com/abhierra2/ucsdpracticum/assets/138847449/5cc19a02-4651-4d92-a723-9a83021035d0">
