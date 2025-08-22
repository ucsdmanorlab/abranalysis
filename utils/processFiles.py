import streamlit as st
import pandas as pd
import os
import tempfile
import numpy as np
import struct
from .ui import * 

def CFTSread(PATH):
    with open(PATH, 'r', encoding='latin1') as file:
        data = file.readlines()

    data_start = False
    data_list = []
    for line in data: 
        if not data_start and line.startswith(':'): # Header lines
            if 'FREQ' in line:
                try:
                    freqs = [float(line.split('FREQ:')[1].split()[0].strip())*1000]
                except: #Click
                    freqs = [line.split('FREQ:')[1].split()[0].strip()]
            if 'LEVELS' in line:
                dbs = (line.split('LEVELS:')[1].strip()[:-1])
                dbs = [int(dB) for dB in dbs.split(';')]
            if 'SAMPLE' in line:
                sample_us = float(line.split('SAMPLE')[1].split(':')[1].strip())
            if 'DATA' in line:
                data_start = True

        elif data_start:
            if len(line.strip()) == 0:
                continue
            data_list.append([float(d) for d in line.strip().split()])

    data = np.array(data_list)

    duration_ms = (data.shape[0]-1) * sample_us/1000
    rows = []
    
    for dB_i in range(len(dbs)):
        db = dbs[dB_i]
        data_col = data[:, dB_i]
        
        wave_data = {f'{i}': data_col[i] for i in range(len(data_col))}#, v in data_col} 
        row = {'Freq(Hz)': freqs[0], 'Level(dB)': db, **wave_data}
        rows.append(row)
        
        df = pd.DataFrame(rows)
    return duration_ms, df

def arfread(PATH, **kwargs):
    def get_str(data):
        # return string up until null character only
        ind = data.find(b'\x00')
        if ind > 0:
            data = data[:ind]
        return data.decode('utf-8')
    # defaults
    RP = kwargs.get('RP', False)
    
    isRZ = not RP
    
    data = {'metadata': {}, 'groups': []}

    # open file
    with open(PATH, 'rb') as fid:
        # open metadata data
        data['metadata']['ftype'] = struct.unpack('h', fid.read(2))[0]
        data['metadata']['ngrps'] = struct.unpack('h', fid.read(2))[0]
        data['metadata']['nrecs'] = struct.unpack('h', fid.read(2))[0]
        data['metadata']['grpseek'] = struct.unpack('200i', fid.read(4*200))
        data['metadata']['recseek'] = struct.unpack('2000i', fid.read(4*2000))
        data['metadata']['file_ptr'] = struct.unpack('i', fid.read(4))[0]

        data['groups'] = []
        bFirstPass = True
        for x in range(data['metadata']['ngrps']):
            # jump to the group location in the file
            fid.seek(data['metadata']['grpseek'][x], 0)

            # open the group
            data['groups'].append({
                'grpn': struct.unpack('h', fid.read(2))[0],
                'frecn': struct.unpack('h', fid.read(2))[0],
                'nrecs': struct.unpack('h', fid.read(2))[0],
                'ID': get_str(fid.read(16)),
                'ref1': get_str(fid.read(16)),
                'ref2': get_str(fid.read(16)),
                'memo': get_str(fid.read(50)),
            })

            # read temporary timestamp
            if bFirstPass:
                if isRZ:
                    ttt = struct.unpack('q', fid.read(8))[0]
                    fid.seek(-8, 1)
                    data['fileType'] = 'BioSigRZ'
                else:
                    ttt = struct.unpack('I', fid.read(4))[0]
                    fid.seek(-4, 1)
                    data['fileType'] = 'BioSigRP'
                #data['fileTime'] = datetime.datetime.utcfromtimestamp(ttt/86400 + datetime.datetime(1970, 1, 1).timestamp()).strftime('%Y-%m-%d %H:%M:%S')
                bFirstPass = False

            if isRZ:
                grp_t_format = 'q'
                beg_t_format = 'q'
                end_t_format = 'q'
                read_size = 8
            else:
                grp_t_format = 'I'
                beg_t_format = 'I'
                end_t_format = 'I'
                read_size = 4

            data['groups'][x]['beg_t'] = struct.unpack(beg_t_format, fid.read(read_size))[0]
            data['groups'][x]['end_t'] = struct.unpack(end_t_format, fid.read(read_size))[0]

            data['groups'][x].update({
                'sgfname1': get_str(fid.read(100)),
                'sgfname2': get_str(fid.read(100)),
                'VarName1': get_str(fid.read(15)),
                'VarName2': get_str(fid.read(15)),
                'VarName3': get_str(fid.read(15)),
                'VarName4': get_str(fid.read(15)),
                'VarName5': get_str(fid.read(15)),
                'VarName6': get_str(fid.read(15)),
                'VarName7': get_str(fid.read(15)),
                'VarName8': get_str(fid.read(15)),
                'VarName9': get_str(fid.read(15)),
                'VarName10': get_str(fid.read(15)),
                'VarUnit1': get_str(fid.read(5)),
                'VarUnit2': get_str(fid.read(5)),
                'VarUnit3': get_str(fid.read(5)),
                'VarUnit4': get_str(fid.read(5)),
                'VarUnit5': get_str(fid.read(5)),
                'VarUnit6': get_str(fid.read(5)),
                'VarUnit7': get_str(fid.read(5)),
                'VarUnit8': get_str(fid.read(5)),
                'VarUnit9': get_str(fid.read(5)),
                'VarUnit10': get_str(fid.read(5)),
                'SampPer_us': struct.unpack('f', fid.read(4))[0],
                'cc_t': struct.unpack('i', fid.read(4))[0],
                'version': struct.unpack('h', fid.read(2))[0],
                'postproc': struct.unpack('i', fid.read(4))[0],
                'dump': get_str(fid.read(92)),
                'recs': [],
            })

            for i in range(data['groups'][x]['nrecs']):
                record_data = {
                        'recn': struct.unpack('h', fid.read(2))[0],
                        'grpid': struct.unpack('h', fid.read(2))[0],
                        'grp_t': struct.unpack(grp_t_format, fid.read(read_size))[0],
                        #'grp_d': datetime.utcfromtimestamp(data['groups'][x]['recs'][i]['grp_t']/86400 + datetime(1970, 1, 1).timestamp()).strftime('%Y-%m-%d %H:%M:%S'),
                        'newgrp': struct.unpack('h', fid.read(2))[0],
                        'sgi': struct.unpack('h', fid.read(2))[0],
                        'chan': struct.unpack('B', fid.read(1))[0],
                        'rtype': get_str(fid.read(1)),
                        'npts': struct.unpack('H' if isRZ else 'h', fid.read(2))[0],
                        'osdel': struct.unpack('f', fid.read(4))[0],
                        'dur_ms': struct.unpack('f', fid.read(4))[0],
                        'SampPer_us': struct.unpack('f', fid.read(4))[0],
                        'artthresh': struct.unpack('f', fid.read(4))[0],
                        'gain': struct.unpack('f', fid.read(4))[0],
                        'accouple': struct.unpack('h', fid.read(2))[0],
                        'navgs': struct.unpack('h', fid.read(2))[0],
                        'narts': struct.unpack('h', fid.read(2))[0],
                        'beg_t': struct.unpack(beg_t_format, fid.read(read_size))[0],
                        'end_t': struct.unpack(end_t_format, fid.read(read_size))[0],
                        'Var1': struct.unpack('f', fid.read(4))[0],
                        'Var2': struct.unpack('f', fid.read(4))[0],
                        'Var3': struct.unpack('f', fid.read(4))[0],
                        'Var4': struct.unpack('f', fid.read(4))[0],
                        'Var5': struct.unpack('f', fid.read(4))[0],
                        'Var6': struct.unpack('f', fid.read(4))[0],
                        'Var7': struct.unpack('f', fid.read(4))[0],
                        'Var8': struct.unpack('f', fid.read(4))[0],
                        'Var9': struct.unpack('f', fid.read(4))[0],
                        'Var10': struct.unpack('f', fid.read(4))[0],
                        'data': [] #list(struct.unpack(f'{data["groups"][x]["recs"][i]["npts"]}f', fid.read(4*data['groups'][x]['recs'][i]['npts'])))
                    }
                
                # skip all 10 cursors placeholders
                fid.seek(36*10, 1)
                record_data['data'] = list(struct.unpack(f'{record_data["npts"]}f', fid.read(4*record_data['npts'])))
                #record_data['grp_d'] = datetime.datetime.utcfromtimestamp(record_data['grp_t'] / 86400 + datetime.datetime(1970, 1, 1).timestamp()).strftime('%Y-%m-%d %H:%M:%S')

                data['groups'][x]['recs'].append(record_data)

    return data

def clear_calculation_cache_for_files(file_names):
    if 'calculated_thresholds' in st.session_state:
        keys_to_remove = []
        for key in st.session_state.calculated_thresholds.keys():
            if any(file_name in key for file_name in file_names):
                keys_to_remove.append(key)
        for key in keys_to_remove:
            del st.session_state.calculated_thresholds[key]
    
    if 'calculated_waves' in st.session_state:
        keys_to_remove = []
        for key in st.session_state.calculated_waves.keys():
            if any(file_name in key for file_name in file_names):
                keys_to_remove.append(key)
        for key in keys_to_remove:
            del st.session_state.calculated_waves[key]

def process_uploaded_files_cached(uploaded_files, is_rz_file, click, is_atten):
    """Process files only once and cache in session state"""
    # Create a cache key based on file names and settings
    file_names = [f.name for f in uploaded_files] if uploaded_files else []
    cache_key = f"{file_names}_{is_rz_file}_{click}_{is_atten}"
    
    # Check if we already processed these files with these settings
    if 'processed_data_cache_key' in st.session_state and st.session_state.processed_data_cache_key == cache_key:
        return st.session_state.processed_dfs, st.session_state.processed_duration
    
    clear_calculation_cache_for_files(file_names)

    dfs = []
    duration = None
    progress_bar, status_text, count = initialize_progress_bar()
    for file in uploaded_files:
        count = update_progress_bar(progress_bar, status_text, count, len(uploaded_files), f"Reading {file.name}...")
        # Use tempfile
        temp_file_path = os.path.join(tempfile.gettempdir(), file.name)
        with open(temp_file_path, 'wb') as temp_file:
            temp_file.write(file.read())

        if file.name.endswith(".arf"):
        # Read ARF file
            data = arfread(temp_file.name, RP=(is_rz_file == 'RP')) 
            
            # Process ARF data
            rows = []

            for group in data['groups']:
                for rec in group['recs']:
                    if not click:
                        freq = rec['Var1']
                        db = rec['Var2']
                    else:
                        freq = 'Click'
                        db = rec['Var1']
                    
                    wave_cols = list(enumerate(rec['data']))
                    wave_data = {f'{i}':v*1e6 for i, v in wave_cols} 
                    
                    if not is_atten:
                        row = {'Freq(Hz)': freq, 'Level(dB)': db, **wave_data}
                        rows.append(row)
                    else:
                        row = {'Freq(Hz)': freq, 'PostAtten(dB)': db, **wave_data}
                        rows.append(row)
                    duration = rec['dur_ms']

            df = pd.DataFrame(rows)

        elif file.name.endswith((".asc", ".tsv")):
            # Process ASC file
            duration, df = CFTSread(temp_file_path)

        elif file.name.endswith(".csv"):
            # Process CSV
            if pd.read_csv(temp_file_path).shape[1] > 1:
                df = pd.read_csv(temp_file_path)
            else:
                df = pd.read_csv(temp_file_path, skiprows=2)
            
        df.name = file.name
        dfs.append(df)
       
    # Cache the results
    st.session_state.processed_dfs = dfs
    st.session_state.processed_duration = duration
    st.session_state.processed_data_cache_key = cache_key
    clear_status_bar(progress_bar, status_text)
    
    return dfs, duration

def get_selected_data():
    """Get currently selected files and their data"""
    if 'processed_dfs' not in st.session_state:
        return [], []
    
    selected_files = []
    selected_dfs = []
    
    for idx, df in enumerate(st.session_state.processed_dfs):
        # Use session state to track checkbox selections
        checkbox_key = f"file_{idx}"
        if st.session_state.get(checkbox_key, True):  # Default to True
            temp_file_path = os.path.join(tempfile.gettempdir(), df.name)
            selected_files.append(temp_file_path)
            selected_dfs.append(df)
    
    return selected_files, selected_dfs

def db_column_name():
    atten = st.session_state.get('atten', False)
    return 'Level(dB)' if not atten else 'PostAtten(dB)'

def db_value(file_name, freq, db):
    atten = st.session_state.get('atten', False)
    if atten:
        return st.session_state.calibration_levels[(file_name, freq)] - int(db)
    else:
        return db
    