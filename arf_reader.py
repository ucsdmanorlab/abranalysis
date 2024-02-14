import struct
from datetime import datetime

def arfread(PATH, **kwargs):
    # defaults
    PLOT = kwargs.get('PLOT', False)
    RP = kwargs.get('RP', False)
    
    isRZ = not RP
    
    data = {'RecHead': {}, 'groups': []}

    # open file
    with open(PATH, 'rb') as fid:
        # open RecHead data
        data['RecHead']['ftype'] = struct.unpack('h', fid.read(2))[0]
        data['RecHead']['ngrps'] = struct.unpack('h', fid.read(2))[0]
        data['RecHead']['nrecs'] = struct.unpack('h', fid.read(2))[0]
        data['RecHead']['grpseek'] = struct.unpack('200i', fid.read(4*200))
        data['RecHead']['recseek'] = struct.unpack('2000i', fid.read(4*2000))
        data['RecHead']['file_ptr'] = struct.unpack('i', fid.read(4))[0]

        data['groups'] = []
        bFirstPass = True
        for x in range(data['RecHead']['ngrps']):
            # jump to the group location in the file
            fid.seek(data['RecHead']['grpseek'][x], 0)

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
                data['fileTime'] = datetime.utcfromtimestamp(ttt/86400 + datetime(1970, 1, 1).timestamp()).strftime('%Y-%m-%d %H:%M:%S')
                bFirstPass = False

            if isRZ:
                data['groups'][x]['beg_t'] = struct.unpack('q', fid.read(8))[0]
                data['groups'][x]['end_t'] = struct.unpack('q', fid.read(8))[0]
            else:
                data['groups'][x]['beg_t'] = struct.unpack('I', fid.read(4))[0]
                data['groups'][x]['end_t'] = struct.unpack('I', fid.read(4))[0]
            
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
                        'grp_t': struct.unpack('q' if isRZ else 'I', fid.read(8))[0],
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
                        'beg_t': struct.unpack('q' if isRZ else 'I', fid.read(8))[0],
                        'end_t': struct.unpack('q' if isRZ else 'I', fid.read(8))[0],
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
                
                record_data['data'] = list(struct.unpack(f'{record_data["npts"]}f', fid.read(4*record_data['npts'])))

                record_data['grp_d'] = datetime.utcfromtimestamp(record_data['grp_t'] / 86400 + datetime(1970, 1, 1).timestamp()).strftime('%Y-%m-%d %H:%M:%S')

                # skip all 10 cursors placeholders
                fid.seek(36*10, 1)
                data['groups'][x]['recs'].append(record_data)

            if PLOT:
                import matplotlib.pyplot as plt

                # determine reasonable spacing between plots
                d = [x['data'] for x in data['groups'][x]['recs']]
                plot_offset = max(max(map(abs, [item for sublist in d for item in sublist]))) * 1.2

                plt.figure()

                for i in range(data['groups'][x]['nrecs']):
                    plt.plot([item - plot_offset * i for item in data['groups'][x]['recs'][i]['data']])
                    plt.hold(True)

                plt.title(f'Group {data["groups"][x]["grpn"]}')
                plt.axis('off')
                plt.show()

    return data

def get_str(data):
    # return string up until null character only
    ind = data.find(b'\x00')
    if ind > 0:
        data = data[:ind]
    return data.decode('utf-8')

# Example usage:
data = arfread('617F_tone.arf')
for group in range(data['RecHead']['ngrps']):
    #this example has only 1 group
    print('group #', group)
    for rec in range(data['groups'][group]['nrecs']):
        # this one group has many recordings (different dBs, freqs)

        freq = data['groups'][group]['recs'][rec]['Var1']
        dB = data['groups'][group]['recs'][rec]['Var2']
        wave_data = data['groups'][0]['recs'][rec]['data']

        gain = data['groups'][group]['recs'][rec]['gain'] # this might be where that 20x multiplier comes from? double check with elena
        total_t = data['groups'][group]['recs'][rec]['dur_ms']

        print('recording #',
                rec,
                'duration (ms)',
                total_t,
                'gain',
                gain,
                'Freq?',
                freq,
                'dB?',
                dB,
                )
