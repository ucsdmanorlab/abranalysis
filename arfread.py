import struct
import numpy as np

def arfread(PATH, PLOT=False, RP=False):
    def get_str(byte_str):
        null_char_idx = byte_str.find(b'\x00')
        if null_char_idx > 0:
            return byte_str[:null_char_idx].decode('utf-8')
        return byte_str.decode('utf-8')
    
    isRZ = not RP
    
    data = {
        'RecHead': {},
        'groups': []
    }

    with open(PATH, 'rb') as file:
        data['RecHead']['ftype'] = struct.unpack('h', file.read(2))[0]
        data['RecHead']['ngrps'] = struct.unpack('h', file.read(2))[0]
        data['RecHead']['nrecs'] = struct.unpack('h', file.read(2))[0]
        data['RecHead']['grpseek'] = struct.unpack('200i', file.read(800))
        data['RecHead']['recseek'] = struct.unpack('2000i', file.read(8000))
        data['RecHead']['file_ptr'] = struct.unpack('i', file.read(4))[0]

        bFirstPass = True

        for x in range(data['RecHead']['ngrps']):
            file.seek(data['RecHead']['grpseek'][x], 0)
            
            group = {
                'grpn': struct.unpack('h', file.read(2))[0],
                'frecn': struct.unpack('h', file.read(2))[0],
                'nrecs': struct.unpack('h', file.read(2))[0],
                'ID': get_str(file.read(16)),
                'ref1': get_str(file.read(16)),
                'ref2': get_str(file.read(16)),
                'memo': get_str(file.read(50))
            }
            
            if bFirstPass:
                if isRZ:
                    ttt = struct.unpack('q', file.read(8))[0]
                    file.seek(-8, 1)
                    data['fileType'] = 'BioSigRZ'
                else:
                    ttt = struct.unpack('I', file.read(4))[0]
                    file.seek(-4, 1)
                    data['fileType'] = 'BioSigRP'
                
                data['fileTime'] = str(np.datetime64('1970-01-01') + np.timedelta64(ttt, 's'))
                bFirstPass = False
            
            if isRZ:
                group['beg_t'] = struct.unpack('q', file.read(8))[0]
                group['end_t'] = struct.unpack('q', file.read(8))[0]
            else:
                group['beg_t'] = struct.unpack('i', file.read(4))[0]
                group['end_t'] = struct.unpack('i', file.read(4))[0]
            
            group['sgfname1'] = get_str(file.read(100))
            group['sgfname2'] = get_str(file.read(100))

            group['VarName1'] = get_str(file.read(15))
            group['VarName2'] = get_str(file.read(15))
            group['VarName3'] = get_str(file.read(15))
            group['VarName4'] = get_str(file.read(15))
            group['VarName5'] = get_str(file.read(15))
            group['VarName6'] = get_str(file.read(15))
            group['VarName7'] = get_str(file.read(15))
            group['VarName8'] = get_str(file.read(15))
            group['VarName9'] = get_str(file.read(15))
            group['VarName10'] = get_str(file.read(15))

            group['VarUnit1'] = get_str(file.read(5))
            group['VarUnit2'] = get_str(file.read(5))
            group['VarUnit3'] = get_str(file.read(5))
            group['VarUnit4'] = get_str(file.read(5))
            group['VarUnit5'] = get_str(file.read(5))
            group['VarUnit6'] = get_str(file.read(5))
            group['VarUnit7'] = get_str(file.read(5))
            group['VarUnit8'] = get_str(file.read(5))
            group['VarUnit9'] = get_str(file.read(5))
            group['VarUnit10'] = get_str(file.read(5))

            group['SampPer_us'] = struct.unpack('f', file.read(4))[0]

            group['cc_t'] = struct.unpack('i', file.read(4))[0]
            group['version'] = struct.unpack('h', file.read(2))[0]
            group['postproc'] = struct.unpack('i', file.read(4))[0]
            group['dump'] = get_str(file.read(92))

            group['recs'] = []

            for i in range(group['nrecs']):
                rec = {
                    'recn': struct.unpack('h', file.read(2))[0],
                    'grpid': struct.unpack('h', file.read(2))[0]
                }
                
                if isRZ:
                    rec['grp_t'] = struct.unpack('q', file.read(8))[0]
                else:
                    rec['grp_t'] = struct.unpack('i', file.read(4))[0]
                
                rec['grp_d'] = str(np.datetime64('1970-01-01') + np.timedelta64(rec['grp_t'], 's'))
                
                rec['newgrp'] = struct.unpack('h', file.read(2))[0]
                rec['sgi'] = struct.unpack('h', file.read(2))[0]
                rec['chan'] = struct.unpack('B', file.read(1))[0]
                rec['rtype'] = get_str(file.read(1))

                if isRZ:
                    rec['npts'] = struct.unpack('H', file.read(2))[0]
                else:
                    rec['npts'] = struct.unpack('h', file.read(2))[0]
                
                rec['osdel'] = struct.unpack('f', file.read(4))[0]
                rec['dur_ms'] = struct.unpack('f', file.read(4))[0]
                rec['SampPer_us'] = struct.unpack('f', file.read(4))[0]

                rec['artthresh'] = struct.unpack('f', file.read(4))[0]
                rec['gain'] = struct.unpack('f', file.read(4))[0]
                rec['accouple'] = struct.unpack('h', file.read(2))[0]

                rec['navgs'] = struct.unpack('h', file.read(2))[0]
                rec['narts'] = struct.unpack('h', file.read(2))[0]

                if isRZ:
                    rec['beg_t'] = struct.unpack('q', file.read(8))[0]
                    rec['end_t'] = struct.unpack('q', file.read(8))[0]
                else:
                    rec['beg_t'] = struct.unpack('i', file.read(4))[0]
                    rec['end_t'] = struct.unpack('i', file.read(4))[0]
                    
                    rec['Var1'] = struct.unpack('f', file.read(4))[0]
                    rec['Var2'] = struct.unpack('f', file.read(4))[0]
                    rec['Var3'] = struct.unpack('f', file.read(4))[0]
                    rec['Var4'] = struct.unpack('f', file.read(4))[0]
                    rec['Var5'] = struct.unpack('f', file.read(4))[0]
                    rec['Var6'] = struct.unpack('f', file.read(4))[0]
                    rec['Var7'] = struct.unpack('f', file.read(4))[0]
                    rec['Var8'] = struct.unpack('f', file.read(4))[0]
                    rec['Var9'] = struct.unpack('f', file.read(4))[0]
                    rec['Var10'] = struct.unpack('f', file.read(4))[0]

                    # skip all 10 cursors placeholders
                    file.seek(36 * 10, 1)
                    
                    rec_data = struct.unpack(f'{rec["npts"]}f', file.read(4 * rec['npts']))
                    rec['data'] = list(rec_data)

                    group['recs'].append(rec)
                
                data['groups'].append(group)
    
    if PLOT:
        import matplotlib.pyplot as plt
        
        for group in data['groups']:
            plt.figure()
            plot_offset = max([max(np.abs(rec['data'])) for rec in group['recs']]) * 1.2
            
            for rec in group['recs']:
                plt.plot(np.array(rec['data']) - plot_offset * group['recs'].index(rec))
            
            plt.title(f'Group {group["grpn"]}')
            plt.axis('off')
    
    return data
