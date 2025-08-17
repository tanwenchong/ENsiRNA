import pandas as pd

import json
from utils.rna_utils import VOCAB
import argparse
import os
from data.mod_utils import MOD_VOCAB
import torch
import RNA
import re
import subprocess
import multiprocessing


#example （5'->3' and 3'->5'）
#seq1 = "CUUACGCUGAGUACUUCGA".lower()
#seq2 = "GAAUGCGACUCAUGAAGCU".lower()[::-1]


#python -m data.get_pdb 
RF='/app/rosetta/rosetta.binary.linux.release-371'
FF='/app/rosetta/rosetta.binary.linux.release-371/main/source/bin/rna_denovo.static.linuxgccrelease'
EX='/app/rosetta/rosetta.binary.linux.release-371/main/tools/rna_tools/silent_util/extract_lowscore_decoys.py'

#MOD_VOCAB.mod2index acid_mod atom_mod

class Data_Prepare:
    def __init__(self,excel_dir,pdb_dir):
        self.excel_dir=excel_dir
        self.pdb_dir=pdb_dir

        self.json_dir=excel_dir[:-5]+'.json'
        self.secondary_structure = True
        self.chunk_size=10
        self.json = True

        if self.secondary_structure == False:
            self.json_dir=excel_dir[:-5]+'no2.json'


    def get_path(self,ID):
        return f"{self.pdb_dir}/{ID}.pdb"

    def get_smask(self,item):
        sp=[]
        ap=[]
        for i in str(item['sense pos']).replace(' ','').split('*'):
            sp += i.split(',')
        for i in str(item['anti pos']).replace(' ','').split('*'):
            ap += i.split(',')
        
        smask=[int(i) for i in set(sp)] + [int(i)+len(item['sense seq']) for i in set(ap)]
        return smask

    def get_atommod(self,item):
        smode=None
        amode=None
        atom_mask=[[0 for _ in range(VOCAB.MAX_ATOM_NUMBER)] for _ in range(1+len(item['sense seq'])+len(item['anti seq']))]
        if item['sense mod']!='none' and item['sense mod'] !=0:
            smode=item['sense mod'].split('* ')
            spose=str(item['sense pos']).replace(' ','').split('*')
        if item['anti mod']!='none' and item['anti mod'] !=0:    
            amode=item['anti mod'].split('* ')
            apose=str(item['anti pos']).replace(' ','').split('*')

        offset=len(item['sense seq'])
        #print(item['ID'])

        

        for pos in range(1,1+len(item['sense seq'])):
            #add sugar
            atom_mask[pos][0]=MOD_VOCAB.mod2index['Standard sugar']
            #add phosphate
            atom_mask[pos][1]=MOD_VOCAB.mod2index['Standard phosphate']
            #add base
            if item['sense raw seq'][pos-1]=='a':
                atom_mask[pos][2]=MOD_VOCAB.mod2index['Standard adenine']
            if item['sense raw seq'][pos-1]=='c':
                atom_mask[pos][2]=MOD_VOCAB.mod2index['Standard cytosine']
            if item['sense raw seq'][pos-1]=='g':
                atom_mask[pos][2]=MOD_VOCAB.mod2index['Standard guanine']
            if item['sense raw seq'][pos-1]=='u':
                atom_mask[pos][2]=MOD_VOCAB.mod2index['Standard uracil']
            if item['sense raw seq'][pos-1]=='t':
                atom_mask[pos][2]=MOD_VOCAB.mod2index['Standard thymine']
                atom_mask[pos][0]=MOD_VOCAB.mod2index['2-Deoxyribonucleotide']

        if smode!=None:                
            for i in range(len(smode)):
                #add mod base
                if smode[i]=='Inverted abasic':
                    for pos in str(spose[i]).split(','): 
                        atom_mask[int(pos)][2]=0
                        atom_mask[int(pos)][2]=0

                if smode[i] in MOD_VOCAB.base_mod:
                    for pos in str(spose[i]).split(','): 
                        if item['sense raw seq'][int(pos)-1]=='a':
                            atom_mask[int(pos)][2]=MOD_VOCAB.mod2index[smode[i]]
                        if item['sense raw seq'][int(pos)-1]=='c':
                            atom_mask[int(pos)][2]=MOD_VOCAB.mod2index[smode[i]]
                        if item['sense raw seq'][int(pos)-1]=='g':
                            atom_mask[int(pos)][2]=MOD_VOCAB.mod2index[smode[i]]
                        if item['sense raw seq'][int(pos)-1]=='u':
                            atom_mask[int(pos)][2]=MOD_VOCAB.mod2index[smode[i]]
                        if item['sense raw seq'][int(pos)-1]=='t':
                            atom_mask[int(pos)][2]=MOD_VOCAB.mod2index[smode[i]]

                #add mod phosphate
                if smode[i] in MOD_VOCAB.phosphate_mod:
                    #print(item['ID'])
                    for pos in str(spose[i]).split(','): 
                        try:
                            atom_mask[int(pos)][1]=MOD_VOCAB.mod2index[smode[i]]
                        except:
                            print(print(item['ID']),'wrong')

                #add mod sugar
                if smode[i] in MOD_VOCAB.sugar_mod:
                    for pos in str(spose[i]).split(','): 
                        try:
                            atom_mask[int(pos)][0]=MOD_VOCAB.mod2index[smode[i]]
                        except:
                            print(print(item['ID']),'wrong')



        _mask=[[0 for _ in range(VOCAB.MAX_ATOM_NUMBER)] for _ in range(len(item['anti seq']))]
        for pos in range(len(item['anti seq'])):
            #add sugar
            _mask[int(pos)][0]=MOD_VOCAB.mod2index['Standard sugar']
            #add phosphate
            _mask[int(pos)][1]=MOD_VOCAB.mod2index['Standard phosphate']
            #add base
            if item['anti raw seq'][pos]=='a':
                _mask[int(pos)][2]=MOD_VOCAB.mod2index['Standard adenine']
            if item['anti raw seq'][pos]=='c':
                _mask[int(pos)][2]=MOD_VOCAB.mod2index['Standard cytosine']
            if item['anti raw seq'][pos]=='g':
                _mask[int(pos)][2]=MOD_VOCAB.mod2index['Standard guanine']
            if item['anti raw seq'][pos]=='u':
                _mask[int(pos)][2]=MOD_VOCAB.mod2index['Standard uracil']
            if item['anti raw seq'][pos]=='t':
                _mask[int(pos)][2]=MOD_VOCAB.mod2index['Standard thymine']   
                _mask[int(pos)][0]=MOD_VOCAB.mod2index['2-Deoxyribonucleotide']         
                        


        if amode!=None:
                
            for i in range(len(amode)):
            
               
                if amode[i]=='Inverted abasic':  #?
                    for pos in str(apose[i]).split(','): 
                        _mask[int(pos)-1][2]=0
                        _mask[int(pos)-1][2]=0
                #add mod base
                if amode[i] in MOD_VOCAB.base_mod:
                    for pos in str(apose[i]).split(','): 
                        if item['anti raw seq'][int(pos)-1]=='a':
                            _mask[int(pos)-1][2]=MOD_VOCAB.mod2index[amode[i]]
                        if item['anti raw seq'][int(pos)-1]=='c':
                            _mask[int(pos)-1][2]=MOD_VOCAB.mod2index[amode[i]]
                        if item['anti raw seq'][int(pos)-1]=='g':
                            _mask[int(pos)-1][2]=MOD_VOCAB.mod2index[amode[i]]
                        if item['anti raw seq'][int(pos)-1]=='u':
                            _mask[int(pos)-1][2]=MOD_VOCAB.mod2index[amode[i]]
                        if item['anti raw seq'][int(pos)-1]=='t':
                            _mask[int(pos)-1][2]=MOD_VOCAB.mod2index[amode[i]]

                #add mod phosphate
                if amode[i] in MOD_VOCAB.phosphate_mod:
                    for pos in str(apose[i]).split(','): 
                        try:
                            _mask[int(pos)-1][1]=MOD_VOCAB.mod2index[amode[i]]
                        except:
                            print(print(item['ID']),'wrong')
                            return None                        

                #add mod sugar
                if amode[i] in MOD_VOCAB.sugar_mod:
                    for pos in str(apose[i]).split(','): 
                        try:
                            _mask[int(pos)-1][0]=MOD_VOCAB.mod2index[amode[i]]
                        except:
                            print(print(item['ID']),'wrong')
                            return None


        atom_mask[offset+1:]=_mask

        return atom_mask

    def seq_pre(self,seq):
        return seq.replace('t','u')

    def raw_pre(self,seq):
        return seq.lower().replace(' + ','').replace('d','').replace('y','g').replace('x','t').replace(' ','')

    def get_rpos(self,item):
        pos = [0]
        for i in range(1,1+len(item['sense seq'])):
            pos.append(i)
        for i in range(30,30+len(item['anti seq'])):
            pos.append(i)
        return pos

    def chunk_dataframe(self,df, chunk_size):
        chunks = []
        chunk_count = len(df) // chunk_size
        for i in range(chunk_count):
            chunks.append(df[i * chunk_size : (i + 1) * chunk_size])
        if len(df) % chunk_size != 0:
            chunks.append(df[chunk_count * chunk_size:])
        return chunks

    def get_anti_start(self,data):
        seq1=data['sense seq']
        seq2=data['anti seq']
        com=f" echo -e \"{seq1}\n{seq2}\n\" | RNAplex "
        raw_se=subprocess.run(com,shell=True,capture_output=True, text=True).stdout 
        secondary_seq1 = re.split(r'\s+', raw_se)[0].split('&')[0]
        secondary_seq2 = re.split(r'\s+', raw_se)[0].split('&')[1]
        seq2_ = re.split(r'\s+', raw_se)[3].split(',')
        seq1_ = re.split(r'\s+', raw_se)[1].split(',')
        anti1=0
        anti2=0
        s1=0
        s2=0
        flag=0
        for i in  secondary_seq2:
            if i=='.':
                anti1+=1
                flag=1
            else:
                if flag==1:
                    anti1+=len(seq2[:int(int(seq2_[0])-1)])
                break    
        for i in  secondary_seq2[::-1]:
            if i=='.':
                flag=-1
                anti2-=1
            else:
                if flag==-1:
                    anti2-=len(seq2[int(seq2_[1]):])
                break  
        flag=0
        for i in  secondary_seq1:
            if i=='.':
                flag=1
                s1+=1
            else:
                if flag==1:
                    s1+=len(seq1[:int(seq1_[0])-1])
                break
        for i in  secondary_seq1[::-1]:
            if i=='.':
                flag=-1
                s2-=1
            else:
                if flag==-1:
                    s2-=len(seq1[int(seq1_[1]):])
                break
        seq2_ = re.split(r'\s+', raw_se)[3].split(',')
        sec_pos = [1000]
        for i in range(len(seq1)):
            sec_pos.append(i)
        for i in range(s2+anti1+len(seq1)-1,s1+anti2-1,-1):
            sec_pos.append(i)
        #if abs(anti1) > 3 or abs(anti2) > 3 or abs(s1) > 3 or abs(s2) > 3:
        #    print('unmatch',data['ID'])
        #    return None
        if len(sec_pos) != len(seq1)+len(seq2)+1:
            print('!=',data['ID'])
            return None
        return sec_pos

    def process(self):
        df=pd.read_excel(self.excel_dir) 
        df['sense raw seq']=df['sense raw seq'].apply(self.raw_pre)
        df['anti raw seq']=df['anti raw seq'].apply(self.raw_pre)

        df['sense seq']=df['sense raw seq'].apply(self.seq_pre)
        df['anti seq']=df['anti raw seq'].apply(self.seq_pre)

        df['start'] = df.apply(self.get_anti_start,axis=1)


        chunks = self.chunk_dataframe(df, self.chunk_size)
        with multiprocessing.Pool(processes=len(chunks)) as pool:
            drops=pool.map(self.get_data, chunks)

        #print(drops)

        #df=df.drop(index=drops)
        if self.json==True:

            dfj=df[['ID','sense seq','sense raw seq','anti seq','anti raw seq','PCT','sense mod','sense pos','anti mod','anti pos','start','cc']] #delect marker

            dfj['pdb_data_path']=dfj['ID'].apply(self.get_path)

            dfj['atom_mask']=dfj.apply(self.get_atommod,axis=1) 
            dfj['smask']=dfj.apply(self.get_smask,axis=1)
            #dfj['rna_pos']=dfj.apply(self.get_rpos,axis=1) 
            
            dfj=dfj.dropna()
            dfj.to_json(self.json_dir, orient='records', lines=True)

    def get_data(self,df):
        
        drops=[]
 
        for index, row in df.iterrows():
            if os.path.exists(f"{self.pdb_dir}/{row['ID']}.pdb")==True:
                continue
            if os.path.exists(f"{self.pdb_dir}/{row['ID']}")==True:
                continue
            if self.secondary_structure ==True:
                if self.get_secondary_structure(row)==False:
                    print(f"drop{self.pdb_dir}/{row['ID']}.pdb")
                    drops.append(index)

            else:
                if self.get_structure(row)==False:
                    print(f"drop{self.pdb_dir}/{row['ID']}.pdb")
                    drops.append(index)

        return drops
    



    

    def get_structure(self,data):
    
        seq1=data['sense seq']
        seq2=data['anti seq']

        pose = assembler.build_init_pose(seq1, seq2)
        pose.dump_pdb(f"{self.pdb_dir}/{data['ID']}.pdb")
        if os.path.exists(f"{self.pdb_dir}/{data['ID']}.pdb"):
            return True
        else:
            return False

    def get_secondary_structure(self,data):
        #if os.path.exists(f"{self.pdb_dir}/{data['ID']}"):
        #    return True
    
        seq1=data['sense seq']
        seq2=data['anti seq']
        seq=seq1+' '+seq2

        #sencondary_seq=RNA.duplexfold(seq1,seq2).structure.replace('&',' ')
        com=f" echo -e \"{seq1}\n{seq2}\n\" | RNAplex "
        raw_se=subprocess.run(com,shell=True,capture_output=True, text=True).stdout

        secondary_seq1 = re.split(r'\s+', raw_se)[0].split('&')[0]
        
        secondary_seq2 = re.split(r'\s+', raw_se)[0].split('&')[1]
        
        
        seq1_ = re.split(r'\s+', raw_se)[1].split(',')
        seq2_ = re.split(r'\s+', raw_se)[3].split(',')

        secondary_seq1 = ''.join(['.' for _ in seq1[:int(seq1_[0])-1]]) + secondary_seq1 + ''.join(['.' for _ in seq1[int(seq1_[1]):]])
        secondary_seq2 = ''.join(['.' for _ in seq2[:int(seq2_[0])-1]]) + secondary_seq2 + ''.join(['.' for _ in seq2[int(seq2_[1]):]])

        secondary_seq = secondary_seq1 + ' ' + secondary_seq2

        os.mkdir(f"{self.pdb_dir}/{data['ID']}")
        os.chdir(f"{self.pdb_dir}/{data['ID']}")

        subprocess.run([FF,'-sequence',seq,'-secstruct',secondary_seq,'-minimize_rna'])
        subprocess.run(['python',EX,'default.out','-rosetta_folder',RF,'1'])
        subprocess.run(['cp','default.out.1.pdb',f"{self.pdb_dir}/{data['ID']}.pdb"])
      
        #os.chdir(f"/public2022/tanwenchong/rna/EnModSIRNA-1main")
        subprocess.run(['rm','-r',f"{self.pdb_dir}/{data['ID']}"])
        

        if os.path.exists(f"{self.pdb_dir}/{data['ID']}.pdb"):
            return True
        else:
            return False

   

def parse():
    parser = argparse.ArgumentParser(description='Process data')
    parser.add_argument('-f','--filenames', nargs='+', help='train/valid/test set')
    parser.add_argument('-p','--pdb_dir', type=str, default=None, help='Path to save processed data')
    return parser.parse_args()

if __name__ == '__main__':
    args = parse()
    for filename in args.filenames:
        Data_Prepare(filename,args.pdb_dir).process()
  

