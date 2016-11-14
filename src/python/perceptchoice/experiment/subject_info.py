from datetime import datetime
import math
import os
import numpy as np
import sys
from perceptchoice.utils import mdm_outliers

LEFT=180
RIGHT=0

coherences=[0.0320, 0.0640, 0.1280, 0.2560, 0.5120]
conditions=['ShamPreCathode','Cathode','ShamPreAnode','Anode']

subject_sessions={
    '1': [
        ['control', 'control', 'control'],
        ['control - anode', 'sham - pre - anode', 'anode'],
        ['control - cathode', 'sham - pre - cathode', 'cathode']
    ],
    '2': [
        ['control', 'control', 'control'],
        ['sham - pre - cathode', 'cathode', 'sham - post - cathode'],
        ['sham - pre - anode', 'anode', 'sham - post - anode']
    ],
    '3': [
        ['control', 'control', 'control'],
        ['sham - pre - anode', 'anode', 'sham - post - anode'],
        ['control - cathode', 'sham - pre - cathode', 'cathode']
    ],
    '4': [
        ['control', 'control', 'control'],
        ['sham - pre - anode', 'anode', 'sham - post - anode'],
        ['sham - pre - cathode', 'cathode', 'sham - post - cathode']
    ],
    '5': [
        ['control', 'control', 'control'],
        ['control - anode', 'sham - pre - anode', 'anode'],
        ['control - cathode', 'sham - pre - cathode', 'cathode']
    ],
    '6': [
        ['control', 'control', 'control'],
        ['control - anode', 'sham - pre - anode', 'anode'],
        ['sham - pre - cathode', 'cathode', 'sham - post - cathode']
    ],
    '7': [
        ['control', 'control', 'control'],
        ['sham - pre - anode', 'anode', 'sham - post - anode'],
        ['control - cathode', 'sham - pre - cathode', 'cathode']
    ],
    '8': [
        ['control', 'control', 'control'],
        ['control - cathode', 'sham - pre - cathode', 'cathode'],
        ['sham - pre - anode', 'anode', 'sham - post - anode']
    ],
    '9': [
        ['control', 'control', 'control'],
        ['control - anode', 'sham - pre - anode', 'anode'],
        ['control - cathode', 'sham - pre - cathode', 'cathode']
    ],
    '10': [
        ['control', 'control', 'control'],
        ['sham - pre - cathode', 'cathode', 'sham - post - cathode'],
        ['sham - pre - anode', 'anode', 'sham - post - anode']
    ],
    '11': [
        ['control', 'control', 'control'],
        ['sham - pre - anode', 'anode', 'sham - post - anode'],
        ['control - cathode', 'sham - pre - cathode', 'cathode']
    ],
    '12': [
        ['control', 'control', 'control'],
        ['sham - pre - anode', 'anode', 'sham - post - anode'],
        ['control - cathode', 'sham - pre - cathode', 'cathode']
    ],
    '13': [
        ['control', 'control', 'control'],
        ['sham - pre - anode', 'anode', 'sham - post - anode'],
        ['control - cathode', 'sham - pre - cathode', 'cathode']
    ],
    '14': [
        ['control', 'control', 'control'],
        ['control - cathode', 'sham - pre - cathode', 'cathode'],
        ['control - anode', 'sham - pre - anode', 'anode']
    ],
    '15': [
        ['control', 'control', 'control'],
        ['control - anode', 'sham - pre - anode', 'anode'],
        ['control - cathode', 'sham - pre - cathode', 'cathode']
    ],
    '16': [
        ['control', 'control', 'control'],
        ['control - cathode', 'sham - pre - cathode', 'cathode'],
        ['sham - pre - anode', 'anode', 'sham - post - anode']
    ],
    '17': [
        ['control', 'control', 'control'],
        ['control - anode', 'sham - pre - anode', 'anode'],
        ['sham - pre - cathode', 'cathode', 'sham - post - cathode']
    ],
    '18': [
        ['control', 'control', 'control'],
        ['control - cathode', 'sham - pre - cathode', 'cathode'],
        ['sham - pre - anode', 'anode', 'sham - post - anode']
    ],
    '19': [
        ['control', 'control', 'control'],
        ['control - cathode', 'sham - pre - cathode', 'cathode'],
        ['control - anode', 'sham - pre - anode', 'anode']
    ],
    '20': [
        ['control', 'control', 'control'],
        ['control - cathode', 'sham - pre - cathode', 'cathode'],
        ['control - anode', 'sham - pre - anode', 'anode']
    ],
    '21': [
        ['control', 'control', 'control'],
        ['control - anode', 'sham - pre - anode', 'anode'],
        ['control - cathode', 'sham - pre - cathode', 'cathode']
    ],
    '22': [
        ['control', 'control', 'control'],
        ['sham - pre - anode', 'anode', 'sham - post - anode'],
        ['control - cathode', 'sham - pre - cathode', 'cathode']
    ],
    '23': [
        ['control', 'control', 'control'],
        ['sham - pre - cathode', 'cathode', 'sham - post - cathode'],
        ['control - anode', 'sham - pre - anode', 'anode']
    ],
    '24': [
        ['control', 'control', 'control'],
        ['sham - pre - cathode', 'cathode', 'sham - post - cathode'],
        ['control - anode', 'sham - pre - anode', 'anode']
    ],
}


class ISISubject:
    def __init__(self, data_dir, subj_id):
        """
        Initialize subject data
        """
        self.data_dir=data_dir
        self.id=subj_id
        self.session_data={}

    def read_data(self):
        # Figure out sessions - look for data files with subject ID in start of the name
        run_files={}
        for file_name in os.listdir(self.data_dir):
            if file_name.lower().endswith('.csv'):
                file_name_parts=file_name.split('.')
                if file_name_parts[0].upper()==self.id:
                    if not file_name_parts[2]=='training':
                        run=int(file_name_parts[2])
                        run_files[run]=file_name
        runs=sorted(run_files.keys())
        # Iterate through subject sessions
        trial_data=[]
        for run in runs:
            file=open(os.path.join(self.data_dir,run_files[run]))

            # Initialize trial data
            # Initialize previous resp, RT and trial coherence
            last_resp=float('NaN')
            last_coherence=float('NaN')
            for line_idx,line in enumerate(file):
                if line_idx>0:
                    cols=line.split(',')

                    # Extract direction
                    direction=int(cols[0])
                    if direction==LEFT:
                        direction=-1
                    elif direction==RIGHT:
                        direction=1
                    coherence=float(cols[1])
                    isi=int(cols[2])
                    correct=int(cols[3])
                    # Determine response based on whether or not correct and the direction
                    resp=direction
                    if correct<1:
                        resp=direction*-1
                        # Convert RT to ms
                    rt=float(cols[4])*1000.0
                    trialIdx=line_idx-1
                    # If a response was made
                    if coherence in coherences and rt<=982:
                        trial_data.append([trialIdx, direction, coherence, correct, resp, last_resp, rt, isi, last_coherence])
                        # Update last trial respnse, RT and coherence
                    last_resp=resp
                    last_coherence=coherence
        # Remove outliers based on RT
        trial_data=np.array(trial_data)
        outliers=mdm_outliers(trial_data[:,6])
        trial_data=trial_data[np.setdiff1d(np.array(range(trial_data.shape[0])),np.array(outliers)),:]
        self.session_data['control']=trial_data

class Subject:

    def __init__(self, data_dir, subj_id):
        """
        Initialize subject data
        """
        self.data_dir=data_dir
        self.id=subj_id
        self.session_data={}

    def read_data(self):
        """
        Read subject data
        """
        # Figure out sessions - look for data files with subject ID in start of the name
        sessions=[]
        for file_name in os.listdir(self.data_dir):
            if file_name.lower().endswith('.csv'):
                file_name_parts=file_name.split('.')
                if file_name_parts[0].upper()==self.id:
                    session_date=datetime.strptime(file_name_parts[1][:11],'%Y_%b_%d')
                    if not session_date in sessions:
                        sessions.append(session_date)
        sessions = sorted(sessions)

        # Iterate through subject sessions
        for idx,session in enumerate(sessions):

            # Look for session file
            for file_name in os.listdir(self.data_dir):
                if file_name.lower().endswith('.csv'):
                    file_name_parts=file_name.split('.')
                    if file_name_parts[0].upper()==self.id:
                        session_date=datetime.strptime(file_name_parts[1][:11],'%Y_%b_%d')                    #
                        if session_date==session and not file_name_parts[2]=='training':

                            # Figure out condition for this session
                            run_num=int(file_name_parts[2])
                            condition=subject_sessions[self.id][idx-1][run_num-1].replace('- ','').title().replace(' ','')

                            # Only need sham and stimulation conditions
                            if not condition=='Control':
                                file=open(os.path.join(self.data_dir,file_name),'r')

                                # Initialize trial data
                                trial_data=[]
                                # Initialize previous resp, RT and trial coherence
                                last_resp=float('NaN')
                                last_rt=float('NaN')
                                last_coherence=float('NaN')
                                for line_idx,line in enumerate(file):
                                    if line_idx>0:
                                        cols=line.split(',')

                                        # Extract direction
                                        direction=int(cols[0])
                                        if direction==LEFT:
                                            direction=-1
                                        elif direction==RIGHT:
                                            direction=1
                                        coherence=float(cols[1])
                                        correct=int(cols[2])
                                        # Determine response based on whether or not correct and the direction
                                        resp=direction
                                        if correct<1:
                                            resp=direction*-1
                                        # Convert RT to ms
                                        rt=float(cols[3])*1000.0
                                        trialIdx=line_idx-1
                                        # Compute ITI
                                        if math.isnan(last_rt) or last_rt>982:
                                            iti=1000
                                        else:
                                            iti=1000+1000-last_rt
                                        # If a response was made
                                        if coherence in coherences and rt<=982:
                                            trial_data.append([trialIdx, direction, coherence, correct, resp, last_resp, rt, iti, last_coherence])
                                        # Update last trial respnse, RT and coherence
                                        last_resp=resp
                                        last_rt=rt
                                        last_coherence=coherence
                                # Remove outliers based on RT
                                trial_data=np.array(trial_data)
                                outliers=mdm_outliers(trial_data[:,6])
                                trial_data=trial_data[np.setdiff1d(np.array(range(trial_data.shape[0])),np.array(outliers)),:]
                                self.session_data[condition]=trial_data



def read_subjects(data_dir):
    subjects={}
    for subj_id in subject_sessions:
        subjects[subj_id]=Subject(data_dir, subj_id)
        subjects[subj_id].read_data()
    return subjects

def read_isi_subjects(data_dir):
    subjects={}
    for subj_id in range(1,25):
        # Exclude 16 because overall accuracy threshold is >.25 (its .54!)
        if not subj_id==16:
            subjects[subj_id]=ISISubject(data_dir, '%d' % subj_id)
            subjects[subj_id].read_data()
    return subjects
